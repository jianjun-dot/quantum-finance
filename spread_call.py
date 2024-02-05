import numpy as np
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    AncillaRegister,
)
from qiskit.circuit.library import VBERippleCarryAdder
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_algorithms import EstimationProblem
from qiskit_algorithms import IterativeAmplitudeEstimation
from qiskit_aer.primitives import Sampler

from datetime import datetime
import csv
import json
from tqdm.auto import tqdm

############
n_trials = 10
#########
##### call
# strike_prices = [0.01, 0.03]
strike_prices = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
# correction_ratio = [0.45699766,  0.29203295, -0.06446088, -0.10473188,  0.03728036,  0.00996379, 0.07142962,  0.64954655,  5.47081125]

# Get the current date and time
current_datetime = datetime.now()

# Convert to string using strftime
date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

num_uncertainty_qubits = 3
num_qubits_for_each_dimension = num_uncertainty_qubits + 1

# parameters for considered random distribution
S = 0.5  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.04  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)


# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = 0
high = mean + 3 * stddev

# map to higher dimensional distribution
# for simplicity assuming dimensions are independent and identically distributed)
dimension = 2
num_qubits = [num_uncertainty_qubits] * dimension
low = low * np.ones(dimension)
high = high * np.ones(dimension)
mu = mu * np.ones(dimension)
cov = sigma**2 * np.eye(dimension)  # covariance matrix

# construct circuit
u = LogNormalDistribution(
    num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high))
)

low_ = low[0]
high_ = high[0]
step = high_/7

def define_spread_option(strike_price, index):
    # set the strike price (should be within the low and the high value of the uncertainty)
    # map strike price from [low, high] to {0, ..., 2^n-1}

    mapped_strike_price = (
        (
            (strike_price - low_)
            / (high_ - low_)
            * (2 ** (num_qubits_for_each_dimension) - 1)
        )
        + 2 ** (num_uncertainty_qubits)
        - 1
    )

    # print("mapped strike price: {}".format(mapped_strike_price))
    # mapped_strike_price = mapped_strike_price * (1-correction_ratio[index])
    # set the approximation scaling for the payoff function
    c_approx = 0.05

    # setup piecewise linear objective fcuntion
    breakpoints = [-8 *step, strike_price]
    slopes = [0, 1]
    offsets = [0, 0]

    f_min = 0
    f_max = high_ - strike_price

    spread_objective = LinearAmplitudeFunction(
        num_qubits_for_each_dimension,
        slopes,
        offsets,
        domain=(-8 *step, high_),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )
    return spread_objective, mapped_strike_price


############# adder

firstRegister = QuantumRegister(num_qubits_for_each_dimension, "first")
secondRegister = QuantumRegister(num_qubits_for_each_dimension, "second")
carryRegister = QuantumRegister(1, "carry")
ancillaRegister = QuantumRegister(num_qubits_for_each_dimension, "ancilla")

adder = VBERippleCarryAdder(num_qubits_for_each_dimension, name="Adder")
num_qubits = len(adder.qubits)

all_results = {}
for (index, strike_price) in tqdm(enumerate(strike_prices), leave=False):
    circ = QuantumCircuit(
        carryRegister, firstRegister, secondRegister, ancillaRegister, name="subtractor"
    )
    circ.x(secondRegister)
    circ.x(carryRegister)
    circ.append(adder, list(range(num_qubits)))

    epsilon = 0.001
    alpha = 0.005

    spread_objective, mapped_strike_price = define_spread_option(strike_price, index)
    ###### objective
    firstRegister = QuantumRegister(num_qubits_for_each_dimension, "first")
    secondRegister = QuantumRegister(num_qubits_for_each_dimension, "second")
    objectiveRegister = QuantumRegister(1, "obj")
    carryRegister = QuantumRegister(1, "carry")
    ancillaRegister = AncillaRegister(
        max(num_qubits_for_each_dimension, spread_objective.num_ancillas), "ancilla"
    )
    optionAncillaRegister = AncillaRegister(
        spread_objective.num_ancillas, "optionAncilla"
    )

    spread_option = QuantumCircuit(
        carryRegister,
        firstRegister,
        secondRegister,
        objectiveRegister,
        ancillaRegister,
        optionAncillaRegister,
    )
    spread_option.append(u, firstRegister[:-1] + secondRegister[:-1])
    spread_option.append(
        circ,
        carryRegister[:] + firstRegister[:] + secondRegister[:] + ancillaRegister[:],
    )
    spread_option.x(secondRegister[-1])
    spread_option.append(
        spread_objective,
        secondRegister[:] + objectiveRegister[:] + optionAncillaRegister[:],
    )

    objective_index = num_qubits_for_each_dimension

    problem = EstimationProblem(
        state_preparation=spread_option,
        objective_qubits=[9],
        post_processing=spread_objective.post_processing,
    )

    # evaluate exact expected value
    sum_values = np.array([v[0] - v[1] for v in u.values])
    exact_value = np.dot(
        u.probabilities[sum_values >= strike_price],
        sum_values[sum_values >= strike_price] - strike_price,
    )

    all_conf_ints = []
    for i in tqdm(range(n_trials), leave=False):
        ae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon,
            alpha=alpha,
            sampler=Sampler(run_options={"shots": 50000}),
        )
        # construct amplitude estimation
        result = ae.estimate(problem)

        all_conf_ints.append(
            [exact_value, result.estimation_processed, result.confidence_interval_processed[0], result.confidence_interval_processed[1]]
        )
        # print("Exact value:        \t%.4f" % exact_value)
        # print("Estimated value:    \t%.4f" % (estimated_value))
    all_results[strike_price] = {"results": all_conf_ints}


with open("spread_call_test_{}.json".format(date_time_string), "w") as file:
    json.dump(all_results, file)

# with open("spread_call_test_{}.csv".format(date_time_string), "a", newline="") as csvfile:
#     writer = csv.writer(csvfile, delimiter=",")
#     for conf_int in all_conf_ints:
#         writer.writerow([exact_value, conf_int[0], conf_int[1], conf_int[2]])
