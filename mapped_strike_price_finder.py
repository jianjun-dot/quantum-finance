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

# Get the current date and time
current_datetime = datetime.now()

# Convert to string using strftime
date_time_string = current_datetime.strftime("%Y-%m-%d-%H:%M:%S")

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

############# adder

firstRegister = QuantumRegister(num_qubits_for_each_dimension, "first")
secondRegister = QuantumRegister(num_qubits_for_each_dimension, "second")
carryRegister = QuantumRegister(1, "carry")
ancillaRegister = QuantumRegister(num_qubits_for_each_dimension, "ancilla")

adder = VBERippleCarryAdder(num_qubits_for_each_dimension, name="Adder")
num_qubits = len(adder.qubits)

circ = QuantumCircuit(
    carryRegister, firstRegister, secondRegister, ancillaRegister, name="subtractor"
)
circ.x(secondRegister)
circ.x(carryRegister)
circ.append(adder, list(range(num_qubits)))
circ.draw()


##### call

error_tolerance = 0.00001

error_accumulated = 1
# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 0.05

# map strike price from [low, high] to {0, ..., 2^n-1}
max_value = 2 ** (num_qubits_for_each_dimension) - 1
low_ = low[0]
high_ = high[0]


# evaluate exact expected value
sum_values = np.array([v[0] - v[1] for v in u.values])
exact_value = np.dot(
    u.probabilities[sum_values >= strike_price],
    sum_values[sum_values >= strike_price] - strike_price,
)

    # set target precision and confidence level
    ############
n_trials = 20
    #########

upper_mapped_strike_price = 8.1
lower_mapped_strike_price = 7.9
num_iter = 0
max_iter = 30
while error_accumulated > error_tolerance and num_iter < max_iter:

    mapped_strike_price = (upper_mapped_strike_price + lower_mapped_strike_price) / 2
    
    print("current iteration: {}".format(num_iter+1))
    print("mapped strike price: {}".format(mapped_strike_price))

    # set the approximation scaling for the payoff function
    c_approx = 0.25

    # setup piecewise linear objective fcuntion
    breakpoints = [0, mapped_strike_price]
    slopes = [0, 1]
    offsets = [0, 0]

    f_min = 0
    f_max = max_value - mapped_strike_price

    spread_objective = LinearAmplitudeFunction(
        num_qubits_for_each_dimension,
        slopes,
        offsets,
        domain=(0, max_value),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )


    ###### objective
    firstRegister = QuantumRegister(num_qubits_for_each_dimension, "first")
    secondRegister = QuantumRegister(num_qubits_for_each_dimension, "second")
    objectiveRegister = QuantumRegister(1, "obj")
    carryRegister = QuantumRegister(1, "carry")
    ancillaRegister = AncillaRegister(
        max(num_qubits_for_each_dimension, spread_objective.num_ancillas), "ancilla"
    )
    optionAncillaRegister = AncillaRegister(spread_objective.num_ancillas, "optionAncilla")

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
        circ, carryRegister[:] + firstRegister[:] + secondRegister[:] + ancillaRegister[:]
    )
    spread_option.x(secondRegister[-1])
    spread_option.append(
        spread_objective,
        secondRegister[:] + objectiveRegister[:] + optionAncillaRegister[:],
    )

    objective_index = num_qubits_for_each_dimension
    ###### QAE

    epsilon = 0.001
    alpha = 0.005

    problem = EstimationProblem(
        state_preparation=spread_option,
        objective_qubits=[9],
        post_processing=spread_objective.post_processing,
    )
    estimated_values = []
    for i in range(n_trials):
        ae = IterativeAmplitudeEstimation(
            epsilon_target=epsilon,
            alpha=alpha,
            sampler=Sampler(run_options={"shots": 10000}),
        )
        # construct amplitude estimation
        result = ae.estimate(problem)

        estimated_value = (
            result.estimation_processed
            / (2 ** (num_qubits_for_each_dimension) - 1)
            * (high_ - low_)
        )
        
        estimated_values.append(estimated_value)
    
    mean_value = np.mean(estimated_values)
    error = mean_value - exact_value
    error_accumulated = np.abs(error)
    print("error: {}".format(error))
    if error > 0:
        lower_mapped_strike_price = mapped_strike_price
    else:
        upper_mapped_strike_price = mapped_strike_price
    num_iter += 1
    
    
with open("strike_price_finder_{}.txt".format(date_time_string), "a") as f:
    f.write("final strike price: {}, error: {}, iterations: {}".format(mapped_strike_price, error_accumulated, num_iter))
    f.close()