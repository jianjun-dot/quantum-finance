import numpy as np
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    AncillaRegister,
)
from qiskit.circuit.library import DraperQFTAdder
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_algorithms import EstimationProblem
from qiskit_aer.primitives import Sampler

from qiskit_aer.primitives import Sampler
from qfinance.ModifiedIQAE.mod_iae_updated import ModifiedIterativeAmplitudeEstimation
from qfinance.utils.tools import results_to_JSON, save_JSON, save_meta_data, time_convert
from qfinance.qArithmetic import oneIncrement
from qfinance.helper import define_covariance_matrix

from datetime import datetime
from time import time
from tqdm.auto import tqdm


# Get the current date and time
current_datetime = datetime.now()
# Convert to string using strftime
date = current_datetime.strftime("%Y-%m-%d")
time_string = current_datetime.strftime("%H:%M:%S")
print("Starting at: ", date, time_string)

############# set asset parameters ##############
num_uncertainty_qubits = 3
###########################
# parameters for considered random distribution
strike_name= "spread_call"
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.04  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity
###########################
correlation = 0.2
###########################
c_approx = 0.05
epsilon = 0.001
alpha = 0.005
n_trials = 5
n_steps = 10
N_shots = 1000
use_GPU = True
discount = True
#########


##### call
# strike_prices = [0.01, 0.03]
# strike_prices = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

strike_prices = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035,0.04, 0.045,0.05, 0.055]

num_qubits_for_each_dimension = num_uncertainty_qubits + 1
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
cov = define_covariance_matrix(sigma**2, sigma**2, correlation)

# construct circuit
uncertainty_model = LogNormalDistribution(
    num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high))
)

discount_factor = np.exp(-r * T)

low_ = low[0]
high_ = high[0]
step = high_/(2**num_uncertainty_qubits-1)

def define_spread_option(strike_price):
    # setup piecewise linear objective fcuntion
    breakpoints = [-2**num_uncertainty_qubits *step, strike_price]
    slopes = [0, 1]
    offsets = [0, 0]

    f_min = 0
    f_max = high_ - strike_price

    spread_objective = LinearAmplitudeFunction(
        num_qubits_for_each_dimension,
        slopes,
        offsets,
        domain=(-2**num_uncertainty_qubits *step, high_),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )
    return spread_objective

meta_data = {
    "asset_params":{
        "asset_name": strike_name,
        "initial_spot_price": S,
        "volatility": vol,
        "annual_interest_rate": r,
        "days_to_maturity": T,
        "mu": list(mu),
        "sigma_T": sigma,
        "mu_S": mean,
        "sigma_S": stddev,
        "num_uncertainty_qubits": num_uncertainty_qubits,
        "lower_bound": list(low),
        "upper_bound": list(high),
        "correlation": correlation,
    },
    "date": date,
    "time": time_string,
    "expt_params": {
        "n_trials": n_trials,
        "epsilon": epsilon,
        "alpha": alpha,
        "N_shots": N_shots,
        "c_approx": c_approx,
        "use_GPU": use_GPU,
        "strike_prices": strike_prices,
        "n_steps": n_steps,
    }
}
save_meta_data(strike_name, date, time_string, meta_data)


start_time = time()

firstRegister = QuantumRegister(num_uncertainty_qubits, "first")
secondRegister = QuantumRegister(num_uncertainty_qubits, "second")
carryRegister = QuantumRegister(1, "carry")

#############
adder = DraperQFTAdder(num_uncertainty_qubits, kind="half", name="Adder")

sampler = Sampler(run_options={"shots": 1000})
all_results = {}
for (index, strike_price) in tqdm(enumerate(strike_prices), leave=False):
    all_results[strike_price] = {}
    ### build subtractor
    circ = QuantumCircuit(
        firstRegister, secondRegister, carryRegister, name="subtractor"
    )
    circ.x(secondRegister[:]+ carryRegister[:])
    circ.append(adder, firstRegister[:] + secondRegister[:] + carryRegister[:])
    circ.append(oneIncrement(num_qubits_for_each_dimension), secondRegister[:]+ carryRegister[:])
    circ.x(carryRegister[:])

    spread_objective = define_spread_option(strike_price)
    ###### objective
    firstRegister = QuantumRegister(num_uncertainty_qubits, "first")
    secondRegister = QuantumRegister(num_uncertainty_qubits, "second")
    objectiveRegister = QuantumRegister(1, "obj")
    carryRegister = QuantumRegister(1, "carry")
    optionAncillaRegister = AncillaRegister(
        spread_objective.num_ancillas, "optionAncilla"
    )

    spread_option = QuantumCircuit(
        firstRegister,
        secondRegister,
        carryRegister,
        objectiveRegister,
        optionAncillaRegister,
    )
    spread_option.append(u, firstRegister[:] + secondRegister[:])
    spread_option.append(
        circ,
        firstRegister[:] + secondRegister[:] + carryRegister[:],
    )
    spread_option.append(
        spread_objective,
        secondRegister[:] + carryRegister[:]+ objectiveRegister[:] + optionAncillaRegister[:],
    )

    objective_index = 2*num_uncertainty_qubits + 1

    problem = EstimationProblem(
        state_preparation=spread_option,
        objective_qubits=[objective_index],
        post_processing=spread_objective.post_processing,
    )

    # evaluate exact expected value
    sum_values = np.array([v[0] - v[1] for v in u.values])
    exact_value = np.dot(
        uncertainty_model.probabilities[sum_values >= strike_price],
        sum_values[sum_values >= strike_price] - strike_price,
    )

    all_conf_ints = []
    # qi = QuantumInstance(backend=AerSimulator(), shots=1000)
    for i in tqdm(range(n_trials), leave=False):
        ae = ModifiedIterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, sampler=sampler)
        result = ae.estimate(problem, shots=N_shots, use_GPU=use_GPU)
        conf_int = list(result.confidence_interval_processed)
        curr_estimate = result.estimation_processed
        if discount:
            exact_value *= discount_factor
            curr_estimate *= discount_factor
            conf_int[0] *= discount_factor
            conf_int[1] *= discount_factor
        all_conf_ints.append(
            [exact_value, curr_estimate, conf_int[0], conf_int[1], result.num_oracle_queries]
        )
        all_results[strike_price]["test_{}_full_results".format(i)] = results_to_JSON(result)
    all_results[strike_price]["results"] = all_conf_ints

save_JSON(strike_name, date, time_string,all_results)
stop_time = time()
stop_date_time = datetime.now()

meta_data["execution_time"] = time_convert(stop_time - start_time)
save_meta_data(strike_name, date, time_string, meta_data)

print("Ended at: ", stop_date_time.strftime("%Y-%m-%d"), stop_date_time.strftime("%H:%M:%S"))
print(f"Execution time: {time_convert(stop_time - start_time)}")
