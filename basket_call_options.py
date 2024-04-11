import numpy as np
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.circuit.library import WeightedAdder, LinearAmplitudeFunction
from qiskit_algorithms import EstimationProblem
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler
from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae_updated import ModifiedIterativeAmplitudeEstimation
from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
from qfinance.helper import define_covariance_matrix

from time import time
from datetime import datetime
import json
from tqdm.auto import tqdm

from qfinance.utils.tools import results_to_JSON, save_JSON, save_meta_data, time_convert

# Get the current date and time
current_datetime = datetime.now()
# Convert to string using strftime
date = current_datetime.strftime("%Y-%m-%d")
time_string = current_datetime.strftime("%H:%M:%S")
print("Starting at: ", date, time_string)

##### number of qubits per dimension to represent the uncertainty #############
num_uncertainty_qubits = 3
strike_name = "basket_call"
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.04  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity
######################
correlation = 0.2
####### Experiment parameters ######## 
c_approx = 0.1
epsilon = 0.001
alpha = 0.005
n_trials = 3
n_steps = 10
N_shots = 1000
use_GPU = False
#################

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
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
uncertainty_model = LogNormalDistribution(num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high)))
# u = NormalDistribution(num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high)))


# determine number of qubits required to represent total loss
weights = []
for n in num_qubits:
    for i in range(n):
        weights += [2**i]
# we can see that weights is according to the binary representation of the number

agg = WeightedAdder(sum(num_qubits), weights) # integer weights only
n_s = agg.num_sum_qubits
n_aux = agg.num_qubits - n_s - agg.num_state_qubits  # number of additional qubits

low_float = np.maximum(0, mean - 3 * stddev)
high_float = mean + 3 * stddev


strike_prices = [round(step/n_steps * (0.7*2*high_float-1.5*2*low_float) + 1.5*2*low_float, 2) for step in range(n_steps)]


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
        "lower_bound": low,
        "upper_bound": high,
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

sampler = Sampler(run_options={"shots": 1000})
all_results = {}
max_value = 2**n_s - 1
start_time = time()
for (index, strike_price) in enumerate(strike_prices):
    curr_start_time = time()
    # map strike price from [low, high] to {0, ..., 2^n-1}
    low_ = low[0]
    high_ = high[0]
    mapped_strike_price = (
        (strike_price - dimension * low_) / (high_ - low_) * (2**num_uncertainty_qubits - 1)
    )
    # set the approximation scaling for the payoff function

    # setup piecewise linear objective fcuntion
    breakpoints = [0, mapped_strike_price]
    slopes = [0, 1]
    offsets = [0, 0]
    f_min = 0
    f_max = 2 * (2**num_uncertainty_qubits - 1) - mapped_strike_price
    basket_objective = LinearAmplitudeFunction(
        n_s,
        slopes,
        offsets,
        domain=(0, max_value),
        image=(f_min, f_max),
        rescaling_factor=c_approx,
        breakpoints=breakpoints,
    )


    # define overall multivariate problem
    qr_state = QuantumRegister(uncertainty_model.num_qubits, "state")  # to load the probability distribution
    qr_obj = QuantumRegister(1, "obj")  # to encode the function values
    ar_sum = AncillaRegister(n_s, "sum")  # number of qubits used to encode the sum
    ar = AncillaRegister(max(n_aux, basket_objective.num_ancillas), "work")  # additional qubits

    objective_index = uncertainty_model.num_qubits

    basket_option = QuantumCircuit(qr_state, qr_obj, ar_sum, ar)
    basket_option.append(uncertainty_model, qr_state)
    basket_option.append(agg, qr_state[:] + ar_sum[:] + ar[:n_aux])
    basket_option.append(basket_objective, ar_sum[:] + qr_obj[:] + ar[: basket_objective.num_ancillas])

    # evaluate exact expected value
    sum_values = np.sum(uncertainty_model.values, axis=1)
    exact_value = np.dot(
        uncertainty_model.probabilities[sum_values >= strike_price],
        sum_values[sum_values >= strike_price] - strike_price,
    )
    problem = EstimationProblem(
        state_preparation=basket_option,
        objective_qubits=[objective_index],
        post_processing=basket_objective.post_processing,
    )
    all_key_results = []
    inner_loop_start_time = time()
    all_results[strike_price] = {}
    for i in range(n_trials):
        # qi = QuantumInstance(backend=AerSimulator(), shots=200)
        # ae = ModifiedIterativeAmplitudeEstimation(
        #     epsilon_target=epsilon, alpha=alpha, quantum_instance=qi)
        # result = ae.estimate(problem, shots=200)
        ae = ModifiedIterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, sampler=sampler)
        result = ae.estimate(problem, shots=N_shots, use_GPU=use_GPU)
        conf_int = list(
            np.array(result.confidence_interval_processed)
            / (2**num_uncertainty_qubits - 1)
            * (high_ - low_)
        )
        curr_estimate = result.estimation_processed / (2**num_uncertainty_qubits - 1) * (high_ - low_)
        all_key_results.append(
            [exact_value, curr_estimate, conf_int[0], conf_int[1], result.num_oracle_queries]
        )
        all_results[strike_price]["test_{}_full_results".format(i)] = results_to_JSON(result)
        curr_inner_loop_end_time = time()
        print([exact_value, curr_estimate, conf_int[0], conf_int[1], result.num_oracle_queries])
        print(f"Inner Loop {i+1} time elapsed: {time_convert(curr_inner_loop_end_time - inner_loop_start_time)}, average time per loop: {time_convert((curr_inner_loop_end_time - inner_loop_start_time)/(i+1))}")
    all_results[strike_price]["results"] = all_key_results
    curr_end_time = time()
    print(f"Strike Price {strike_price} time elapsed: {time_convert(curr_end_time - curr_start_time)}, average time per loop: {time_convert((curr_end_time - curr_start_time)/(index+1))}")
    
save_JSON(strike_name, date, time_string,all_results)
stop_time = time()
stop_date_time = datetime.now()
print("Ended at: ", stop_date_time.strftime("%Y-%m-%d"), stop_date_time.strftime("%H:%M:%S"))
print(f"Execution time: {time_convert(stop_time - start_time)}")

    