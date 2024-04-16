from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import EstimationProblem, MaximumLikelihoodAmplitudeEstimation
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import DraperQFTAdder
import numpy as np
from time import time
from datetime import datetime
from tqdm.auto import tqdm

from qfinance.ModifiedIQAE.mod_iae_updated import ModifiedIterativeAmplitudeEstimation
from qfinance.helper import define_covariance_matrix, loadNumber
from qfinance.qArithmetic import QComp
from qfinance.qArithmetic import subtractorDraper
from qfinance.utils.tools import results_to_JSON, save_JSON, time_convert, save_meta_data

# Get the current date and time
current_datetime = datetime.now()
# Convert to string using strftime
date = current_datetime.strftime("%Y-%m-%d")
time_string = current_datetime.strftime("%H:%M:%S")
print("Starting at: ", date, time_string)

#################
# number of qubits per dimension to represent the uncertainty
num_uncertainty_qubits = 3
##################
# parameters for considered random distribution
strike_name = "best_of_call"
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.04  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity
##################
correlation = 0.2
###############
c_approx = 0.01
epsilon = 0.0005
alpha = 0.005
n_trials = 5
n_steps = 10
N_shots = 1000
use_GPU = True
###############

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
strike_prices = [round(step/n_steps * (2*stddev) + mean - stddev, 2) for step in range(n_steps)]
# construct circuit
uncertainty_model = LogNormalDistribution(num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high)))

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

comparator = QComp(num_uncertainty_qubits+1, num_uncertainty_qubits+1)
adder = DraperQFTAdder(num_uncertainty_qubits, kind="half")
subtractor = subtractorDraper(num_uncertainty_qubits)


def map_strike_price_int(strike_price, high, low, num_uncertainty_qubits):
    return int(
        np.ceil((strike_price - low) / (high - low) * (2**num_uncertainty_qubits - 1))
    )

def map_strike_price_float(strike_price, high, low, num_uncertainty_qubits):
    return (strike_price - low) / (high - low) * (2**num_uncertainty_qubits - 1)

def objective_function_two_strike(
    num_uncertainty_qubits, strike_price_1, strike_price_2, c_approx=0.01
):

    # map strike prices
    mapped_strike_price_1_float = map_strike_price_float(
        strike_price_1, high, low, num_uncertainty_qubits
    )
    mapped_strike_price_1_int = map_strike_price_int(
        strike_price_1, high, low, num_uncertainty_qubits
    )

    mapped_strike_price_2_float = map_strike_price_float(
        strike_price_2, high, low, num_uncertainty_qubits
    )
    mapped_strike_price_2_int = map_strike_price_int(
        strike_price_2, high, low, num_uncertainty_qubits
    )

    step = (np.pi / 6) / (2**num_uncertainty_qubits - 1) * c_approx
    # determine offset
    offset_1 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_1_int - 1)) / 2
    offset_2 = (2**num_uncertainty_qubits - 1 - (mapped_strike_price_2_int - 1)) / 2

    offset = np.mean([offset_1, offset_2])
    # offset = 0
    offset_angle = step * offset * 2

    # create first payoff function
    model_register = QuantumRegister(num_uncertainty_qubits, "model")
    objective_register = QuantumRegister(1, "objective")

    circuit_one = QuantumCircuit(model_register, objective_register)
    circuit_one.ry(-step * mapped_strike_price_1_float * 2, objective_register[0])
    for i in range(num_uncertainty_qubits):
        circuit_one.cry(step * 2 ** (i + 1), model_register[i], objective_register[0])

    # Create second_payoff_function
    circuit_two = QuantumCircuit(model_register, objective_register)
    circuit_two.ry(-step * mapped_strike_price_2_float * 2, objective_register[0])
    for i in range(num_uncertainty_qubits):
        circuit_two.cry(step * 2 ** (i + 1), model_register[i], objective_register[0])

    def post_processor(prob_of_one):
        coeff = (prob_of_one - 0.5) / (step) + offset
        fmax = high - min(strike_price_1, strike_price_2)
        og_range = (
            2**num_uncertainty_qubits
            - 1
            - min(mapped_strike_price_1_float, mapped_strike_price_2_float)
        )
        return coeff / og_range * (fmax)

    return (
        circuit_one.to_gate(label="F1"),
        circuit_two.to_gate(label="F2"),
        post_processor,
        offset_angle,
    )

all_mean_estimates = []
all_std = []

all_results = {}
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


# exc = ThreadPoolExecutor(max_workers=4)
start_time = time()
sampler = Sampler(run_options={"shots": 1000})
save_meta_data(strike_name, date, time_string, meta_data)
# fix second strike price
strike_price_2 = strike_prices[int(len(strike_prices)/2)]
for (index1, strike_price_1) in enumerate(strike_prices):
    curr_start_time = time()
    first_var_register = QuantumRegister(num_uncertainty_qubits, "var1")
    first_ancilla_register = QuantumRegister(num_uncertainty_qubits, "ancilla1")
    first_carry_register = QuantumRegister(1, "carry")

    second_var_register = QuantumRegister(num_uncertainty_qubits, "var2")
    second_ancilla_register = QuantumRegister(num_uncertainty_qubits, "ancilla2")
    second_carry_register = QuantumRegister(1, "carry2")
    second_carry_register_copy = QuantumRegister(1, "carry2_copy")

    comparator_carry = QuantumRegister(1, "comparator_carry")
    comparator_ancilla = QuantumRegister(3, "comparator_ancilla")

    adder_ancilla = QuantumRegister(1, "adder_ancilla")

    objective_register = QuantumRegister(1, "objective")
    circuit = QuantumCircuit(
        first_var_register,
        first_ancilla_register,
        first_carry_register,
        second_var_register,
        second_ancilla_register,
        second_carry_register,
        second_carry_register_copy,
        comparator_carry,
        comparator_ancilla,
        objective_register,
    )

    # objective_fn, post_processor = basic_function_loader(num_uncertainty_qubits, c_approx)
    objective_1, objective_2, post_processor, offset_angle = (
        objective_function_two_strike(
            num_uncertainty_qubits, strike_price_1, strike_price_2, c_approx
        )
    )

    controlled_objective_1 = objective_1.control(num_ctrl_qubits=2, ctrl_state="10")
    controlled_objective_2 = objective_2.control(num_ctrl_qubits=2, ctrl_state="10")

    circuit.append(uncertainty_model, first_var_register[:] + second_var_register[:])

    # set sin(x) to sin(x+pi/4)
    circuit.ry(np.pi / 2, objective_register)
    # offset
    circuit.ry(-offset_angle, objective_register[0])

    # load strike prices
    loadNumber(
        circuit,
        first_ancilla_register,
        map_strike_price_int(strike_price_1, high, low, num_uncertainty_qubits),
    )
    loadNumber(
        circuit,
        second_ancilla_register,
        map_strike_price_int(strike_price_2, high, low, num_uncertainty_qubits),
    )
    circuit.append(
        subtractor,
        first_var_register[:] + first_ancilla_register[:] + [first_carry_register[0]],
    )
    circuit.append(
        subtractor,
        second_var_register[:]
        + second_ancilla_register[:]
        + [second_carry_register[0]],
    )
    circuit.cx(second_carry_register[0], second_carry_register_copy[0])
    circuit.append(
        comparator,
        first_ancilla_register[:]
        + first_carry_register[:]
        + second_ancilla_register[:]
        + second_carry_register[:]
        + comparator_carry[:]
        + comparator_ancilla[:],
    )
    circuit.append(
        controlled_objective_2,
        second_carry_register_copy[:]
        + [comparator_ancilla[0]]
        + second_var_register[:]
        + objective_register[:],
    )
    circuit.append(
        controlled_objective_1,
        first_carry_register[:]
        + [comparator_ancilla[1]]
        + first_var_register[:]
        + objective_register[:],
    )
    circuit.append(
        controlled_objective_1,
        first_carry_register[:]
        + [comparator_ancilla[2]]
        + first_var_register[:]
        + objective_register[:],
    )
    problem = EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[circuit.num_qubits - 1],
        post_processing=post_processor,
    )
    
    curr_exact_expectation = 0
    for i in range(len(uncertainty_model.probabilities)):
        curr_exact_expectation += uncertainty_model.probabilities[i] * max(0, uncertainty_model.values[i][0] - strike_price_1, uncertainty_model.values[i][1] - strike_price_2)
    all_key_results = []
    curr_dict = all_results.get(strike_price_1,{})
    curr_dict[strike_price_2] = {}
    for i in range(n_trials):
        ae = ModifiedIterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, sampler=sampler)
        result = ae.estimate(problem, shots=N_shots, use_GPU=use_GPU)
        curr_results = [curr_exact_expectation, result.estimation_processed, result.confidence_interval_processed[0], result.confidence_interval_processed[1], result.num_oracle_queries]
        all_key_results.append(curr_results)
        curr_dict[strike_price_2]["test_{}_full_results".format(i)] = results_to_JSON(result)
    curr_dict[strike_price_2]["results"] = all_key_results
    all_results[strike_price_1] = curr_dict
    curr_end_time = time()
    print(curr_results)
    print(f"[{index1+1}][{strike_price_1}, {strike_price_2}] time elapsed: {time_convert(curr_end_time - curr_start_time)}, average time per loop: {time_convert((curr_end_time - start_time)/((index1+1)))}")
    print()
    end_time = time()

strike_price_1 = strike_prices[int(len(strike_prices)/2)]
for (index2, strike_price_2) in enumerate(strike_prices):
    curr_start_time = time()
    first_var_register = QuantumRegister(num_uncertainty_qubits, "var1")
    first_ancilla_register = QuantumRegister(num_uncertainty_qubits, "ancilla1")
    first_carry_register = QuantumRegister(1, "carry")

    second_var_register = QuantumRegister(num_uncertainty_qubits, "var2")
    second_ancilla_register = QuantumRegister(num_uncertainty_qubits, "ancilla2")
    second_carry_register = QuantumRegister(1, "carry2")
    second_carry_register_copy = QuantumRegister(1, "carry2_copy")

    comparator_carry = QuantumRegister(1, "comparator_carry")
    comparator_ancilla = QuantumRegister(3, "comparator_ancilla")

    adder_ancilla = QuantumRegister(1, "adder_ancilla")

    objective_register = QuantumRegister(1, "objective")
    circuit = QuantumCircuit(
        first_var_register,
        first_ancilla_register,
        first_carry_register,
        second_var_register,
        second_ancilla_register,
        second_carry_register,
        second_carry_register_copy,
        comparator_carry,
        comparator_ancilla,
        objective_register,
    )

    # objective_fn, post_processor = basic_function_loader(num_uncertainty_qubits, c_approx)
    objective_1, objective_2, post_processor, offset_angle = (
        objective_function_two_strike(
            num_uncertainty_qubits, strike_price_1, strike_price_2, c_approx
        )
    )

    controlled_objective_1 = objective_1.control(num_ctrl_qubits=2, ctrl_state="10")
    controlled_objective_2 = objective_2.control(num_ctrl_qubits=2, ctrl_state="10")

    circuit.append(uncertainty_model, first_var_register[:] + second_var_register[:])

    # set sin(x) to sin(x+pi/4)
    circuit.ry(np.pi / 2, objective_register)
    # offset
    circuit.ry(-offset_angle, objective_register[0])

    # load strike prices
    loadNumber(
        circuit,
        first_ancilla_register,
        map_strike_price_int(strike_price_1, high, low, num_uncertainty_qubits),
    )
    loadNumber(
        circuit,
        second_ancilla_register,
        map_strike_price_int(strike_price_2, high, low, num_uncertainty_qubits),
    )
    circuit.append(
        subtractor,
        first_var_register[:] + first_ancilla_register[:] + [first_carry_register[0]],
    )
    circuit.append(
        subtractor,
        second_var_register[:]
        + second_ancilla_register[:]
        + [second_carry_register[0]],
    )
    circuit.cx(second_carry_register[0], second_carry_register_copy[0])
    circuit.append(
        comparator,
        first_ancilla_register[:]
        + first_carry_register[:]
        + second_ancilla_register[:]
        + second_carry_register[:]
        + comparator_carry[:]
        + comparator_ancilla[:],
    )
    circuit.append(
        controlled_objective_2,
        second_carry_register_copy[:]
        + [comparator_ancilla[0]]
        + second_var_register[:]
        + objective_register[:],
    )
    circuit.append(
        controlled_objective_1,
        first_carry_register[:]
        + [comparator_ancilla[1]]
        + first_var_register[:]
        + objective_register[:],
    )
    circuit.append(
        controlled_objective_1,
        first_carry_register[:]
        + [comparator_ancilla[2]]
        + first_var_register[:]
        + objective_register[:],
    )
    problem = EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[circuit.num_qubits - 1],
        post_processing=post_processor,
    )
    
    curr_exact_expectation = 0
    for i in range(len(uncertainty_model.probabilities)):
        curr_exact_expectation += uncertainty_model.probabilities[i] * max(0, uncertainty_model.values[i][0] - strike_price_1, uncertainty_model.values[i][1] - strike_price_2)
    all_key_results = []
    curr_dict = all_results.get(strike_price_1,{})
    curr_dict[strike_price_2] = {}
    for i in range(n_trials):
        ae = ModifiedIterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, sampler=sampler)
        result = ae.estimate(problem, shots=N_shots, use_GPU=use_GPU)
        curr_results = [curr_exact_expectation, result.estimation_processed, result.confidence_interval_processed[0], result.confidence_interval_processed[1], result.num_oracle_queries]
        all_key_results.append(curr_results)
        curr_dict[strike_price_2]["test_{}_full_results".format(i)] = results_to_JSON(result)
    curr_dict[strike_price_2]["results"] = all_key_results
    all_results[strike_price_1] = curr_dict
    curr_end_time = time()
    print(curr_results)
    print(f"[{len(strike_prices)+index2+1}][{strike_price_1}, {strike_price_2}] time elapsed: {time_convert(curr_end_time - curr_start_time)}, average time per loop: {time_convert((curr_end_time - start_time)/((len(strike_prices)+index2+1)))}")
    print()
    end_time = time()

save_JSON(strike_name, date, time_string,all_results)
stop_time = time()
stop_date_time = datetime.now()
meta_data["execution_time"] = time_convert(stop_time - start_time)
save_meta_data(strike_name, date, time_string, meta_data)
print("Ended at: ", stop_date_time.strftime("%Y-%m-%d"), stop_date_time.strftime("%H:%M:%S"))
print(f"Execution time: {time_convert(stop_time - start_time)}")