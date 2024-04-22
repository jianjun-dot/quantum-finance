import numpy as np
from qfinance.helper import define_covariance_matrix
from qfinance.qArithmetic import QComp
from tqdm import tqdm
from datetime import datetime

import numpy as np
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import DraperQFTAdder
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_algorithms import EstimationProblem
from qfinance.ModifiedIQAE.mod_iae_updated import ModifiedIterativeAmplitudeEstimation
from qiskit_aer.primitives import Sampler
from datetime import datetime
from time import time
from tqdm.auto import tqdm

from qfinance.utils.tools import results_to_JSON, save_JSON, time_convert, save_meta_data

# Get the current date and time
current_datetime = datetime.now()
# Convert to string using strftime
date = current_datetime.strftime("%Y-%m-%d")
time_string = current_datetime.strftime("%H:%M:%S")
print("Starting at: ", date, time_string)

############# set asset parameters ##############
# number of qubits to represent the uncertainty for our probability distribution
num_uncertainty_qubits = 3
# parameters for considered random distribution
strike_name= "call_on_max"
#### asset 1 ####
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity
###########
correlation = 0.2
############## Estimation parameters ##############
c_approx = 0.1
epsilon = 0.001
alpha = 0.005
N_shots = 1000
n_trials = 3
n_step = 10
use_GPU=True
discount=True
##################################################

num_qubits_for_each_dimension = num_uncertainty_qubits + 1
# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

sigma = vol * np.sqrt(T)

# map to higher dimensional distribution
# for simplicity assuming dimensions are independent and identically distributed)
dimension = 2
num_qubits = [num_uncertainty_qubits] * dimension
low = low * np.ones(dimension)
high = high * np.ones(dimension)
mu = mu * np.ones(dimension)

cov = define_covariance_matrix(sigma**2, sigma**2, correlation)

uncertainty_model = LogNormalDistribution(
    num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high))
)
discount_factor = np.exp(-r * T)

low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

strike_prices = [round(step/n_step * (0.9*high-1.1*low) + 1.1*low, 2) for step in range(n_step)]


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
        "n_steps": n_step,
    }
}
save_meta_data(strike_name, date, time_string, meta_data)

start_time = time()
sampler = Sampler(run_options={"shots": 1000})
all_results = {}
for (index, strike_price) in tqdm(enumerate(strike_prices), leave=False):
    all_results[strike_price] = {}
    # set the approximation scaling for the payoff function
    

    # setup piecewise linear objective fcuntion
    breakpoints = [low, strike_price] 
    # low is the lower bound, strike price is where our payoff function starts to increase
    slopes = [0, 1]
    # can be float or list of floats.
    # for list of floats, the floats are the slopes of the individual linear functions

    offsets = [0, 0]
    # the offsets of each linear function
    f_min = 0
    # minimum y value
    f_max = high - strike_price
    # maximum y value

    european_call_objective = LinearAmplitudeFunction(
        num_uncertainty_qubits,
        slopes,
        offsets,
        domain=(low, high),
        image=(f_min, f_max),
        breakpoints=breakpoints,
        rescaling_factor=c_approx,
    )
    call_objective = european_call_objective.to_gate()
    controlled_objective = call_objective.control(1)
    
    bit_length = num_uncertainty_qubits+1

    carry_register = QuantumRegister(1, name='c')
    second_carry_register = QuantumRegister(1, name='c\'')
    first_number_register = QuantumRegister(bit_length, name='a')
    second_number_register = QuantumRegister(bit_length, name='b')
    objective_register = QuantumRegister(1, name='objective')
    ancilla_register = QuantumRegister(3, name='ancilla')
    second_ancilla_register = QuantumRegister(num_uncertainty_qubits, name='objective_ancilla')

    adder = DraperQFTAdder(bit_length, kind="half")

    circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, second_carry_register, objective_register, ancilla_register, second_ancilla_register)
    circuit.append(uncertainty_model, first_number_register[:-1]+ second_number_register[:-1])
    qcomp = QComp(bit_length,bit_length)
    circuit.append(qcomp, first_number_register[:] + second_number_register[:] + carry_register[:] + ancilla_register[:])

    circuit.append(adder, first_number_register[:] + second_number_register[:]+ second_carry_register[:])
    circuit.append(controlled_objective, [ancilla_register[0]]+ second_number_register[:-1] + objective_register[:]+ second_ancilla_register[:])
    circuit.append(controlled_objective, [ancilla_register[1]]+ first_number_register[:-1] + objective_register[:] + second_ancilla_register[:])
    circuit.append(controlled_objective, [ancilla_register[2]]+ first_number_register[:-1] + objective_register[:] + second_ancilla_register[:])
    
    exact_value = 0
    for i in range(len(uncertainty_model.probabilities)):
        exact_value += uncertainty_model.probabilities[i]*max(0, max(uncertainty_model.values[i][0], uncertainty_model.values[i][1])-strike_price)

    problem = EstimationProblem(
        state_preparation=circuit,
        objective_qubits=[2*num_uncertainty_qubits+4],
        post_processing=european_call_objective.post_processing,
    )
    # construct amplitude estimation
    all_key_results = []
    for i in tqdm(range(n_trials), leave=False):
        # qi = QuantumInstance(backend=AerSimulator(), shots=200)
        # ae = ModifiedIterativeAmplitudeEstimation(
        #     epsilon_target=epsilon, alpha=alpha, quantum_instance=qi)
        # result = ae.estimate(problem, shots=200)
        ae = ModifiedIterativeAmplitudeEstimation(
                epsilon_target=epsilon, alpha=alpha, sampler=sampler)
        result = ae.estimate(problem, shots=N_shots, use_GPU=use_GPU)
        conf_int = list(result.confidence_interval_processed)
        curr_estimate = result.estimation_processed
        if discount:
            curr_estimate *= discount_factor
            conf_int[0] *= discount_factor
            conf_int[1] *= discount_factor
        all_key_results.append(
            [exact_value, curr_estimate, conf_int[0], conf_int[1], result.num_oracle_queries]
        )
        all_results[strike_price]["test_{}_full_results".format(i)] = results_to_JSON(result)
    all_results[strike_price]["results"] = all_key_results

save_JSON(strike_name, date, time_string,all_results)
stop_time = time()
stop_date_time = datetime.now()
print("Ended at: ", stop_date_time.strftime("%Y-%m-%d"), stop_date_time.strftime("%H:%M:%S"))
print(f"Execution time: {time_convert(stop_time - start_time)}")
