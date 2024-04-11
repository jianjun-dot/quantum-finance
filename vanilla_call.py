import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
# from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae import ModifiedIterativeAmplitudeEstimation
from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae_updated import ModifiedIterativeAmplitudeEstimation
from tqdm.auto import tqdm
from datetime import datetime
from qfinance.utils.tools import results_to_JSON, save_JSON, save_meta_data, time_convert
from qiskit_aer.primitives import Sampler
from time import time
import sys

def main():
    
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
    strike_name= "vanilla_call"
    S = 2.0  # initial spot price
    vol = 0.4  # volatility of 40%
    r = 0.05  # annual interest rate of 4%
    T = 40 / 365  # 40 days to maturity
    ############## Estimation parameters ##############
    c_approx = 0.1 # rescaling factor for the piecewise linear objective function
    epsilon = 0.001 # target accuracy
    alpha = 0.005 # confidence level
    N_shots = 1000 # number of shots for each evaluation of the quantum circuit
    n_trials = 2 # number of independent evaluations of the quantum circuit
    n_step = 3 # number of distinct strike prices to consider
    use_GPU = False # whether to use GPU for the quantum circuit simulation
    ##################################################

    # resulting parameters for log-normal distribution
    mu = (r - 0.5 * vol**2) * T + np.log(S)
    sigma = vol * np.sqrt(T)
    mean = np.exp(mu + sigma**2 / 2)
    variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
    stddev = np.sqrt(variance)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low = np.maximum(0, mean - 3 * stddev)
    high = mean + 3 * stddev

    uncertainty_model = LogNormalDistribution(
        num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
    )
    
    strike_prices = [round(step/n_step * (0.9*high-1.1*low) + 1.1*low, 2) for step in range(n_step)]
    
    meta_data = {
        "asset_params":{
            "asset_name": strike_name,
            "initial_spot_price": S,
            "volatility": vol,
            "annual_interest_rate": r,
            "days_to_maturity": T,
            "mu": mu,
            "sigma_T": sigma,
            "mu_S": mean,
            "sigma_S": stddev,
            "num_uncertainty_qubits": num_uncertainty_qubits,
            "lower_bound": low,
            "upper_bound": high,
            "correlation": "N/A",
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
    
    ### Pay off function
    
    start_time = time()

    all_results = {}
    sampler = Sampler(run_options={"shots": 1000})
    # use_GPU = sys.argv[1]
    # use_GPU = use_GPU == 'True'
    for (index, strike_price) in tqdm(enumerate(strike_prices), leave=False):
        all_results[strike_price] = {}
        # setup piecewise linear objective fcuntion
        breakpoints = [low, strike_price] 
        # low is the lower bound, strike price is where our payoff function starts to increase
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = high - strike_price

        european_call_objective = LinearAmplitudeFunction(
            num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(low, high),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        num_qubits = european_call_objective.num_qubits
        european_call = QuantumCircuit(num_qubits)
        european_call.append(uncertainty_model, range(num_uncertainty_qubits))
        european_call.append(european_call_objective, range(num_qubits))

        problem = EstimationProblem(
            state_preparation=european_call,
            objective_qubits=[num_uncertainty_qubits],
            post_processing=european_call_objective.post_processing,
        )
        exact_value = np.dot(
            uncertainty_model.probabilities[uncertainty_model.values >= strike_price], uncertainty_model.values[uncertainty_model.values >= strike_price] - strike_price
        )
        estimation_results = []
        for i in tqdm(range(n_trials), leave=False):
            # qi = QuantumInstance(backend=AerSimulator(), shots=1000)
            # ae = ModifiedIterativeAmplitudeEstimation(
            #     epsilon_target=epsilon, alpha=alpha, quantum_instance=qi)
            ae = ModifiedIterativeAmplitudeEstimation(
                epsilon_target=epsilon, alpha=alpha, sampler=sampler)
            result = ae.estimate(problem, shots=N_shots, use_GPU=use_GPU)
            estimation_results.append(
                [exact_value, result.estimation_processed, result.confidence_interval_processed[0], result.confidence_interval_processed[1], result.num_oracle_queries]
            )
            # print([exact_value, result.estimation_processed, result.confidence_interval_processed[0], result.confidence_interval_processed[1], result.num_oracle_queries])
            all_results[strike_price]["test_{}_full_results".format(i)] = results_to_JSON(result)
        all_results[strike_price]["results"] = estimation_results
        # print(all_results[strike_price]["results"])

    save_JSON(strike_name, date, time_string,all_results)
    stop_time = time()
    stop_date_time = datetime.now()
    print("Ended at: ", stop_date_time.strftime("%Y-%m-%d"), stop_date_time.strftime("%H:%M:%S"))
    print(f"Execution time: {time_convert(stop_time - start_time)}")


if __name__ == "__main__":
    main()