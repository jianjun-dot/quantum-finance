import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import (
    LogNormalDistribution,
    NormalDistribution,
    UniformDistribution,
)
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae import (
    ModifiedIterativeAmplitudeEstimation,
)
from tqdm.auto import tqdm

##############
# uncertainty model
##############

# number of qubits to represent the uncertainty for our probability distribution
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 2.0  # initial spot price
vol = 0.4  # volatility of 40%
r = 0.05  # annual interest rate of 4%
T = 40 / 365  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)

##################
# payoff functions
##################

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1.896
# set the approximation scaling for the payoff function
c_approx = 0.25
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

call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective

####################
# construct circuits
####################

num_qubits = call_objective.num_qubits
call = QuantumCircuit(num_qubits)
call.append(uncertainty_model, range(num_uncertainty_qubits))
call.append(call_objective, range(num_qubits))


####################
# create problem
####################

problem = EstimationProblem(
    state_preparation=call,
    objective_qubits=[3],
    post_processing=call_objective.post_processing,
)

####################
# define target errors and confidence level
####################
epsilon = 0.01  # error threshold
alpha = 0.05  # significance level


epsilon_targets = np.linspace(0.001, 0.01, 10)
alpha_targets = np.linspace(0.01, 0.05, 5)

x = uncertainty_model.values
y = np.maximum(0, x - strike_price)
exact_value = np.dot(uncertainty_model.probabilities, y)

all_results = []
####################
# run algorithm
####################

qi = QuantumInstance(backend=AerSimulator())
oracle_calls = []

for epsilon in tqdm(epsilon_targets, desc="epsilon", disable=True):
    for alpha in alpha_targets:
        ae = ModifiedIterativeAmplitudeEstimation(
            epsilon_target=epsilon, alpha=alpha, quantum_instance=qi
        )
        result = ae.estimate(problem, shots=100)
        conf_int = np.array(result.confidence_interval_processed)
        estimated_value = result.estimation_processed
        all_results.append(
            [epsilon, alpha, estimated_value, conf_int[0], conf_int[1], exact_value]
        )
        oracle_calls.append([epsilon, alpha, result.num_oracle_queries])
        print("epsilon:", epsilon, "alpha:", alpha, "estimated_value:", estimated_value, "exact_value:", exact_value)


####################
# save results
####################
all_results = np.array(all_results)
np.savetxt(
    "results.csv",
    all_results,
    delimiter=",",
    header="epsilon,alpha,estimated_value,lower_bound,upper_bound,exact_value",
)

oracle_calls = np.array(oracle_calls)
np.savetxt(
    "oracle_calls.csv",
    oracle_calls,
    delimiter=",",
    header="epsilon,alpha,num_oracle_queries",
)

####################
# plot results
####################
plt.figure(figsize=(10, 10))
plt.xlabel("epsilon")
plt.ylabel("alpha")
plt.title("Estimated value")
plt.scatter(all_results[:, 0], all_results[:, 1], c=np.abs(all_results[:, 2] - all_results[:,-1]), cmap="RdYlBu")
plt.colorbar()
plt.savefig("estimated_value.png")
