import numpy as np
from qiskit_finance.circuit.library import LogNormalDistribution
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae import (
    ModifiedIterativeAmplitudeEstimation,
)
from qiskit.circuit.library import WeightedAdder, LinearAmplitudeFunction
from qiskit_algorithms import EstimationProblem
from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit
import random


class BasketOptionParameters:
    def __init__(self, days_to_maturity, strike_price):
        self.days_to_maturity = days_to_maturity  # time to maturity in days
        self.strike_price = strike_price  # set the strike price (should be within the low and the high value of the uncertainty)
        self.T = days_to_maturity / 365  # time to maturity in years


class StockParameters:
    def __init__(self, S, vol, r):
        self.S = S  # initial spot price
        self.vol = vol  # volatility
        self.r = r  # annual interest rate


def estimate_basket_option_price(
    option_parameters: BasketOptionParameters,
    stock_one_parameters: StockParameters,
    stock_two_parameters: StockParameters,
    epsilon=0.01,
    alpha=0.05,
    num_uncertainty_qubits=3,
):
    # parameters for options
    days_to_maturity = option_parameters.days_to_maturity
    strike_price = option_parameters.strike_price
    T = days_to_maturity / 365

    # parameters for first distribution
    S1 = stock_one_parameters.S
    vol1 = stock_one_parameters.vol
    r1 = stock_one_parameters.r

    # parameters for second distribution
    S2 = stock_two_parameters.S
    vol2 = stock_two_parameters.vol
    r2 = stock_two_parameters.r

    # log-normal variables for first distribution
    mu1 = (r1 - 0.5 * vol1**2) * T + np.log(S1)
    sigma1 = vol1 * np.sqrt(T)
    mean1 = np.exp(mu1 + sigma1**2 / 2)
    variance1 = (np.exp(sigma1**2) - 1) * np.exp(2 * mu1 + sigma1**2)
    stddev1 = np.sqrt(variance1)

    # log-normal variables for second distribution
    mu2 = (r2 - 0.5 * vol2**2) * T + np.log(S2)
    sigma2 = vol2 * np.sqrt(T)
    mean2 = np.exp(mu2 + sigma2**2 / 2)
    variance2 = (np.exp(sigma2**2) - 1) * np.exp(2 * mu2 + sigma2**2)
    stddev2 = np.sqrt(variance2)

    # lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
    low = np.maximum(0, min(mean1 - 3 * stddev1, mean2 - 3 * stddev2))
    high = max(mean1 + 3 * stddev1, mean2 + 3 * stddev2)

    # map to higher dimensional distribution
    dimension = 2
    num_qubits = [num_uncertainty_qubits] * dimension
    low = low * np.ones(dimension)
    high = high * np.ones(dimension)
    mu = np.array([mu1, mu2])
    cov = np.diag([sigma1, sigma2])  # covariance matrix
    cov[0, 1] = sigma1 * sigma2 * 0.7
    cov[1, 0] = sigma1 * sigma2 * 0.7

    # construct circuit
    u = LogNormalDistribution(
        num_qubits=num_qubits, mu=mu, sigma=cov, bounds=list(zip(low, high))
    )

    # determine number of qubits required to represent total loss
    weights = []
    for n in num_qubits:
        for i in range(n):
            weights += [2**i]

    agg = WeightedAdder(sum(num_qubits), weights)  # integer weights only
    n_s = agg.num_sum_qubits
    n_aux = agg.num_qubits - n_s - agg.num_state_qubits  # number of additional qubits

    # map strike price from [low, high] to {0, ..., 2^n-1}
    max_value = 2**n_s - 1
    low_ = low[0]
    high_ = high[0]
    mapped_strike_price = (
        (strike_price - dimension * low_)
        / (high_ - low_)
        * (2**num_uncertainty_qubits - 1)
    )

    # set the approximation scaling for the payoff function
    c_approx = 0.25

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
    qr_state = QuantumRegister(
        u.num_qubits, "state"
    )  # to load the probability distribution
    qr_obj = QuantumRegister(1, "obj")  # to encode the function values
    ar_sum = AncillaRegister(n_s, "sum")  # number of qubits used to encode the sum
    ar = AncillaRegister(
        max(n_aux, basket_objective.num_ancillas), "work"
    )  # additional qubits

    objective_index = u.num_qubits

    basket_option = QuantumCircuit(qr_state, qr_obj, ar_sum, ar)
    basket_option.append(u, qr_state)
    basket_option.append(agg, qr_state[:] + ar_sum[:] + ar[:n_aux])
    basket_option.append(
        basket_objective, ar_sum[:] + qr_obj[:] + ar[: basket_objective.num_ancillas]
    )

    # evaluate exact expected value
    sum_values = np.sum(u.values, axis=1)
    exact_value = np.dot(
        u.probabilities[sum_values >= strike_price],
        sum_values[sum_values >= strike_price] - strike_price,
    )

    problem = EstimationProblem(
        state_preparation=basket_option,
        objective_qubits=[objective_index],
        post_processing=basket_objective.post_processing,
    )
    # construct amplitude estimation

    qi = QuantumInstance(backend=AerSimulator(), shots=200)
    ae = ModifiedIterativeAmplitudeEstimation(
        epsilon_target=epsilon, alpha=alpha, quantum_instance=qi
    )
    result = ae.estimate(problem, shots=200)

    conf_int = (
        np.array(result.confidence_interval_processed)
        / (2**num_uncertainty_qubits - 1)
        * (high_ - low_)
    )

    estimated_value = (
        result.estimation_processed / (2**num_uncertainty_qubits - 1) * (high_ - low_)
    )
    print("Exact value:        \t%.4f" % exact_value)
    print("Estimated value:    \t%.4f" % (estimated_value))
    print("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))

    return exact_value, estimated_value, conf_int


if __name__ == "__main__":
    
    # current basket
    basket = BasketOptionParameters(40, 3.5)
    
    # go through a list of different values for stock 1 and 2
    spot_prices = [2.0, 2.2, 2.4, 2.5]
    volatilities = [0.2, 0.3, 0.4, 0.5]
    interest_rates = [0.03, 0.04, 0.05, 0.06]
    
    all_test_cases = []
    
    for spot_price in spot_prices:
        for volatility in volatilities:
            for interest_rate in interest_rates:
                stock_params = StockParameters(spot_price, volatility, interest_rate)
                all_test_cases.append(stock_params)

    # Determine how many 2-tuples you want to generate
    num_tests = 5

    # Generate a random set of 2-tuples with replacement
    random_tests = [(random.choice(all_test_cases), random.choice(all_test_cases)) for _ in range(num_tests)]
    for index, test_case in enumerate(random_tests):
        results = estimate_basket_option_price(basket, test_case[0], test_case[1])
        if results[2][0] <= results[0] <= results[2][1]:
            print("-------------------------------------------------")
            print("Test case {} passed".format(index))
            print("-------------------------------------------------")

    
    