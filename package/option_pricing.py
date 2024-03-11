import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction, WeightedAdder, DraperQFTAdder
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution, NormalDistribution, UniformDistribution
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae import ModifiedIterativeAmplitudeEstimation
from typing import List, Union
from scipy.interpolate import griddata
from .qArithmetic import QComp, subtractor


############
# option input should be a JSON object
###########
'''
option_params = {
    'option_type'[str]: call, call, basket call, spread call, call-on-max, call-on-min, best-of-call
    # this defines the type of option that we will be pricing
    'option_params'[dict]:{
        'r'[float]: 0.04, # annual interest rate of 4%
        'vol'[float]: 0.4, # volatility of 40%
        'T'[float]: 40/365, # 40 days to maturity
        'S'[float]: 0.5, # initial spot price
        'strike_price'[float]: 0.01, # strike price
    }
}
'''


class OptionParams():
    def __init__(self, num_uncertainty_qubits: int, option_type):
        self.individual_params = []
        self.cov = None
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.strike_prices = None
        self.option_type = option_type
        
    def define_strike_prices(self, strike_prices: Union[List[float], float]):
        if isinstance(strike_prices, float):
            self.strike_prices = [strike_prices]
        else:
            self.strike_prices = strike_prices
         
    def define_covariance_matrix(self, cov: np.ndarray):
        self.cov = cov
    
    def add_variable(self, variable: Union[list, dict]):
        if type(variable) is dict:
            variable = self._process_variable(variable)
            self.individual_params.append(variable)
        elif type(variable) is list:
            for var in variable:
                curr_variable = self._process_variable(var)
                self.individual_params.append(curr_variable)
    
    def _process_variable(self, variable: dict):
        mu = (variable['r'] - 0.5 * variable['vol']**2)* variable['T'] + np.log(variable['S'])
        sigma = variable['vol'] * np.sqrt(variable['T'])
        mean = np.exp(mu + sigma**2/2)
        std = np.sqrt((np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2))
        if self.option_type == 'spread call':
            low = 0
        else:
            low = np.maximum(0, mean - 3*std)
        high = mean + 3*std
        variable['mu'] = mu
        variable['sigma'] = sigma
        variable['mean'] = mean
        variable['std'] = std
        variable['low'] = low
        variable['high'] = high
        return variable

class OptionPricing():
    
    def __init__(self, options_params: OptionParams):
        self.num_variables = len(options_params.individual_params)
        self.num_uncertainty_qubits = options_params.num_uncertainty_qubits
        self.options_params = options_params
        self.objective_index = None
        self.option = None
        self.objective = None
        self.strike_prices = options_params.strike_prices
        self.define_uncertainty_model(options_params)
        # self._define_payoff_function(options_params.option_type, options_params.strike_prices)
        self.option_type = options_params.option_type
    
    def define_uncertainty_model(self, option_params: OptionParams):
        self.all_variables = option_params.individual_params
        self.dimension = len(self.all_variables)
        if self.dimension == 1:
            lower_bound = self.all_variables[0]['low']
            upper_bound = self.all_variables[0]['high']
            mu = self.all_variables[0]['mu']
            std = self.all_variables[0]['std']
            self.uncertainty_model = LogNormalDistribution(num_qubits=self.num_uncertainty_qubits, mu=mu, sigma=std**2, bounds=(lower_bound, upper_bound))
        else:
            self.num_dist_qubits = [self.num_uncertainty_qubits] * self.dimension
            lower_bound = np.array([var['low'] for var in self.all_variables])
            upper_bound = np.array([var['high'] for var in self.all_variables])
            mu = np.array([var['mu'] for var in self.all_variables])
            if option_params.cov is None:
                cov = np.diag([var['sigma']**2 for var in self.all_variables])
            else:
                cov = option_params.cov
            self.uncertainty_model = LogNormalDistribution(num_qubits=self.num_dist_qubits, mu=mu, sigma=cov, bounds=list(zip(lower_bound, upper_bound)))

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        
    def _define_payoff_function(self, option_type: str, strike_price: list[float], c_approx=0.125):
        if option_type == 'call':
            strike_price = strike_price[0]
            self._define_basic_call_options(strike_price, c_approx)
        elif option_type == 'basket call':
            strike_price = strike_price[0]
            self._define_basket_call_options(strike_price, c_approx)
        elif option_type == 'spread call':
            strike_price = strike_price[0]
            self._define_spread_call_options(strike_price, c_approx=0.01)
        elif option_type == 'call-on-max':
            strike_price = strike_price[0]
            self._define_call_on_max_options(strike_price, c_approx)
        elif option_type == 'call-on-min':
            strike_price = strike_price[0]
            self._define_call_on_min_options(strike_price, c_approx)
        elif option_type == 'best-of-call':
            self._define_best_of_call_options(strike_price, c_approx)
        else:
            raise Exception("Option type not defined!")
    
    def _define_basic_call_options(self, strike_price: float, c_approx=0.125):
        params = self.options_params.individual_params[0]
        
        self.high = params['high']
        self.low = params['low']
        
        breakpoints = [params['low'], strike_price]
        slopes = [0,1]
        offsets = [0,0]
        f_min = 0
        f_max = params['high'] - strike_price
        
        call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(params['low'], params['high']),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        self.strike_price = strike_price
        
        num_qubits = call_objective.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.append(self.uncertainty_model, range(self.uncertainty_model.num_qubits))
        circuit.append(call_objective, range(call_objective.num_qubits))
        
        self.objective = call_objective
        self.objective_index = self.uncertainty_model.num_qubits
        self.option = circuit
    
    def _define_basket_call_options(self, strike_price: float, c_approx=0.125):
        weights = []
        num_qubits = [self.num_uncertainty_qubits] * self.dimension
        for n in num_qubits:
            for i in range(n):
                weights += [2**i]
        
        agg_circuit = WeightedAdder(sum(num_qubits), weights=weights)
        n_sum_qubits = agg_circuit.num_sum_qubits
        n_ancilla = agg_circuit.num_qubits - n_sum_qubits - agg_circuit.num_state_qubits
        # set the strike price (should be within the low and the high value of the uncertainty)
        
        # map strike price from [low, high] to {0, ..., 2^n-1}
        max_value = 2**n_sum_qubits - 1
        self.low = self.lower_bound[0]
        self.high = self.upper_bound[0]
        mapped_strike_price = (
            (strike_price - self.dimension * self.low) / (self.high - self.low) * (2**self.num_uncertainty_qubits - 1)
        )
        # setup piecewise linear objective fcuntion
        breakpoints = [0, mapped_strike_price]
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = 2 * (2**self.num_uncertainty_qubits - 1) - mapped_strike_price
        
        basket_objective = LinearAmplitudeFunction(
            n_sum_qubits,
            slopes,
            offsets,
            domain=(0, max_value),
            image=(f_min, f_max),
            rescaling_factor=c_approx,
            breakpoints=breakpoints,
        )
        self.strike_price = strike_price
        
        qr_state = QuantumRegister(self.uncertainty_model.num_qubits, "state")  # to load the probability distribution
        qr_obj = QuantumRegister(1, "obj")  # to encode the function values
        ar_sum = AncillaRegister(n_sum_qubits, "sum")  # number of qubits used to encode the sum
        ar = AncillaRegister(max(n_ancilla, basket_objective.num_ancillas), "work")  # additional qubits

        objective_index = self.uncertainty_model.num_qubits

        basket_option = QuantumCircuit(qr_state, qr_obj, ar_sum, ar)
        basket_option.append(self.uncertainty_model, qr_state)
        basket_option.append(agg_circuit, qr_state[:] + ar_sum[:] + ar[:n_ancilla])
        basket_option.append(basket_objective, ar_sum[:] + qr_obj[:] + ar[: basket_objective.num_ancillas])
        
        self.objective = basket_objective
        self.objective_index = objective_index
        self.option = basket_option

    def _define_spread_call_options(self, strike_price: float, c_approx=0.01):
        params = self.options_params.individual_params[0]
        num_qubits_for_each_dimension = self.num_uncertainty_qubits+1
        low_ = params['low']
        high_ = params['high']
        self.strike_price = strike_price
        # print(self.num_uncertainty_qubits)
        # print(low_)
        # print(high_)
        step = high_/(2**self.num_uncertainty_qubits-1)
        # print(step)
        # print("domain: {}".format([-2**self.num_uncertainty_qubits *step, high_]))

        # setup piecewise linear objective fcuntion
        breakpoints = [-2**self.num_uncertainty_qubits*step, strike_price]
        slopes = [0, 1]
        offsets = [0, 0]

        f_min = 0
        f_max = high_ - strike_price
        spread_objective = LinearAmplitudeFunction(
            num_qubits_for_each_dimension,
            slopes,
            offsets,
            domain=(-2**self.num_uncertainty_qubits *step, high_),
            image=(f_min, f_max),
            rescaling_factor=c_approx,
            breakpoints=breakpoints,
        )
        
        firstRegister = QuantumRegister(num_qubits_for_each_dimension, 'first')
        secondRegister = QuantumRegister(num_qubits_for_each_dimension, 'second')
        objectiveRegister = QuantumRegister(1, 'obj')
        carryRegister = QuantumRegister(1, 'carry')
        ancillaRegister = AncillaRegister(max(num_qubits_for_each_dimension, spread_objective.num_ancillas), 'ancilla')
        optionAncillaRegister = AncillaRegister(spread_objective.num_ancillas, 'optionAncilla')

        subtract_circuit = subtractor(num_qubits_for_each_dimension, num_qubits_for_each_dimension)

        spread_option = QuantumCircuit(carryRegister, firstRegister, secondRegister,objectiveRegister, ancillaRegister,  optionAncillaRegister)
        spread_option.append(self.uncertainty_model, firstRegister[:-1] + secondRegister[:-1])
        spread_option.append(subtract_circuit, carryRegister[:] + firstRegister[:]+ secondRegister[:] + ancillaRegister[:])
        spread_option.x(secondRegister[-1])
        spread_option.append(spread_objective, secondRegister[:] + objectiveRegister[:] + optionAncillaRegister[:])
        objective_index = num_qubits_for_each_dimension * 2 + 1
        print(objective_index)
        self.objective = spread_objective
        self.objective_index = objective_index
        self.option = spread_option
        
    def _define_call_on_max_options(self, strike_price: float, c_approx=0.125):
        params = self.options_params.individual_params[0]
        self.high = params['high']
        self.low = params['low']
        self.strike_price = strike_price

        breakpoints = [self.low, strike_price] 
        slopes = [0, 1]
        offsets = [0, 0]
        f_min = 0
        f_max = self.high - strike_price

        european_call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(self.low, self.high),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        call_objective = european_call_objective.to_gate()
        controlled_objective = call_objective.control(1)
        
        bit_length = self.num_uncertainty_qubits+1

        carry_register = QuantumRegister(1, name='c')
        second_carry_register = QuantumRegister(1, name='c\'')
        first_number_register = QuantumRegister(bit_length, name='a')
        second_number_register = QuantumRegister(bit_length, name='b')
        objective_register = QuantumRegister(1, name='objective')
        ancilla_register = QuantumRegister(3, name='ancilla')
        second_ancilla_register = QuantumRegister(self.num_uncertainty_qubits, name='objective_ancilla')

        adder = DraperQFTAdder(bit_length, kind="half")
        circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, second_carry_register, objective_register, ancilla_register, second_ancilla_register)
        circuit.append(self.uncertainty_model, first_number_register[:-1] + second_number_register[:-1])
        qcomp = QComp(bit_length,bit_length)
        circuit.append(qcomp, first_number_register[:] + second_number_register[:] + carry_register[:] + ancilla_register[:])

        circuit.append(adder, first_number_register[:] + second_number_register[:]+ second_carry_register[:])
        circuit.append(controlled_objective, [ancilla_register[0]]+ second_number_register[:-1] + objective_register[:]+ second_ancilla_register[:])
        circuit.append(controlled_objective, [ancilla_register[1]]+ first_number_register[:-1] + objective_register[:] + second_ancilla_register[:])
        circuit.append(controlled_objective, [ancilla_register[2]]+ first_number_register[:-1] + objective_register[:] + second_ancilla_register[:])

        self.objective = european_call_objective
        self.objective_index = self.uncertainty_model.num_qubits + 4
        self.option = circuit
    
    def _define_call_on_min_options(self, strike_price: float, c_approx=0.125):
        params = self.options_params.individual_params[0]
        self.high = params['high']
        self.low = params['low']
        self.strike_price = strike_price

        breakpoints = [self.low, strike_price] 
        slopes = [-1, 0]
        offsets = [strike_price - self.low, 0]
        f_min = 0
        f_max = strike_price - self.low

        european_call_objective = LinearAmplitudeFunction(
            self.num_uncertainty_qubits,
            slopes,
            offsets,
            domain=(self.low, self.high),
            image=(f_min, f_max),
            breakpoints=breakpoints,
            rescaling_factor=c_approx,
        )
        call_objective = european_call_objective.to_gate()
        controlled_objective = call_objective.control(1)
        
        bit_length = self.num_uncertainty_qubits+1

        carry_register = QuantumRegister(1, name='c')
        second_carry_register = QuantumRegister(1, name='c\'')
        first_number_register = QuantumRegister(bit_length, name='a')
        second_number_register = QuantumRegister(bit_length, name='b')
        objective_register = QuantumRegister(1, name='objective')
        ancilla_register = QuantumRegister(3, name='ancilla')
        second_ancilla_register = QuantumRegister(self.num_uncertainty_qubits, name='objective_ancilla')

        adder = DraperQFTAdder(bit_length, kind="half")
        circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, second_carry_register, objective_register, ancilla_register, second_ancilla_register)
        circuit.append(self.uncertainty_model, first_number_register[:-1] + second_number_register[:-1])
        qcomp = QComp(bit_length,bit_length)
        circuit.append(qcomp, first_number_register[:] + second_number_register[:] + carry_register[:] + ancilla_register[:])
        circuit.append(adder, first_number_register[:] + second_number_register[:]+ second_carry_register[:])
        circuit.append(controlled_objective, [ancilla_register[0]]+ first_number_register[:-1] + objective_register[:]+ second_ancilla_register[:])
        circuit.append(controlled_objective, [ancilla_register[1]]+ second_number_register[:-1] + objective_register[:] + second_ancilla_register[:])
        circuit.append(controlled_objective, [ancilla_register[2]]+ second_number_register[:-1] + objective_register[:] + second_ancilla_register[:])

        self.objective = european_call_objective
        self.objective_index = self.uncertainty_model.num_qubits + 4
        self.option = circuit
    
    def _define_best_of_call_options(self):
        pass
    
    def create_state_prep_circuit(self):
        num_qubits = self.objective.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.append(self.uncertainty_model, range(self.uncertainty_model.num_qubits))
        circuit.append(self.objective, range(self.objective.num_qubits))
        self.objective_index = self.uncertainty_model.num_qubits
        self.option = circuit
        return circuit
    
    def create_estimation_problem(self, epsilon=0.01):
        scaling_param = np.sqrt(epsilon)
        self._define_payoff_function(self.options_params.option_type, self.options_params.strike_prices, c_approx=scaling_param)
        self.option_type = self.options_params.option_type
        # check if pre-requisites are satisfied
        if self.objective_index is None:
            raise Exception("Objective index not defined yet!")
        if self.option is None:
            raise Exception("Option not defined yet!")
        if self.objective is None:
            raise Exception("Objective not defined yet!")
        
        # create estimation problem
        self.problem = EstimationProblem(
            state_preparation=self.option,
            objective_qubits=[self.objective_index],
            post_processing=self.objective.post_processing,
        )
    
    def run(self, epsilon=0.01, alpha=0.05, shots=100, method="MIQAE"):
        # construct amplitude estimation
        if method=="IQAE":
            ae = IterativeAmplitudeEstimation(
                epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": shots})
            )
            self.result = ae.estimate(self.problem)
        elif method == "MIQAE":
            qi = QuantumInstance(backend=AerSimulator(), shots=shots)
            ae = ModifiedIterativeAmplitudeEstimation(
                epsilon_target=epsilon, alpha=alpha, quantum_instance=qi)
            self.result = ae.estimate(self.problem, shots=shots)
        return self.result
    
    def process_results(self):
        if self.option_type in ['call', 'call-on-max', 'call-on-min', 'spread call']:
            conf_int = (
                np.array(self.result.confidence_interval_processed)
            )
            estimated_value = self.result.estimation_processed
        elif self.option_type == "basket call":
            conf_int = (
                np.array(self.result.confidence_interval_processed)
                / (2**self.num_uncertainty_qubits - 1)
                * (self.high - self.low)
            )
            estimated_value = self.result.estimation_processed / (2**self.num_uncertainty_qubits - 1) * (self.high - self.low)
        return estimated_value, conf_int
    
    def compute_exact_expectation(self):
        if self.option_type == "call":
            exact_value = self._compute_exact_call_expectation()
            return exact_value
        elif self.option_type == 'basket call':
            exact_value = self._compute_exact_basket_call_expectation()
            return exact_value
        elif self.option_type == 'spread call':
            exact_value = self._compute_exact_spread_call_expectation()
            return exact_value
        elif self.option_type == 'call-on-max':
            exact_value = self._compute_exact_call_on_max_expectation()
            return exact_value
        elif self.option_type == 'call-on-min':
            exact_value = self._compute_exact_call_on_min_expectation()
            return exact_value
        elif self.option_type == 'best-of-call':
            self.objective = self._compute_exact_best_of_call_expectation()
        else:
            raise Exception("Option type not defined!")
            
    def _compute_exact_call_expectation(self):
        payoff = np.maximum(self.uncertainty_model.values - self.strike_price, 0)
        expected_value = np.dot(self.uncertainty_model.probabilities, payoff)
        return expected_value
    
    def _compute_exact_basket_call_expectation(self):
        sum_values = np.sum(self.uncertainty_model.values, axis=1)
        expected_value = np.dot(
            self.uncertainty_model.probabilities[sum_values >= self.strike_price], 
            sum_values[sum_values >= self.strike_price] - self.strike_price
        )
        return expected_value
    
    def _compute_exact_spread_call_expectation(self):
        diff_values = np.array([v[0]-v[1] for v in self.uncertainty_model.values])

        exact_value = np.dot(
            self.uncertainty_model.probabilities[diff_values >= self.strike_price],
            diff_values[diff_values >= self.strike_price] - self.strike_price,
        )
        
        return exact_value
    
    def _compute_exact_call_on_max_expectation(self):
        exact_value = 0
        for i in range(len(self.uncertainty_model.probabilities)):
            exact_value += self.uncertainty_model.probabilities[i]*max(0, max(self.uncertainty_model.values[i][0], self.uncertainty_model.values[i][1])-self.strike_price)
        return exact_value
    
    def _compute_exact_call_on_min_expectation(self):
        exact_value = 0
        for i in range(len(self.uncertainty_model.probabilities)):
            exact_value += self.uncertainty_model.probabilities[i]*max(0, self.strike_price - min(self.uncertainty_model.values[i][0], self.uncertainty_model.values[i][1]))
        return exact_value
    
    def estimate_expectation(self, epsilon=0.01, alpha=0.05, shots=100):
        self.create_estimation_problem(epsilon)
        self.run(epsilon, alpha, shots)
        return self.process_results()
    
    def get_num_oracle_calls(self):
        return self.result.num_oracle_queries
    
        
class Plotter():
    def __init__(self):
        pass
    
    def plot_distribution(self, uncertainty_model: LogNormalDistribution, num_variables: int):
        if num_variables == 1:
            x = uncertainty_model.values
            y = uncertainty_model.probabilities
            plt.bar(x, y, width=0.2)
            plt.xticks(x, size=15, rotation=90)
            plt.yticks(size=15)
            plt.grid()
            plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
            plt.ylabel("Probability ($\%$)", size=15)
            plt.show()
        elif num_variables == 2:
            x = [v[0] for v in uncertainty_model.values]
            y = [v[1] for v in uncertainty_model.values]
            z = uncertainty_model.probabilities
            # z = map(float, z)
            # z = list(map(float, z))
            resolution = np.array([2**n for n in uncertainty_model.num_qubits]) * 1j
            grid_x, grid_y = np.mgrid[min(x) : max(x) : resolution[0], min(y) : max(y) : resolution[1]]
            grid_z = griddata((x, y), z, (grid_x, grid_y))
            plt.figure(figsize=(10, 8))
            ax = plt.axes(projection="3d")
            ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
            ax.set_xlabel("Spot Price $S_T^1$ (\$)", size=14)
            ax.set_ylabel("Spot Price $S_T^2$ (\$)", size=14)
            ax.set_zlabel("Probability (\%)", size=15)
            plt.show()