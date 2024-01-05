import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution, NormalDistribution, UniformDistribution
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from ModifiedIQAE.algorithms.amplitude_estimators.mod_iae import ModifiedIterativeAmplitudeEstimation
from typing import List
from scipy.interpolate import griddata

class OptionPricing():
    
    def __init__(self, num_uncertainty_qubits, options_params: List[dict]):
        self.num_variables = len(options_params)
        self.num_uncertainty_qubits = num_uncertainty_qubits
        self.define_uncertainty_model(options_params)
        self.objective_index = None
        self.option = None
        self.objective = None
        self.strike_prices = options_params["strike_prices"]
        
    
    def define_uncertainty_model(self, params: List[dict]):
        self.all_variables = []
        for var in params:
            curr_var_dict = {}
            curr_mu = (var['r'] - 0.5 * var['vol']**2)* var['T'] + np.log(var['S'])
            curr_sigma = var['vol'] * np.sqrt(var['T'])
            curr_mean = np.exp(curr_mu + curr_sigma**2/2)
            curr_std = np.sqrt((np.exp(curr_sigma**2) - 1) * np.exp(2*curr_mu + curr_sigma**2))
            curr_low = np.maximum(0, curr_mean - 3*curr_std)
            curr_high = curr_mean + 3*curr_std
            # append all parameters
            curr_var_dict['low'] = curr_low
            curr_var_dict['high'] = curr_high
            curr_var_dict['mu'] = curr_mu
            curr_var_dict['sigma'] = curr_sigma
            curr_var_dict['mean'] = curr_mean
            curr_var_dict['std'] = curr_std
            self.all_variables.append(curr_var_dict)

        self.dimension = len(params)
        self.num_dist_qubits = [self.num_uncertainty_qubits] * self.dimension
        lower_bound = np.diag([var['low'] for var in self.all_variables])
        upper_bound = np.diag([var['high'] for var in self.all_variables])
        mu = np.array([var['mu'] for var in self.all_variables])
        cov = np.diag([var['sigma']**2 for var in self.all_variables])
        self.uncertainty_model = LogNormalDistribution(num_qubits=self.num_dist_qubits, mu=mu, sigma=cov, bounds=list(zip(lower_bound, upper_bound)))
    
    def plot_distribution(self):
        if hasattr(self, 'uncertainty_model'):
            if self.num_variables == 1:
                x = self.uncertainty_model.values
                y = self.uncertainty_model.probabilities
                plt.bar(x, y, width=0.2)
                plt.xticks(x, size=15, rotation=90)
                plt.yticks(size=15)
                plt.grid()
                plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
                plt.ylabel("Probability ($\%$)", size=15)
                plt.show()
            elif self.num_variables == 2:
                x = [v[0] for v in self.uncertainty_model.values]
                y = [v[1] for v in self.uncertainty_model.values]
                z = self.uncertainty_model.probabilities
                # z = map(float, z)
                # z = list(map(float, z))
                resolution = np.array([2**n for n in self.num_dist_qubits]) * 1j
                grid_x, grid_y = np.mgrid[min(x) : max(x) : resolution[0], min(y) : max(y) : resolution[1]]
                grid_z = griddata((x, y), z, (grid_x, grid_y))
                plt.figure(figsize=(10, 8))
                ax = plt.axes(projection="3d")
                ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
                ax.set_xlabel("Spot Price $S_T^1$ (\$)", size=14)
                ax.set_ylabel("Spot Price $S_T^2$ (\$)", size=14)
                ax.set_zlabel("Probability (\%)", size=15)
                plt.show()
        else:
            raise Exception("Uncertainty model not defined yet!")
        
    def define_payoff_function(self):
        pass
    
    def create_circuit(self):
        pass
    
    def create_estimation_problem(self):
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
    
    def run(self):
        pass
        