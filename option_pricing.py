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


class OptionPricing():
    
    def __init__(self):
        pass
    
    def define_uncertainty_model(self):
        pass
    
    def define_payoff_function(self):
        pass
    
    def create_circuit(self):
        pass
    
    def create_estimation_problem(self):
        pass
    
    def run(self):
        pass
        