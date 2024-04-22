"""
This is a modification of Qiskit quantum machine learning tutorial code, with the following license:
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
"""

from qiskit import QuantumCircuit
from qiskit.utils import algorithm_globals
from qiskit.circuit.library import EfficientSU2, TwoLocal
from qiskit.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_finance.circuit.library import NormalDistribution, LogNormalDistribution, UniformDistribution
from torch import nn
import numpy as np
import torch
from torch.optim import Adam
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time
from scipy.stats import entropy
from typing import Callable

class GeneratorCreator():
    def __init__(self, num_qubits: int, random_seed: int=None):
        self.num_qubits = num_qubits
        if random_seed is None:
            self.random_seed = 123456
            algorithm_globals.random_seed = self.random_seed
        else:
            self.random_seed = random_seed
            algorithm_globals.random_seed = self.random_seed

    def define_ansatz(self, ansatz: str = 'two-local', reps: int = 1):
        if ansatz == 'two-local':
            self.ansatz = TwoLocal(num_qubits=self.num_qubits, rotation_blocks=['ry', 'rz'], entanglement_blocks='cx', entanglement='linear', reps=reps)
        elif ansatz == 'efficient-su2':
            self.ansatz = EfficientSU2(num_qubits=self.num_qubits, entanglement='linear', reps=reps)
        else:
            raise NotImplementedError

    def define_init_distribution(self, init_dist_dict: dict) -> None:
        if init_dist_dict["init_dist"] == "uniform":
            self.init_dist = UniformDistribution(num_qubits=self.num_qubits) 
            init_circuit = QuantumCircuit(self.num_qubits)
            init_circuit.h(range(self.num_qubits))
            self.init_dist = init_circuit

        elif init_dist_dict["init_dist"] == "normal":
            mu = init_dist_dict["mu"]
            sigma = init_dist_dict["sigma"]
            bounds = init_dist_dict["bounds"]
            self.init_dist = NormalDistribution(num_qubits=self.num_qubits, mu=mu, sigma=sigma, bounds=bounds)
        elif init_dist_dict["init_dist"] == "logNormal":
            mu = init_dist_dict["mu"]
            sigma = init_dist_dict["sigma"]
            bounds = init_dist_dict["bounds"]
            self.init_dist = LogNormalDistribution(num_qubits=self.num_qubits, mu=mu, sigma=sigma, bounds=bounds)
        else:
            raise Exception("Distribution not implemented")
        
    def compose_circuit(self) -> None:
        self.circuit = self.init_dist.compose(self.ansatz, inplace=False)
        return self.circuit

    def define_sampler(self, shots: int) -> None:
        self.sampler = Sampler(options={"shots": shots, "seed": algorithm_globals.random_seed})
    
    def create_generator(self) -> TorchConnector:
        qnn = SamplerQNN(
            circuit=self.circuit,
            sampler=self.sampler,
            input_params=[],
            weight_params=self.circuit.parameters,
            sparse=False,
        )
        self.qnn = qnn

        initial_weights = algorithm_globals.random.random(self.circuit.num_parameters)
        return TorchConnector(qnn, initial_weights)
    
    @property
    def num_parameters(self):
        return self.circuit.num_parameters

class Discriminator(nn.Module):
    def __init__(self, input_size: int, intermediate_layer_size = 20, num_parameters:int=None):
        super(Discriminator, self).__init__()

        self.linear_input = nn.Linear(input_size, intermediate_layer_size)
        self.leaky_relu = nn.LeakyReLU(0.2)
        # self.linear_intermediate20 = nn.Linear(intermediate_layer_size, intermediate_layer_size)
        self.linear20 = nn.Linear(intermediate_layer_size, 1)
        self.sigmoid = nn.Sigmoid()
        if num_parameters is None:
            self.num_layers = 0
        else:
            self.compute_layers(num_parameters)
    
    @property
    def num_parameters(self):
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return self.num_params

    def compute_layers(num_parameters: int) -> int:
        return max(0, int(np.round((num_parameters - 81)/400)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(input)
        x = self.leaky_relu(x)
        # for _ in range(self.num_layers):
        #     x = self.linear_intermediate20(x)
        #     x = self.leaky_relu(x)
        x = self.linear20(x)
        x = self.sigmoid(x)
        return x

class DistributionCircuitTrainer():

    def __init__(self, num_dims: int, num_discrete_values: int):
        self.num_dims = num_dims
        self.num_discrete_values = num_discrete_values

    
    def define_init_distribution(self, init_dist: str = 'uniform', mu=0, sigma=1, bounds: np.ndarray = None) -> None:
        self.init_dist = {}
        self.init_dist["init_dist"] = init_dist
        self.init_dist["mu"] = mu
        self.init_dist["sigma"] = sigma
        self.init_dist["bounds"] = bounds
    
    def define_generator(self, ansatz: str = 'two-local', reps: int = 6, shots=10000) -> None:
        num_qubits = self.num_dims * int(np.log2(self.num_discrete_values))
        generator_creator = GeneratorCreator(num_qubits=num_qubits)
        generator_creator.define_ansatz(ansatz, reps=reps)
        generator_creator.define_init_distribution(self.init_dist)
        generator_creator.compose_circuit()
        generator_creator.define_sampler(shots=shots)
        self.generator_creator = generator_creator
        self.generator_circuit = generator_creator.circuit
        self.generator = generator_creator.create_generator()

    def define_discriminator(self, intermediate_layer_size=20) -> None:
        discrimator = Discriminator(input_size=self.num_dims, intermediate_layer_size=intermediate_layer_size)
        self.discriminator = discrimator

    def binary_cross_entropy_loss(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> float:
        bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
        weighted_loss = weight * bce_loss
        total_loss = -torch.sum(weighted_loss)
        return total_loss
    
    def define_loss(self, loss_function: Callable[[torch.Tensor,torch.Tensor,torch.Tensor], float] = None) -> None:
        if loss_function is None:
            self.loss_function = self.binary_cross_entropy_loss
        else:
            self.loss_function = loss_function
        

    def define_optimizer(self, learning_rate: float, b1: float, b2: float) -> None:
        self.generator_optimizer = Adam(self.generator.parameters(), lr=learning_rate, betas=(b1, b2), weight_decay=0.005)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(), lr=learning_rate, betas=(b1, b2), weight_decay=0.005)

    def train(self, grid_elements: np.ndarray, prob_training_data: np.ndarray, num_epochs = 50, plot=True):

        num_qnn_outputs = self.num_discrete_values**self.num_dims

        generator_loss_values = []
        discriminator_loss_values = []
        entropy_values = []

        start = time.time()
        # print(self.generator.__dict__)
        for num in range(num_epochs):
            valid = torch.ones(num_qnn_outputs, 1, dtype=torch.float)
            fake = torch.zeros(num_qnn_outputs, 1, dtype=torch.float)

            # Configure input
            real_dist = torch.tensor(prob_training_data, dtype=torch.float).reshape(-1, 1)

            # Configure samples
            samples = torch.tensor(grid_elements, dtype=torch.float)
            disc_value = self.discriminator(samples)
            # print(disc_value)

            # Generate data
            gen_dist = self.generator(torch.tensor([])).reshape(-1, 1)
            # print(gen_dist)

            # Train generator
            self.generator_optimizer.zero_grad()
            generator_loss = self.binary_cross_entropy_loss(disc_value, valid, gen_dist)

            # store for plotting
            generator_loss_values.append(generator_loss.detach().item())

            generator_loss.backward(retain_graph=True)

            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1)


            self.generator_optimizer.step()

            self.discriminator_optimizer.zero_grad()

            real_loss = self.binary_cross_entropy_loss(disc_value, valid, real_dist)
            fake_loss = self.binary_cross_entropy_loss(disc_value, fake, gen_dist.detach())
            discriminator_loss = (real_loss + fake_loss) / 2


            # Store for plotting
            discriminator_loss_values.append(discriminator_loss.detach().item())

            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            entropy_value = entropy(gen_dist.detach().squeeze().numpy(), prob_training_data)
            entropy_values.append(entropy_value)
            if plot:
                self.visualize_training_progress(generator_loss_values, discriminator_loss_values, entropy_values)

        elapsed = time.time() - start
        print(f"Fit in {elapsed:0.2f} sec")


    def get_generated_distribution(self):
        with torch.no_grad():
            generated_probabilities = self.generator().numpy()
        return generated_probabilities
    
    def visualize_training_progress(self, generator_loss_values: list, discriminator_loss_values: list, entropy_values: list) -> None:
        if len(generator_loss_values) < 2:
            return

        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

        # Generator Loss
        ax1.set_title("Loss")
        ax1.plot(generator_loss_values, label="generator loss", color="royalblue")
        ax1.plot(discriminator_loss_values, label="discriminator loss", color="magenta")
        ax1.legend(loc="best")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Loss")
        ax1.grid()

        # Relative Entropy
        ax2.set_title("Relative entropy")
        ax2.plot(entropy_values)
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Relative entropy")
        ax2.grid()

        plt.show()

    @property
    def generator_parameters(self):
        return self.generator.parameters
    
    @property
    def discriminator_parameters(self):
        return self.discriminator.parameters
    
    @property
    def num_generator_parameters(self):
        return self.generator_creator.num_parameters
    
    @property
    def num_discriminator_parameters(self):
        return self.discriminator.num_parameters

