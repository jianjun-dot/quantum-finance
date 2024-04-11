from qiskit_finance.circuit.library import LogNormalDistribution
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np

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