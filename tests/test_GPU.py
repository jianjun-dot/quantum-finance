from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from time import time
from qiskit.circuit.library import QFT
import numpy as np
from qiskit_aer import AerSimulator
# from cusvaer.backends import StatevectorSimulator

def create_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    return circuit

def create_QFT_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    qft = QFT(num_qubits=n_qubits, approximation_degree=0, do_swaps=True, inverse=False)
    circuit.append(qft, range(n_qubits))
    return circuit


n_runs = 20
shots = 100

qubits_list = [5]
# qubits_list = [5, 10, 15, 20, 25, 26, 27, 28, 29, 30]

# circuit = create_ghz_circuit(n_qubits=10)
# circuit.measure_all()
# ==== CPU ==== 
# time
data = []
for n_qubit in qubits_list:
    
    # circuit = create_ghz_circuit(n_qubits=n_qubit)
    circuit = create_QFT_circuit(n_qubits=n_qubit)
    circuit.measure_all()
    
    # ==== GPU ====
    # time
    print(f"==== GPU {n_qubit} ====")
    # simulator = Aer.get_backend('aer_simulator_statevector')
    simulator = AerSimulator()
    simulator.set_options(device='GPU')
    
    gpu_circuit = transpile(circuit, simulator)

    start = time()
    for i in range(n_runs):
        job = simulator.run(gpu_circuit, shots=shots)
        result = job.result()
        print(result.get_counts())
    end = time()
    GPU_time = end - start
    print(f'GPU time per run: {GPU_time/n_runs}')
    
    
    print(f"==== CPU {n_qubit} ====")
    simulator = AerSimulator()
    simulator.set_options(device='CPU')
    cpu_circuit = transpile(circuit, simulator)

    start = time()
    for i in range(n_runs):
        job = simulator.run(cpu_circuit, shots=shots)
        result = job.result()

    end = time()
    CPU_time = end - start
    print(f'CPU time per run: {CPU_time/n_runs}')

    print("Speedup: ", CPU_time/GPU_time)
    print("\n")
    data.append([n_qubit, CPU_time, GPU_time, CPU_time/GPU_time])
    
data = np.array(data)
np.savetxt("GPU_speed_up_qft_nvidia_thrust.csv", data, delimiter=",", header= "n_qubit, CPU_time, GPU_time, Speedup")


