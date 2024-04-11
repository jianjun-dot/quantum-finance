from qiskit.circuit.library import MCXGate, DraperQFTAdder
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
class OneIncrement():
    def __init__(self, num_qubits):
        compute_register = QuantumRegister(num_qubits, 'compute')
        self.circuit = QuantumCircuit(compute_register)
        self.circuit.x(compute_register[0])
        self.circuit.cx(compute_register[0], compute_register[1], ctrl_state=0)
        for i in range(2, num_qubits):
            mcxGate = MCXGate(num_ctrl_qubits=i, ctrl_state="0"*i)
            self.circuit.append(mcxGate, compute_register[:i+1])
            
class Subtractor():
    def __init__(self, bitLength):
        firstRegister = QuantumRegister(bitLength, 'first')
        secondRegister = QuantumRegister(bitLength, 'second')
        carryRegister = QuantumRegister(1, 'carry')

        test_circuit = QuantumCircuit(firstRegister, secondRegister, carryRegister, name="subtractor")
        
        adder = DraperQFTAdder(bitLength, kind="half")
        oneIncrementer = OneIncrement(bitLength+1).circuit
        test_circuit.x(secondRegister[:]+carryRegister[:])
        test_circuit.append(adder, firstRegister[:] + secondRegister[:] + carryRegister[:])
        test_circuit.append(oneIncrementer, secondRegister[:]+ carryRegister[:])
        self.circuit = test_circuit
        
def loadNumber(circ: QuantumCircuit, register: QuantumRegister, number: int):
    number_in_binary = bin(number)[2:].zfill(register.size)
    # print(number_in_binary)
    for i in range(len(number_in_binary)):
        if number_in_binary[::-1][i] == "1":
            circ.x(register[i])
    return circ

def define_covariance_matrix(first_var, second_var, correlation=0.5):
    covariance_matrix = np.array(
        [
            [first_var, correlation * np.sqrt(first_var) * np.sqrt(second_var)],
            [correlation * np.sqrt(first_var) * np.sqrt(second_var), second_var],
        ]
    )
    return covariance_matrix