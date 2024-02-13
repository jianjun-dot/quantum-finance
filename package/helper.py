from qiskit.circuit.library import MCXGate, DraperQFTAdder
from qiskit import QuantumCircuit, QuantumRegister

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
        