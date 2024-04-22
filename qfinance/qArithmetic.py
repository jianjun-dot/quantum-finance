from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.library import QFT, MCXGate, VBERippleCarryAdder, DraperQFTAdder

def rotation(k):
    return np.array([[1, 0], [0, np.exp(2 * np.pi * 1j / 2**k)]])

def controlled_inverse_rotation_gate(k):
    inverse_rotation = np.array(
        [[1, 0], [0, np.exp(-2 * np.pi * 1j / 2**k)]], dtype=np.complex128
    )
    # print(inverse_rotation)
    inverse_rotation_gate = Operator(inverse_rotation)
    circuit = QuantumCircuit(1)
    circuit.append(inverse_rotation_gate, [0])
    inverse_rotation_gate = circuit.to_gate(label="R'{}".format(k))
    controlled_rotation = inverse_rotation_gate.control(1)
    return controlled_rotation

def build_NMSub(m, n):
    first_number_register = QuantumRegister(n + 1, name="phi(a)")
    second_number_register = QuantumRegister(m, name="b")
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register)
    counter = 1
    for i in reversed(range(1, n + 1)):
        for j in reversed(range(counter)):
            quantum_circuit.cp(-2 * np.pi /2**(counter - j), second_number_register[m - j - 1], first_number_register[i])
        if counter < m:
            counter += 1
        quantum_circuit.barrier()
    for j in reversed(range(counter)):

        quantum_circuit.cp(-2 * np.pi /2**(counter - j + 1), second_number_register[m - j - 1], first_number_register[0])
    return quantum_circuit

def build_NMAdd(m, n):
    first_number_register = QuantumRegister(n, name='phi(a)')
    second_number_register = QuantumRegister(m+1, name='b')
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register, name='NMAdd')
    counter = 1
    for i in range(0, m):
        for j in range(counter):
            quantum_circuit.cp(2*np.pi/2**(counter-j), first_number_register[j], second_number_register[i])
        if counter < n:
            counter += 1
        quantum_circuit.barrier()
    for j in range(counter):
        quantum_circuit.cp(2*np.pi/2**(j+2), first_number_register[n-j-1], second_number_register[-1])
    return quantum_circuit

def build_QNMAdd(m, n):
    carry_register = QuantumRegister(1, name='c')
    first_number_register = QuantumRegister(n, name='a')
    second_number_register = QuantumRegister(m, name='b')
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, name='QNMAdd')
    
    quantum_circuit.ccx(first_number_register[-1], second_number_register[-1], carry_register[0], ctrl_state='10')
    quantum_circuit.ccx(first_number_register[-1], second_number_register[-1], carry_register[0], ctrl_state='01')

    qft = QFT(n+1, do_swaps=False)
    nmSub = build_NMSub(m, n)
    inverse_qft = qft.inverse()
    quantum_circuit.append(qft, second_number_register[:]+carry_register[:])
    quantum_circuit.append(nmSub, first_number_register[:]+ second_number_register[:]+carry_register[:])
    quantum_circuit.append(inverse_qft, second_number_register[:]+carry_register[:])
    return quantum_circuit

def build_NMSub(m, n):
    first_number_register = QuantumRegister(n, name='phi(a)')
    second_number_register = QuantumRegister(m+1, name='b')
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register, name='NMSub')
    counter = 1
    for i in range(0, m):
        for j in range(counter):
            # print(i, j)
            quantum_circuit.cp(-2*np.pi/2**(counter-j), first_number_register[j], second_number_register[i])
            # quantum_circuit.append(controlled_inverse_rotation_gate(j+1), [second_number_register[m-j-1]]+ [first_number_register[i]])
        if counter < n:
            counter += 1
        quantum_circuit.barrier()
    for j in range(counter):
        quantum_circuit.cp(-2*np.pi/2**(j+2), first_number_register[n-j-1], second_number_register[-1])
    return quantum_circuit

def build_QNMSub(m, n):
    carry_register = QuantumRegister(1, name='c')
    first_number_register = QuantumRegister(n, name='a')
    second_number_register = QuantumRegister(m, name='b')
    quantum_circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, name='QNMSub')
    
    quantum_circuit.ccx(first_number_register[-1], second_number_register[-1], carry_register[0], ctrl_state='10')
    quantum_circuit.ccx(first_number_register[-1], second_number_register[-1], carry_register[0], ctrl_state='01')

    qft = QFT(n+1, do_swaps=False)
    nmSub = build_NMSub(m, n)
    inverse_qft = qft.inverse()
    quantum_circuit.append(qft, second_number_register[:]+carry_register[:])
    quantum_circuit.append(nmSub, first_number_register[:]+ second_number_register[:]+carry_register[:])
    quantum_circuit.append(inverse_qft, second_number_register[:]+carry_register[:])
    return quantum_circuit

def QComp(n,m):
    # 001: a = b, 010: a < b, 100: a > b
    carry_register = QuantumRegister(1, name='c')
    first_number_register = QuantumRegister(n, name='a')
    second_number_register = QuantumRegister(m, name='b')
    ancilla_register = QuantumRegister(3, name='ancilla')
    circuit = QuantumCircuit(first_number_register, second_number_register, carry_register, ancilla_register, name='QComp')
    qnmsub = build_QNMSub(m, n)
    circuit.append(qnmsub, first_number_register[:] + second_number_register[:]+ carry_register[:])
    circuit.cx(carry_register[:], ancilla_register[0], ctrl_state=0)
    circuit.cx(carry_register[:], ancilla_register[1])
    mcx = MCXGate(num_ctrl_qubits=n+1, ctrl_state='0'*(n+1))
    circuit.append(mcx, carry_register[:]+ second_number_register[:]+ [ancilla_register[0]])
    circuit.append(mcx, carry_register[:]+ second_number_register[:]+ [ancilla_register[2]])
    return circuit

def subtractorVBE(first_num_size, second_num_size):
    firstRegister = QuantumRegister(first_num_size, 'first')
    secondRegister = QuantumRegister(second_num_size, 'second')
    carryRegister = QuantumRegister(1, 'carry')
    ancillaRegister = QuantumRegister(first_num_size, 'ancilla')

    adder = VBERippleCarryAdder(first_num_size, name="Adder")
    # adder = DraperQFTAdder(num_qubits_for_each_dimension, kind="half",name="Adder")
    num_qubits = len(adder.qubits)
    
    circ = QuantumCircuit(carryRegister, firstRegister, secondRegister, ancillaRegister, name="subtractor")
    circ.x(secondRegister)
    circ.x(carryRegister)
    circ.append(adder, list(range(num_qubits)))
    return circ
            
def oneIncrement(num_qubits):
    compute_register = QuantumRegister(num_qubits, 'compute')
    circuit = QuantumCircuit(compute_register)
    circuit.x(compute_register[0])
    circuit.cx(compute_register[0], compute_register[1], ctrl_state=0)
    for i in range(2, num_qubits):
        mcxGate = MCXGate(num_ctrl_qubits=i, ctrl_state="0"*i)
        circuit.append(mcxGate, compute_register[:i+1])
    return circuit

def subtractorDraper(bitLength):
    firstRegister = QuantumRegister(bitLength, 'first')
    secondRegister = QuantumRegister(bitLength, 'second')
    carryRegister = QuantumRegister(1, 'carry')

    circuit = QuantumCircuit(firstRegister, secondRegister, carryRegister, name="subtractor")
    
    adder = DraperQFTAdder(bitLength, kind="half")
    # oneIncrementer = OneIncrement(bitLength+1).circuit
    oneIncrementer = oneIncrement(bitLength+1)
    circuit.x(secondRegister[:]+carryRegister[:])
    circuit.append(adder, firstRegister[:] + secondRegister[:] + carryRegister[:])
    circuit.append(oneIncrementer, secondRegister[:]+ carryRegister[:])
    return circuit