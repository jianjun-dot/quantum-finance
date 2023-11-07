import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator

# Define the operator A. For the counting problem, this is the H gate applied on each qubit. 
def A(nqubits):
    qc = QuantumCircuit(nqubits)
    for qubit in range(nqubits):
        qc.h(qubit)

    operator = qc.to_gate()
    operator.name = "A"
    return operator

## S_chi
# Flip the sign of the marked items
def S_chi(N, marked):
    S_chi_array = []
    for i in range(N):
        row = np.zeros(N)
        if (i in marked):
            row[i] = -1
        else:
            row[i] = 1
        S_chi_array.append(row)
    return Operator(S_chi_array)

## S_0
# Flip the sign of the 0 state
def S_0(N):
    array_op = []
    for i in range(N):
        row = np.zeros(N)
        if (i == 0):
            row[i] = -1
        else:
            row[i] = 1
        array_op.append(row)

    return Operator(array_op)

# Define the operator Q.
def Q(nqubits, marked):
    qc = QuantumCircuit(nqubits)
    qc.append(S_chi(2**nqubits, marked), range(nqubits))
    qc.append(A(nqubits),                range(nqubits))
    qc.append(S_0(2**nqubits),           range(nqubits))
    qc.append(A(nqubits),                range(nqubits))

    operator = qc.to_gate()
    operator.name = "Q"
    return operator