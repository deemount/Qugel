import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import sys
from pennylane.operation import Tensor
import psutil
import matplotlib.pyplot as plt
import torch.nn as nn

## ------------ Quantum gates ------------ ##

'''
In general, variational classifiers utilize a core element known as a layer to build the complete variational circuit. 
These layers are applied iteratively and consist of rotational operations on each qubit, 
combined with CNOT gates that create entanglement between neighboring qubits.
'''

## ------------ Quantum gates ------------ ##

def Q_H(nqubits):
    """
    Applies Hadamard gates to a given number of qubits.

    Args:
        nqubits (int): Number of qubits.

    Returns:
        None
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def Q_RY(w):
    """
    Applies RY rotation gates to each qubit in a given parameter array.

    Args:
        w (array-like): Array of rotation angles.

    Returns:
        None
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def Q_RX(w):
    """
    Applies RX rotation gates to each qubit in a given parameter array.

    Args:
        w (array-like): Array of rotation angles.

    Returns:
        None
    """
    for idx, element in enumerate(w):
        qml.RX(element, wires=idx)


def Q_Entangle_A(nqubits):
    """
    Applies entangling gates (CNOT) between adjacent qubits.

    Args:
        nqubits (int): Number of qubits.

    Returns:
        None
    """
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def Q_Entangle_B(nqubits):
    """
    Applies entangling gates (CNOT) between specific qubit pairs.

    Args:
        nqubits (int): Number of qubits.

    Returns:
        None
    """
    for i in range(nqubits):
        if i < nqubits - 1 and ((nqubits - i - 1) != i) and (i + 2 < nqubits):
            qml.CNOT(wires=[i + 2, i])
        else:
            print("Q_Entangle_B - Skipping an invalid CNOT wire assignment: i=", i)


def Q_Entangle_C(nqubits):
    """
    Applies entangling gates (CNOT) between specific qubit pairs, with reversed parameter order.

    Args:
        nqubits (int): Number of qubits.

    Returns:
        None
    """
    for i in range(nqubits):
        if (i < nqubits - 1) and ((nqubits - i - 1) != i):
            if i % 2 == 0:
                qml.CNOT(wires=[nqubits - i - 1, i])
            else:
                qml.CNOT(wires=[i, nqubits - i - 1])


def Q_encoding_block(inputs, n_qubits):
    """
    Applies the encoding block of the variational circuit to encode inputs into the qubits.

    Args:
        inputs (array-like): Array of input values.
        n_qubits (int): Number of qubits.

    Returns:
        None
    """
    for qub in range(n_qubits):
        qml.Hadamard(wires=qub)
        qml.RY(inputs[qub], wires=qub)


def Q_quanvol_block_A(q_weights, n_qubits, q_depth):
    """
    Applies the quantum volume block A of the variational circuit.

    Args:
        q_weights (array-like): Array of quantum weights.
        n_qubits (int): Number of qubits.
        q_depth (int): Number of layers in the circuit.

    Returns:
        None
    """
    for layer in range(q_depth):
        for i in range(n_qubits):
            if (n_qubits - i - 1) != i:
                if i % 2 != 0:
                    qml.CNOT(wires=[n_qubits - i - 1, i])
                else:
                    qml.CNOT(wires=[i, n_qubits - i - 1, ])
                if (i < n_qubits - 1) and ((n_qubits - i - 1) != i):
                    qml.CRZ(q_weights[layer], wires=[i, (i + 1) % n_qubits])
                    if i % 2 == 0:
                        qml.CNOT(wires=[n_qubits - i - 1, i])
                    else:
                        qml.CNOT(wires=[i, n_qubits - i - 1, ])
                        qml.Hadamard(n_qubits - i - 1)
        qml.RY(q_weights[layer], wires=i)


def Q_quanvol_block_B(q_weights, n_qubits, q_depth):
    """
    Applies the quantum volume block B of the variational circuit.

    Args:
        q_weights (array-like): Array of quantum weights.
        n_qubits (int): Number of qubits.
        q_depth (int): Number of layers in the circuit.

    Returns:
        None
    """
    for layer in range(q_depth):
        for i in range(n_qubits):
            qml.RX(q_weights[layer], wires=[(i + 1) % n_qubits])
            if (i < n_qubits - 1) and (
                    (n_qubits - i - 1) != i):
                if i % 2 == 0:
                    qml.CNOT(wires=[i, n_qubits - i - 1])
                else:
                    qml.CNOT(wires=[n_qubits - i - 1, i])
            qml.RY(q_weights[layer], wires=i)


def Q_quanvol_block_C(q_weights, n_qubits, q_depth):
    """
    Applies the quantum volume block C of the variational circuit.

    Args:
        q_weights (array-like): Array of quantum weights.
        n_qubits (int): Number of qubits.
        q_depth (int): Number of layers in the circuit.

    Returns:
        None
    """
    for layer in range(q_depth):
        for i in range(n_qubits):
            qml.RY(q_weights[layer], wires=[(i + 1) % n_qubits])
            if (i < n_qubits - 1) and (
                    (n_qubits - i) != i):
                if i % 2 == 0:
                    qml.CNOT(wires=[i, n_qubits - i - 1])
                else:
                    if i % 2 != 0:
                        qml.Hadamard(i)
                    else:
                        qml.Hadamard(n_qubits - i - 1)
                        qml.CNOT(wires=[i, n_qubits - i - 1])
            qml.RX(q_weights[layer], wires=i)


def density_matrix(state):
    """
    Calculates the density matrix of a quantum state.

    Args:
        state (array-like): Quantum state.

    Returns:
        array-like: Density matrix of the quantum state.
    """
    return state * np.conj(state).T


def Q_encoding_circuit_A(inputs, q_weights, n_qubits, q_depth):
    """
    Constructs the variational quantum circuit for encoding inputs and applying the quantum volume block A.

    Args:
        inputs (array-like): Array of input values.
        q_weights (array-like): Array of quantum weights.
        n_qubits (int): Number of qubits.
        q_depth (int): Number of layers in the circuit.

    Returns:
        array-like: Expectation values of Pauli-Z measurements on each qubit.
    """
    Q_encoding_block(inputs, n_qubits)

    Q_quanvol_block_A(q_weights, n_qubits, q_depth)

    exp_vals = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return exp_vals


def Q_Plot(cirq_0, q_b, q_d, text=True):
    """
    Plots the quantum circuit diagram.

    Args:
        cirq_0 (function): Quantum circuit function.
        q_b (int): Number of qubits.
        q_d (int): Number of layers in the circuit.
        text (bool, optional): Flag to display circuit text instead of a diagram. Defaults to True.

    Returns:
        None
    """
    print("Plot Q/D:{}/{}".format(q_b, q_d))
    if text is not True:
        fig, ax = qml.draw_mpl(cirq_0, expansion_strategy='device')(torch.zeros(q_b), torch.zeros(q_d), q_b, q_d)
        plt.show()
        fig.show()
    else:
        print(qml.draw(cirq_0, expansion_strategy='device')(torch.zeros(q_b), torch.zeros(q_d), q_b, q_d))
