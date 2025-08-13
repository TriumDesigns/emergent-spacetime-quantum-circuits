
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def create_4qubit_emergence_circuit_for_drawing():
    """
    Reconstructs the 4-qubit emergence circuit from the LaTeX document's
    verbatim block for drawing.
    """
    qr = QuantumRegister(4, 'q')
    cr = ClassicalRegister(4, 'c')
    qc = QuantumCircuit(qr, cr)

    # Corresponds to L(G): Simulating gravitational entanglement
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])  # spread gravity influence

    # Corresponds to T(G): Simulating gauge dynamics
    qc.rx(np.pi/4, qr[1])
    qc.h(qr[2])
    qc.cx(qr[2], qr[3])

    # Prepare "matter-like" qubit (q3) in superposition
    qc.ry(np.pi/3, qr[3])
    qc.h(qr[3]) # This H gate is part of the Controlled-Z decomposition

    # Controlled-Z (from q1 to q3) implemented as H-CX-H
    qc.cx(qr[1], qr[3])
    qc.h(qr[3])

    # Controlled-X (from q2 to q3)
    qc.cx(qr[2], qr[3])

    # Final measurements
    # Note: In the LaTeX verbatim block, 'qc.measure(qr, cr)' is used.
    # This implies measuring all qubits into corresponding classical bits.
    qc.measure(qr, cr)

    return qc

if __name__ == "__main__":
    # Build the circuit
    circuit_to_draw = create_4qubit_emergence_circuit_for_drawing()

    # Draw the circuit and save it as a PNG file
    # 'mpl' (matplotlib) is generally good for saving to file.
    # 'filename' specifies where to save the image.
    try:
        circuit_drawer(circuit_to_draw, output='mpl', filename='figure1_4qubit_emergence_circuit.png',
                       fold=-1) # fold=-1 prevents folding the circuit, showing it all on one line if possible
        print("Successfully generated 'figure1_4qubit_emergence_circuit.png'.")
        print("The image file should be in the same directory as this script.")
    except Exception as e:
        print(f"Error drawing or saving circuit: {e}")
        print("Please ensure you have matplotlib installed (`pip install matplotlib`)")
        print("and that your Qiskit environment is correctly set up.")

    # To display the plot interactively (optional, if running in a graphical environment)
    # plt.show()
