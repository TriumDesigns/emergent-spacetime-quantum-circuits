
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from collections import Counter
import math

# --- ACTUAL DATA FOR 4-Qubit Emergence Circuit (Figure 2) ---
# Provided by user for Job ID: d1b8pkk7tq0c73dc9s30 on ibm_sherbrooke
num_qubits_4q = 4
num_shots_4q = 1024
actual_counts_4q_emergence = Counter({
    '1001': 9, '0011': 149, '0111': 156, '0010': 28, '0001': 42,
    '0100': 158, '1000': 57, '1111': 69, '0101': 34, '0000': 128,
    '0110': 38, '1010': 12, '1100': 67, '1110': 12, '1011': 57, '1101': 8
})


# --- ACTUAL DATA FOR 8-Qubit Chain Experiments (Figures 3, 4, 5) ---
num_qubits_8q = 8
num_shots_8q = 1024 # Confirmed from total counts

# Actual Counts for 8-qubit conditional chain experiment (Figure 3)
# Cleaned bitstrings by removing the trailing ' 0' or ' 1'
actual_counts_8q_conditional = Counter({
    '10000011': 8, '00100000': 10, '00010111': 12, '01001111': 11, '10000111': 10,
    '00010011': 6, '11011100': 13, '01110000': 10, '11010011': 7, '01000111': 8,
    '01111100': 11, '10111000': 8, '01111000': 13, '01101111': 6, '00000000': 7,
    '00010000': 12, '11101000': 12, '00110100': 9, '01101100': 9, '11110000': 11,
    '01010000': 11, '11011111': 10, '00111011': 9, '11010111': 12, '01011000': 9,
    '11100100': 6, '10001100': 10, '01000000': 6, '10111111': 11, '11010000': 14,
    '01110011': 5, '00110000': 7, '11001111': 6, '11111111': 6, '00101100': 5,
    '11001011': 14, '01000100': 5, '10101011': 9, '11001100': 12, '00101011': 9,
    '00110011': 6, '11111000': 5, '10100000': 11, '11100111': 12, '00010100': 4,
    '01101011': 11, '00011111': 10, '10011111': 9, '10101000': 14, '01011100': 7,
    '01001100': 6, '11000100': 6, '01100100': 8, '11000111': 12, '10001111': 12,
    '00001000': 9, '01011111': 3, '00000100': 9, '11110100': 8, '01101000': 11,
    '10011100': 5, '01000011': 15, '11100011': 5, '00011100': 8, '11010100': 8,
    '00111000': 9, '10100100': 6, '10111011': 4, '11111100': 4, '00101111': 9,
    '11101100': 8, '10000100': 6, '11101011': 11, '11110011': 4, '10101100': 7,
    '10010011': 13, '10100111': 8, '11000011': 9, '10100011': 6, '10010100': 12,
    '00111111': 10, '10011011': 5, '00000111': 9, '00100011': 3, '10000000': 5,
    '01010100': 7, '10010111': 8, '01111011': 5, '10110000': 9, '11110111': 9,
    '00011011': 11, '11011000': 10, '10111100': 9, '00000011': 10, '10010000': 6,
    '01001000': 9, '10110100': 5, '00100111': 7, '11001000': 5, '00110111': 7,
    '10110111': 5, '10101111': 5, '10011000': 5, '01111111': 5, '00001011': 5,
    '01110111': 9, '01011011': 8, '11111011': 3, '01100000': 7, '00001100': 8,
    '10110011': 8, '11000000': 6, '10001000': 9, '00101000': 6, '01100111': 6,
    '01100011': 5, '11100000': 6, '11011011': 6, '01010011': 7, '00001111': 6,
    '00100100': 7, '11101111': 8, '00111100': 6, '01010111': 11, '10001011': 5,
    '01110100': 6, '01001011': 4, '00011000': 4
})


# Actual Counts for 8-qubit simple chain experiment (Figure 4)
# Cleaned bitstrings by removing the trailing ' 0' or ' 1'
actual_counts_8q_simple = Counter({
    '00000100': 10, '10100000': 6, '11110001': 16, '01110100': 8, '00101100': 7,
    '10010101': 12, '01001100': 9, '10001100': 11, '11001100': 6, '01111101': 12,
    '10011101': 7, '01000100': 9, '11000001': 11, '00100101': 7, '01100001': 10,
    '00110101': 7, '10010000': 11, '00110001': 10, '11011100': 7, '01100000': 8,
    '01110101': 14, '00010100': 9, '11000101': 8, '10110000': 6, '10011001': 10,
    '10100100': 8, '01101001': 10, '01110000': 8, '01100101': 11, '00100001': 10,
    '01001101': 5, '01010101': 12, '11010101': 3, '00111001': 9, '00010101': 4,
    '11010000': 10, '10000001': 5, '10100101': 11, '11111001': 6, '01011100': 8,
    '01000001': 5, '11000100': 7, '10111100': 11, '01111000': 7, '10101101': 9,
    '10101100': 9, '11100001': 5, '11111000': 10, '00001101': 13, '10110100': 11,
    '01101000': 8, '01111001': 12, '01000101': 2, '00000101': 9, '00101101': 11,
    '10010100': 8, '01011000': 11, '11001001': 13, '00101001': 9, '00110100': 5,
    '11000000': 9, '10110101': 9, '01111100': 9, '10000101': 12, '00111000': 8,
    '00010001': 15, '10100001': 9, '01010001': 11, '11101100': 7, '11101000': 4,
    '11011101': 11, '11110101': 7, '00001000': 9, '00001100': 10, '00110000': 4,
    '10010001': 9, '11101001': 12, '10001101': 13, '00010000': 7, '11111100': 8,
    '01011001': 4, '11100101': 7, '00111100': 10, '11010100': 8, '10011000': 5,
    '11001000': 6, '01010000': 7, '10000100': 5, '10110001': 7, '01110001': 16,
    '00001001': 7, '11100100': 3, '01011101': 4, '10011100': 11, '00011101': 4,
    '01100100': 8, '10000000': 3, '00111101': 8, '11100000': 10, '10101001': 9,
    '00011000': 7, '00011001': 5, '01101101': 7, '11110000': 11, '10001000': 9,
    '11110100': 5, '01000000': 11, '00011100': 7, '10111001': 8, '10001001': 5,
    '11011001': 3, '01001000': 6, '01101100': 6, '10101000': 7, '01001001': 9,
    '11011000': 12, '11101101': 3, '10111000': 5, '00000001': 8, '00100100': 6,
    '00101000': 6, '11010001': 6, '01010100': 2, '00100000': 4, '10111101': 7,
    '00000000': 4, '11111101': 3, '11001101': 1
})


# Actual Von Neumann Entropies for 8-qubit simulated circuits (Figure 5)
# "With Conditional Corrections"
vn_entropies_8q_conditional = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]

# "No Corrections" (Simple Chain)
vn_entropies_8q_simple_chain = [0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]


# --- ACTUAL DATA FOR 9-Qubit 2D Grid Experiment (Figures 6, 7) ---
num_qubits_9q = 9
# Raw counts provided by the user for Job ID: d1cmvr6c0o9c73apvqq0
actual_counts_9q_grid = Counter({
    '011111101': 1, '010000111': 3, '110001011': 4, '101001110': 2, '110111100': 6, '010010101': 5, '111001100': 6, '101100100': 3, '011001111': 2, '101001001': 3, '111010000': 2, '111101000': 3, '101100111': 6, '111010001': 1, '010100100': 3, '100011011': 3, '010100010': 2, '000110010': 4, '110111001': 3, '011101111': 4, '100100100': 4, '101111111': 2, '111110000': 7, '001000011': 3, '100100101': 5, '010001100': 3, '001001110': 12, '011001000': 2, '100010101': 3, '010110101': 1, '010100111': 10, '001000101': 5, '011100010': 4, '011010101': 4, '100000111': 1, '010010110': 1, '101100101': 11, '001111001': 6, '010110111': 4, '101011010': 7, '100100001': 3, '111101111': 2, '111010111': 2, '010010001': 6, '100011010': 3, '110111010': 11, '111001110': 10, '001110010': 3, '001111101': 1, '100100111': 6, '000111101': 3, '110101100': 3, '010011010': 5, '111111001': 5, '110110111': 1, '101110111': 2, '011100111': 4, '001101100': 2, '110010110': 2, '001101110': 1, '000110011': 3, '101101101': 2, '110100110': 1, '011010001': 6, '100011001': 4, '001001101': 3, '000111011': 4, '011100101': 5, '111000110': 3, '001101000': 3, '001111100': 10, '000011010': 1, '100100110': 3, '011001010': 3, '101010010': 5, '000001011': 3, '000101001': 5, '110000010': 2, '111011000': 4, '110001110': 9, '001010111': 2, '000000101': 3, '011011010': 7, '111001111': 5, '111111011': 3, '101000000': 2, '000110100': 1, '010101110': 6, '111110001': 2, '010110000': 2, '111001000': 7, '011101011': 1, '101000100': 2, '110110010': 3, '100111001': 2, '110111111': 5, '000001110': 8, '110000101': 4, '111011010': 1, '011100000': 3, '001101011': 3, '010000010': 4, '101100110': 1, '111111010': 5, '110000000': 2, '110001010': 2, '100110000': 3, '001011010': 2, '011110000': 4, '101101000': 4, '100101110': 4, '000111100': 7, '001111010': 9, '011111110': 1, '001101010': 1, '111000111': 3, '001101111': 1, '110011000': 1, '100101010': 2, '010011001': 4, '000111000': 5, '111001101': 6, '110000111': 5, '100101100': 2, '000101111': 5, '110110000': 3, '110100000': 2, '010100101': 7, '111001011': 5, '000110101': 2, '100100011': 2, '001011000': 3, '101110001': 2, '001110101': 1, '011110001': 1, '010000110': 1, '110111101': 3, '001110011': 4, '000000111': 3, '010001001': 2, '000101010': 1, '011010000': 2, '000011111': 2, '100100000': 5, '011100001': 4, '000011000': 4, '101011000': 3, '000010000': 2, '001001000': 4, '100010100': 4, '001011011': 2, '011000111': 1, '000111010': 6, '100010010': 6, '000111111': 2, '000000110': 5, '000111001': 5, '011010010': 2, '100111100': 1, '001111011': 2, '110010000': 2, '001000111': 3, '001001001': 2, '110100101': 2, '110111011': 5, '011000101': 1, '111101010': 2, '010111010': 1, '000100001': 1, '101100001': 1, '011110100': 2, '110110001': 2, '110011100': 2, '110010101': 1, '011011001': 6, '101010110': 1, '111011111': 1, '001100011': 4, '110110101': 2, '001001011': 6, '010000101': 2, '101010000': 2, '110111000': 4, '100110100': 1, '101010001': 4, '100110101': 3, '010111001': 1, '111111100': 7, '111100010': 4, '101110000': 2, '010011101': 2, '000001111': 3, '111111111': 5, '100010001': 8, '011110010': 1, '010010010': 6, '110100111': 1, '101011001': 6, '100000101': 4, '010010111': 3, '111111101': 2, '001000000': 2, '000000100': 1, '001111110': 5, '101001101': 2, '010011111': 2, '111101100': 2, '100101111': 5, '110000011': 1, '001010000': 2, '001100000': 1, '100111111': 1, '101010011': 6, '101111100': 1, '011000010': 2, '111100111': 1, '100001000': 4, '101101011': 2, '000010111': 4, '000100000': 2, '011010111': 6, '011101000': 3, '011100100': 5, '011111001': 1, '011001011': 1, '001010100': 2, '000001010': 3, '000011110': 4, '110110110': 2, '111110011': 2, '100010111': 2, '001010011': 2, '010100000': 6, '010000011': 1, '100111011': 2, '001000110': 3, '101001010': 1, '011011100': 1, '111111000': 2, '110001000': 4, '011011111': 2, '011011011': 2, '110001111': 3, '111011101': 1, '111110010': 2, '001110111': 1, '110101001': 4, '011010011': 1, '011100011': 1, '110110100': 1, '110101010': 3, '000100011': 3, '111100101': 1, '010001010': 1, '101101001': 4, '101101100': 2, '101100010': 4, '101000101': 1, '110011001': 2, '110000110': 1, '001110110': 1, '111011011': 2, '000001101': 4, '101101111': 1, '000011001': 1, '010011110': 1, '011000000': 3, '010011011': 1, '000010010': 1, '010101111': 1, '111110101': 3, '001110001': 3, '100101001': 2, '011110111': 3, '100010000': 4, '000101000': 1, '101011101': 1, '110101000': 1, '111000100': 1, '011010100': 1, '000110110': 1, '110010111': 2, '111011110': 2, '100000100': 2, '101011111': 1, '010100110': 5, '100000010': 1, '001011001': 3, '111111110': 2, '001100111': 1, '111011001': 1, '010100011': 1, '111000000': 1, '010111100': 1, '000001001': 2, '000101100': 2, '001111000': 1, '100010110': 1, '101110011': 1, '011101110': 2, '011001110': 1, '000111110': 1, '001101101': 2, '100110110': 2, '111010011': 1, '111000101': 1, '001100010': 2, '001111111': 3, '011010110': 3, '101010101': 2, '000110000': 3, '101010100': 1, '010011100': 2, '111101001': 1, '010010011': 1, '010100001': 3, '110001100': 2, '001011100': 1, '100100010': 2, '001110100': 2, '101011011': 1, '101100000': 3, '101100011': 1, '111100000': 3, '101110101': 1, '101101010': 1, '011011110': 1, '100011111': 1, '001101001': 1, '010010000': 2, '000100010': 2, '111101011': 2, '010001101': 1, '100010011': 2, '011101001': 2, '000000000': 3, '011001101': 1, '001011110': 2, '101101110': 2, '110100001': 1, '100011101': 1, '100101000': 2, '100000011': 1, '110001001': 2, '001110000': 3, '111001001': 1, '101111000': 1, '100000001': 1, '101010111': 3, '000011101': 1, '001001111': 2, '011111111': 1, '010000000': 2, '010111000': 1, '101001100': 1, '000001000': 3, '001011111': 1, '010101101': 1, '010010100': 1, '100000000': 1, '100000110': 1, '110001101': 2, '010000001': 1, '110000100': 1, '101000010': 2, '110011110': 1, '010111110': 2, '011111100': 1, '110010100': 1, '000010011': 2, '001001010': 1, '110101110': 1, '110111110': 1, '001011101': 1, '111001010': 1, '000000010': 1, '101110100': 1, '010101011': 1, '010111101': 1, '011000001': 1, '110011101': 1
})
num_shots_9q = sum(actual_counts_9q_grid.values())


# --- Quantum Circuit Definition (Corrected 4-Qubit Emergence Circuit for Figure 1) ---
def create_4qubit_emergence_circuit_for_drawing():
    """
    Reconstructs the 4-qubit emergence circuit as defined in the user's
    experiment script for drawing Figure 1.
    """
    qr = QuantumRegister(4, 'q')
    cr = ClassicalRegister(4, 'c_reg') # Use 'c_reg' for consistency
    qc = QuantumCircuit(qr, cr)

    # L(G): gravitational entanglement
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])

    # T(G): gauge dynamics
    qc.rx(np.pi/4, qr[1]) # Corrected gate and qubit
    qc.h(qr[2])
    qc.cx(qr[2], qr[3])

    # Prepare “matter-like” qubit
    qc.ry(np.pi/3, qr[3]) # Corrected angle and placement

    # Controlled-Z (q1->q3) implemented as H-CX-H
    qc.h(qr[3])
    qc.cx(qr[1], qr[3])
    qc.h(qr[3])

    # Controlled-X (q2->q3)
    qc.cx(qr[2], qr[3])

    # Final measurements
    qc.measure(qr, cr)
    return qc


# --- Helper Functions ---

def calculate_shannon_entropy(counts, num_shots):
    """Calculates Shannon entropy for each qubit from measurement counts."""
    qubit_entropies = {}
    # Determine the number of qubits from the length of the bitstrings
    # Assuming all bitstrings have the same length
    if not counts:
        return {} # Return empty if no counts are provided

    num_qubits = len(next(iter(counts))) # Get length of the first bitstring

    for q_idx in range(num_qubits):
        prob_0 = 0
        prob_1 = 0
        # Qiskit bitstrings are ordered with qubit 0 on the right (LSB),
        # so bitstring[num_qubits - 1 - q_idx] accesses the correct qubit.
        for bitstring, count in counts.items():
            if bitstring[num_qubits - 1 - q_idx] == '0':
                prob_0 += count
            else:
                prob_1 += count

        p0 = prob_0 / num_shots if num_shots > 0 else 0
        p1 = prob_1 / num_shots if num_shots > 0 else 0

        entropy = 0
        if p0 > 0:
            entropy -= p0 * math.log2(p0)
        if p1 > 0:
            entropy -= p1 * math.log2(p1)
        qubit_entropies[f'q{q_idx}'] = entropy
    return qubit_entropies

def plot_histogram(counts, title, filename, num_shots, top_n_labels=20):
    """Generates and saves a bitstring frequency histogram, showing top N outcomes."""
    plt.figure(figsize=(12, 6))

    # Sort bitstrings by frequency in descending order
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    # Take only the top N outcomes for display
    display_counts = sorted_counts[:top_n_labels]
    
    # Sort these top N outcomes numerically for consistent display
    display_counts = sorted(display_counts, key=lambda item: int(item[0], 2))

    sorted_bitstrings = [item[0] for item in display_counts]
    frequencies = [item[1] / num_shots for item in display_counts]

    plt.bar(sorted_bitstrings, frequencies, color='skyblue')
    plt.xlabel("Bitstring Outcome")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(rotation=90, fontsize=8) # Always rotate for better visibility
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Generated {filename}")


# --- Figure Generation ---

# Figure 1: Full 4-Qubit Emergence Circuit (CORRECTED)
print("Generating Figure 1: Full 4-Qubit Emergence Circuit (Corrected)...")
qc_4q_figure1 = create_4qubit_emergence_circuit_for_drawing()

fig_circuit = qc_4q_figure1.draw('mpl', style={'backgroundcolor': '#ffffff', 'displaycolor': {
    'h': '#648fff', 'cx': '#dc267f', 'rx': '#ffb000', 'ry': '#fe6100', 'cz': '#785ef0', 'swap': '#648fff'
}}, scale=0.7)
fig_circuit.tight_layout()
fig_circuit.savefig('figure1_4qubit_emergence_circuit.png', dpi=300)
plt.close(fig_circuit)
print("Generated figure1_4qubit_emergence_circuit.png")


# Figure 2: Measurement outcome frequency histogram for the 4-qubit emergence circuit
print("Generating Figure 2: 4-Qubit Emergence Circuit Histogram...")
plot_histogram(
    actual_counts_4q_emergence,
    "Figure 2: Measurement outcome frequency histogram for the 4-qubit emergence circuit",
    "figure2_4qubit_emergence_histogram.png",
    num_shots_4q,
    top_n_labels=16 # For 4 qubits, all 16 bitstrings can be shown
)


# Figure 3: Bitstring frequency histogram for the 8-qubit conditional chain experiment (IMPROVED)
print("Generating Figure 3: 8-Qubit Conditional Chain Histogram (Improved)...")
plot_histogram(
    actual_counts_8q_conditional,
    "Figure 3: Top 20 Measurement Outcomes for 8-Qubit Conditional Chain",
    "figure3_8qubit_conditional_histogram.png",
    num_shots_8q,
    top_n_labels=20 # Show top 20 most frequent outcomes
)


# Figure 4: Bitstring frequency histogram for the 8-qubit simple chain experiment (IMPROVED)
print("Generating Figure 4: 8-Qubit Simple Chain Histogram (Improved)...")
plot_histogram(
    actual_counts_8q_simple,
    "Figure 4: Top 20 Measurement Outcomes for 8-Qubit Simple Chain",
    "figure4_8qubit_simple_histogram.png",
    num_shots_8q,
    top_n_labels=20 # Show top 20 most frequent outcomes
)


# Figure 5: Von Neumann Entropy per qubit for the 8-qubit simulated circuits (LINE CHART)
print("Generating Figure 5: 8-Qubit Von Neumann Entropy Comparison (Line Chart)...")
qubit_labels_8q = [f'q{i}' for i in range(8)]

plt.figure(figsize=(10, 6))
index = np.arange(len(qubit_labels_8q))

plt.plot(index, vn_entropies_8q_conditional, marker='o', linestyle='-', label='With Conditional Corrections', color='teal')
plt.plot(index, vn_entropies_8q_simple_chain, marker='x', linestyle='--', label='No Corrections (Simple Chain)', color='orange')

plt.xlabel("Qubit Index")
plt.ylabel("Von Neumann Entropy (bits)")
plt.title("Figure 5: Von Neumann Entropy per Qubit for 8-Qubit Simulated Circuits")
plt.xticks(index, qubit_labels_8q)
plt.ylim(-0.05, 1.05) # Adjusted y-limit to show 0 clearly
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figure5_8qubit_vn_entropy_comparison.png')
plt.close()
print("Generated figure5_8qubit_vn_entropy_comparison.png")


# Figure 6: Measurement outcome frequency histogram for the 9-qubit 2D grid experiment (IMPROVED)
print("Generating Figure 6: 9-Qubit Grid Histogram (Improved)...")
plot_histogram(
    actual_counts_9q_grid,
    "Figure 6: Top 20 Measurement Outcomes for 9-Qubit 2D Grid",
    "figure6_9qubit_grid_histogram.png",
    num_shots_9q,
    top_n_labels=20 # Show top 20 most frequent outcomes
)


# Figure 7: Shannon Entropy per qubit, spatially visualized on the 3x3 grid for the 9-qubit experiment (IMPROVED)
print("Generating Figure 7: 9-Qubit Shannon Entropy Heatmap (Improved)...")
# Calculate Shannon entropies for 9-qubit grid
shannon_entropies_9q_dict = calculate_shannon_entropy(actual_counts_9q_grid, num_shots_9q)
shannon_entropies_9q_list = [shannon_entropies_9q_dict[f'q{i}'] for i in range(num_qubits_9q)]

# Reshape to 3x3 grid for visualization
# Assuming a mapping like:
# q0 q1 q2
# q3 q4 q5
# q6 q7 q8
vn_entropies_9q_grid_reshaped = np.array(shannon_entropies_9q_list).reshape((3, 3))


plt.figure(figsize=(8, 7)) # Slightly adjusted figure size for better fit
plt.imshow(vn_entropies_9q_grid_reshaped, cmap='viridis', origin='upper', vmin=0, vmax=1)

# Add text annotations for entropy values AND qubit numbers
for i in range(vn_entropies_9q_grid_reshaped.shape[0]):
    for j in range(vn_entropies_9q_grid_reshaped.shape[1]):
        q_idx = i * 3 + j # Calculate qubit index based on grid position
        value = vn_entropies_9q_grid_reshaped[i, j]
        text_color = 'white' if value < 0.5 else 'black' # Choose text color based on background luminance
        plt.text(j, i, f'q{q_idx}\n{value:.2f}', ha='center', va='center', color=text_color, fontsize=12,
                 fontweight='bold')

plt.colorbar(label="Shannon Entropy (bits)")
plt.title("Figure 7: Shannon Entropy on 3x3 Qubit Grid", fontsize=14) # Reduced title font size
plt.xticks(np.arange(3), ['Column 1', 'Column 2', 'Column 3']) # Improved column labels
plt.yticks(np.arange(3), ['Row 1', 'Row 2', 'Row 3']) # Improved row labels
plt.tight_layout()
plt.savefig('figure7_9qubit_shannon_entropy_heatmap.png')
plt.close()
print("Generated figure7_9qubit_shannon_entropy_heatmap.png")

print("\nAll figures generated successfully using your provided experimental data.")
print("Note: For Figure 7, Shannon Entropy was calculated from measurement counts as a proxy for Von Neumann Entropy.")
