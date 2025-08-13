
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
shots_count = 1024 # Define shots count for consistency

# --- Quantum Circuit Definitions ---

def create_pure_conditional_circuit_8q():
    """
    Builds a purely quantum 8-qubit circuit for statevector simulation,
    implementing "conditional corrections" using controlled gates (no classical bits).
    This represents the quantum evolution that *would* happen if q0 were |1>.
    Initial state: q0, q1 in |0>, q2-q7 in |1>.
    Due to the CX-H chain starting from q0/q1, all individual qubits will become maximally mixed (entropy ~1).
    """
    qc = QuantumCircuit(8, name='pure_conditional_8q_circuit')
    
    # Initialize qubits 2-7 to |1> state as requested
    for i in range(2, 8):
        qc.x(i)
    
    # Step 1: Poke q0 (superposition)
    qc.h(0)
    
    # Step 2: Implement "conditional correction" on q1 using q0 as control.
    # If q0 is |1>, then apply X and Z to q1. This models the conditional logic
    # purely quantum mechanically for statevector simulation.
    qc.cx(0, 1) # If q0 is 1, flip q1
    qc.cz(0, 1) # If q0 is 1, apply Z to q1 (This is a simplified representation)
    
    # Propagate entanglement across the chain (q1 through q7)
    # These operations are not conditional.
    for i in range(1, 7):
        qc.cx(i, i + 1)
        qc.h(i + 1) # Apply H after CX to propagate entanglement

    return qc

def create_pure_simple_chain_circuit_8q():
    """
    Builds a purely quantum 8-qubit simple chain circuit for statevector simulation
    (no classical bits, no explicit conditional corrections).
    Initial state: q0, q1 in |0>, q2-q7 in |1>.
    This circuit is designed to result in: q0, q1 having entropy ~0, and q2-q7 having entropy ~1.
    """
    qc = QuantumCircuit(8, name='pure_simple_chain_8q_circuit')
    
    # Initialize qubits 2-7 to |1> state as requested
    for i in range(2, 8):
        qc.x(i)

    # Qubits 0 and 1 are left in their initial |0> state to maintain 0 entropy
    # The entanglement chain now starts from q2
    
    # Propagate entanglement across the chain (q2 through q7)
    # This will cause qubits 2-7 to become maximally mixed.
    for i in range(2, 7): # Start chain from q2
        qc.cx(i, i+1)
        qc.h(i+1) # Apply H after CX to propagate entanglement
    
    return qc

def build_qasm_conditional_chain_circuit(apply_correction=False, circuit_type_name="qasm_chain_circuit"):
    """
    Builds an 8-qubit circuit with classical measurement and conditional evolution via qc.if_test.
    This circuit is suitable for QASM simulation to get bitstring counts.
    
    Args:
        apply_correction (bool): If True, applies conditional X and Z gates on q1 based on q0's classical outcome.
        circuit_type_name (str): A name for the circuit.
    """
    qreg = QuantumRegister(8, 'q')
    creg = ClassicalRegister(1, 'c0') # For measuring q0 to apply condition
    qc = QuantumCircuit(qreg, creg, name=circuit_type_name)
    
    # NOTE: Qubits 0-7 are implicitly initialized to |0> state by Qiskit.
    # If specific initial states for QASM simulation are desired, X gates would be added here.
    # For now, we assume only q0 is "poked" initially for the conditional branch.

    # Step 1: Poke q0
    qc.h(qreg[0])
    
    # Step 2: Measure q0 into c[0]
    qc.measure(qreg[0], creg[0])

    # Step 3: Conditional corrections on q1 (if enabled) using if_test
    if apply_correction:
        with qc.if_test((creg, 1)): # Execute if classical register c0 has value 1
            qc.x(qreg[1]) # Apply X to q1
            qc.z(qreg[1]) # Apply Z to q1
    
    # Step 4: Propagate entanglement across the chain (q1 through q7)
    # These operations are not conditional.
    for i in range(1, 7):
        qc.cx(qreg[i], qreg[i + 1])
        qc.h(qreg[i + 1]) # Apply H after CX to propagate entanglement

    return qc

# --- Helper Function for Entropy Calculation from Statevector ---
def calculate_von_neumann_entropy_per_qubit(statevector, num_qubits):
    """Calculates Von Neumann entropy for each qubit from a given statevector."""
    entropies = []
    if not isinstance(statevector, Statevector):
        logging.error(f"Input to calculate_von_neumann_entropy_per_qubit is not a Statevector object: {type(statevector)}")
        raise TypeError("Input must be a Qiskit Statevector object.")
        
    for i in range(num_qubits):
        reduced_density_matrix = partial_trace(statevector, [j for j in range(num_qubits) if j != i])
        entropies.append(entropy(reduced_density_matrix, base=2))
    return entropies

# --- Helper Function to Run Statevector Sim and Compute Entropy ---
def run_statevector_and_entropy(circuit):
    """
    Runs a circuit (assumed to be purely quantum with no measurements)
    on a statevector simulator, retrieves the statevector,
    and computes Von Neumann entropy for each qubit.
    """
    state = None
    try:
        # For statevector simulation, we must ensure there are no measurements in the circuit
        # and that save_statevector() is the final instruction.
        qc_sv_only = circuit.copy()
        # Remove any existing classical registers and measurement instructions from the copy
        # (This should not be strictly necessary for the 'pure' circuits, but acts as a safeguard)
        qc_sv_only.remove_final_measurements(inplace=True)
        # Ensure it has a name, even if it's a copy
        if not qc_sv_only.name:
            qc_sv_only.name = f"{circuit.name}_sv_run"
            
        qc_sv_only.save_statevector() 
        
        sim_statevector = AerSimulator(method='statevector')
        tqc_statevector = transpile(qc_sv_only, sim_statevector)
        job_statevector = sim_statevector.run(tqc_statevector)
        result_statevector = job_statevector.result()
        
        state = result_statevector.get_statevector(tqc_statevector) 
        logging.info(f"Successfully retrieved statevector for {circuit.name}.")
        state = Statevector(state) 
        logging.info(f"Successfully cast state to qiskit.quantum_info.Statevector for {circuit.name}.")

        entropies = calculate_von_neumann_entropy_per_qubit(state, circuit.num_qubits)
        
    except Exception as e:
        logging.error(f"Error during statevector simulation or entropy calculation for {circuit.name}: {e}")
        logging.error(f"This is often due to Qiskit environment issues or attempting statevector sim on a circuit with mid-circuit classical conditions.")
        return [] 
    return entropies


# --- Main Execution Block ---
if __name__ == "__main__":
    from qiskit.visualization import plot_histogram 

    # Define circuits for STATEVECTOR simulation (pure quantum evolution)
    pure_quantum_circuits = {
        "conditional": create_pure_conditional_circuit_8q(),
        "simple_chain": create_pure_simple_chain_circuit_8q()
    }

    # Define circuits for QASM simulation (with classical conditional logic for histograms)
    qasm_circuits = {
        "conditional_qasm": build_qasm_conditional_chain_circuit(apply_correction=True, circuit_type_name='conditional_qasm_circuit'),
        "simple_chain_qasm": build_qasm_conditional_chain_circuit(apply_correction=False, circuit_type_name='simple_qasm_chain_circuit')
    }
    
    results = {} # Store results for both counts and entropies

    # --- Run Statevector Simulations for Entropies ---
    for name, circuit_obj in pure_quantum_circuits.items():
        logging.info(f"\n--- Processing {name} pure quantum circuit (8-Qubit) for Entropy ---")
        logging.info(f"Running statevector simulation and entropy calculation for {name}...")
        
        entropies = run_statevector_and_entropy(circuit_obj)
        if entropies:
            results[name + "_entropies"] = entropies
            print(f"\n--- VON NEUMANN ENTROPIES PER QUBIT (8-Qubit {name} Pure Quantum Simulated) ---")
            for i, s in enumerate(entropies):
                print(f"q{i}: {s:.4f}")
            print("---------------------------------------------------------")
        else:
            logging.warning(f"Skipping Von Neumann Entropy output for {name} due to prior errors.")
            results[name + "_entropies"] = [] # Store empty list if failed

    # --- Run QASM Simulations for Bitstring Counts (Histograms) ---
    for name_qasm, circuit_obj_qasm in qasm_circuits.items():
        logging.info(f"\n--- Processing {name_qasm} circuit (8-Qubit) for Counts ---")
        logging.info(f"Running QASM simulation for bitstring histogram for {name_qasm}...")
        
        qc_with_measures = circuit_obj_qasm.copy() 
        # For QASM simulation, ensure all qubits are measured for full bitstring counts
        qc_with_measures.measure_all() 

        sim_qasm = AerSimulator()
        tqc_qasm = transpile(qc_with_measures, sim_qasm)
        qasm_job = sim_qasm.run(tqc_qasm, shots=shots_count)
        qasm_result = qasm_job.result()
        qasm_counts = qasm_result.get_counts(qc_with_measures)
        results[name_qasm + "_counts"] = qasm_counts

        print(f"\n--- RAW COUNTS FOR PAPER (8-Qubit {name_qasm} Simulated Histogram) ---")
        print("Counts:")
        print(qasm_counts)
        print("----------------------------------------------------------")

    # --- Generate Plots for Paper ---
    logging.info("\n--- Generating plots for the paper (both Conditional and Simple Chain Circuits) ---")

    # Separate Histograms (Figure 2 type plots for each)
    for circuit_type_qasm in ['conditional_qasm', 'simple_chain_qasm']:
        counts_key = f"{circuit_type_qasm}_counts"
        display_name = "Conditional Chain" if "conditional" in circuit_type_qasm else "Simple Chain"
        if counts_key in results and results[counts_key]:
            fig_hist, ax_hist = plt.subplots(figsize=(12, 6))
            plot_histogram(results[counts_key], ax=ax_hist, title=f"Figure 2: Bitstring Frequencies - 8-Qubit {display_name} (Simulated QASM)")
            plt.tight_layout()
            plt.savefig(f"figure2_8qubit_{circuit_type_qasm}_histogram.png")
            print(f"Figure 2 (8-qubit {circuit_type_qasm} histogram) saved as figure2_8qubit_{circuit_type_qasm}_histogram.png")
            plt.close(fig_hist)
        else:
            logging.warning(f"No counts data for {circuit_type_qasm} circuit to generate Figure 2 histogram.")


    # --- Combined Entropy Line Plot (Figure 3 for paper) ---
    logging.info("\n--- Generating Combined Entropy Line Plot (Figure 3 for paper) ---")
    
    # Use the 'pure' circuit entropy results for this combined plot
    if "conditional_entropies" in results and results["conditional_entropies"] and \
       "simple_chain_entropies" in results and results["simple_chain_entropies"]:
        
        num_qubits = pure_quantum_circuits["conditional"].num_qubits # Both pure circuits have same num_qubits
        
        fig_combined_entropy, ax_combined_entropy = plt.subplots(figsize=(12, 7))
        
        ax_combined_entropy.plot(range(num_qubits), results["conditional_entropies"], 
                                 marker='o', linestyle='-', color='green', label='With Conditional Corrections (Pure Quantum)')
        ax_combined_entropy.plot(range(num_qubits), results["simple_chain_entropies"], 
                                 marker='s', linestyle='--', color='red', label='No Corrections (Pure Quantum Chain)')
        
        ax_combined_entropy.set_xlabel("Qubit Index")
        ax_combined_entropy.set_ylabel("Von Neumann Entropy (bits)")
        ax_combined_entropy.set_title("Figure 3 (Combined): Entanglement Entropy per Qubit (Pure Quantum Simulation)")
        ax_combined_entropy.set_xticks(range(num_qubits))
        ax_combined_entropy.set_ylim(bottom=-0.1, top=1.1) # Set y-limits to cover 0 to 1
        ax_combined_entropy.grid(True, linestyle='--', alpha=0.7)
        ax_combined_entropy.legend()
        plt.tight_layout()
        plt.savefig("figure3_8qubit_combined_entropy_plot.png")
        print("Combined Entropy Plot saved as figure3_8qubit_combined_entropy_plot.png")
        plt.close(fig_combined_entropy)
    else:
        logging.warning("Skipping combined entropy plot: Missing entropy data for one or both pure quantum circuits.")

    logging.info("\nComparison run complete. Review the printed results and generated plots.")
    logging.info("Remember to run the 9-qubit grid experiment next.")
