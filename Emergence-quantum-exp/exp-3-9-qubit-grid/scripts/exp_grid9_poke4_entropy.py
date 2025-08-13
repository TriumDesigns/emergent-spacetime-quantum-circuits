
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import os # Import os to get environment variables
import time # Import time for job monitoring loop
import logging

from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.providers.jobstatus import JobStatus # Needed for job monitoring
from qiskit.visualization import plot_histogram
from qiskit.result import Counts # For robust counts handling
from qiskit_aer import AerSimulator # Ensure AerSimulator is imported only from qiskit_aer
from qiskit.quantum_info import Statevector, partial_trace, entropy # For entropy calculation

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
shots_count = 1024 # Define shots count for consistency

# --- Authentication Configuration ---
# This script is configured for IBM Quantum instances provisioned on IBM Cloud (cloud.ibm.com).
# You MUST set the following environment variables BEFORE running this script:
#
# 1. QISKIT_IBM_TOKEN: Your IBM Cloud API Key.
#    (Find this on cloud.ibm.com: Manage > Access (IAM) > API keys. Create one if you don't have one.)
#
# 2. QISKIT_IBM_CHANNEL: Set this to "ibm_cloud".
#
# 3. QISKIT_IBM_INSTANCE: Your IBM Cloud Quantum service instance CRN (Cloud Resource Name).
#    (Find this on cloud.ibm.com: Navigation Menu > Resource list > Find your Quantum service instance > Click its row > Look for "CRN".)
#
# How to set environment variables in Windows Command Prompt (DO NOT use quotes around the values):
# set QISKIT_IBM_TOKEN=<YOUR_IBM_CLOUD_API_KEY_HERE>
# set QISKIT_IBM_CHANNEL=ibm_cloud
# set QISKIT_IBM_INSTANCE=<YOUR_IBM_CLOUD_QUANTUM_CRN_HERE>

try:
    # Safely retrieve and strip any accidental quotes from environment variables
    api_token = os.environ.get("QISKIT_IBM_TOKEN").strip('\'"') if os.environ.get("QISKIT_IBM_TOKEN") else None
    api_channel = os.environ.get("QISKIT_IBM_CHANNEL", "ibm_cloud").strip('\'"')
    api_instance = os.environ.get("QISKIT_IBM_INSTANCE").strip('\'"') if os.environ.get("QISKIT_IBM_INSTANCE") else None

    # Print the values being read for debugging purposes
    logging.debug(f"QISKIT_IBM_TOKEN (first 10 chars): {api_token[:10]}..." if api_token else "QISKIT_IBM_TOKEN: Not set")
    logging.debug(f"QISKIT_IBM_CHANNEL: '{api_channel}'")
    logging.debug(f"QISKIT_IBM_INSTANCE: '{api_instance}'")

    if not api_token:
        raise ValueError("QISKIT_IBM_TOKEN environment variable is not set. Please set your IBM Cloud API Key.")
    if not api_instance:
        raise ValueError("QISKIT_IBM_INSTANCE environment variable (CRN) is not set. Please set your IBM Cloud Quantum service instance CRN.")
    if api_channel != "ibm_cloud":
        raise ValueError(f"Invalid QISKIT_IBM_CHANNEL: '{api_channel}'. For IBM Cloud instances, it must be 'ibm_cloud'.")

    service = QiskitRuntimeService(
        token=api_token,
        channel=api_channel,
        instance=api_instance
    )
    logging.info("IBM Quantum service initialized successfully using environment variables.")

except ValueError as e:
    logging.error(f"Authentication setup error: {e}")
    logging.error("Please ensure QISKIT_IBM_TOKEN (IBM Cloud API Key), QISKIT_IBM_CHANNEL ('ibm_cloud'), and QISKIT_IBM_INSTANCE (CRN) environment variables are set correctly.")
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred during service initialization: {e}")
    exit()

# --- Quantum Circuit Definitions (3x3 Grid) ---
def create_grid_circuit_with_measurements():
    """
    Builds a 9-qubit (3x3 grid) quantum circuit with entanglement and a "poke"
    at the center, followed by measurements for QASM simulation.
    """
    num_qubits = 9 # For a 3x3 grid
    qc = QuantumCircuit(num_qubits, num_qubits) # 9 qubits, 9 classical bits for measurements

    # Define 2D grid connectivity (lattice)
    grid_connections = [
        (0, 1), (1, 2), # Row 0
        (3, 4), (4, 5), # Row 1
        (6, 7), (7, 8), # Row 2
        (0, 3), (3, 6), # Column 0
        (1, 4), (4, 7), # Column 1
        (2, 5), (5, 8)  # Column 2
    ]

    # Apply entangling gates to mimic spatial correlations
    # Applying H on the control and CX to entangle
    for q1, q2 in grid_connections:
        qc.h(q1) # Put control in superposition
        qc.cx(q1, q2)

    # Poke center (qubit 4) to simulate "mass" or initial energy
    qc.ry(np.pi / 3, 4)

    # IMPORTANT: Add measurements for real hardware/QASM execution.
    qc.measure(range(num_qubits), range(num_qubits)) # Measure all qubits into corresponding classical bits
    
    return qc

def create_pure_grid_circuit_for_entropy():
    """
    Builds a purely quantum 9-qubit (3x3 grid) circuit without measurements
    for statevector simulation and entropy calculation.
    """
    num_qubits = 9
    qc = QuantumCircuit(num_qubits, name='pure_grid_circuit')

    # Define 2D grid connectivity (lattice)
    grid_connections = [
        (0, 1), (1, 2), # Row 0
        (3, 4), (4, 5), # Row 1
        (6, 7), (7, 8), # Row 2
        (0, 3), (3, 6), # Column 0
        (1, 4), (4, 7), # Column 1
        (2, 5), (5, 8)  # Column 2
    ]

    # Apply entangling gates to mimic spatial correlations
    for q1, q2 in grid_connections:
        qc.h(q1) # Put control in superposition
        qc.cx(q1, q2)

    # Poke center (qubit 4) to simulate "mass" or initial energy
    qc.ry(np.pi / 3, 4)

    qc.save_statevector() # Save statevector at the end for retrieval
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
        qc_sv_only = circuit.copy()
        # Ensure save_statevector is the final instruction if not already present
        if not qc_sv_only.data or qc_sv_only.data[-1].operation.name != 'save_statevector':
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
    
    # --- Part 1: Statevector Simulation and Entropy (for Heatmap) ---
    logging.info("\n--- Running Statevector Simulation for 9-Qubit Grid Entropy (Heatmap) ---")
    pure_grid_circuit = create_pure_grid_circuit_for_entropy()
    entropies = run_statevector_and_entropy(pure_grid_circuit)

    if entropies:
        print("\n--- VON NEUMANN ENTROPIES PER QUBIT (9-Qubit Grid Simulated) ---")
        for i, s in enumerate(entropies):
            print(f"q{i}: {s:.4f}")
        print("---------------------------------------------------------")

        # Reshape entropies for 3x3 heatmap
        # Assuming qubits are arranged row-wise:
        # 0 1 2
        # 3 4 5
        # 6 7 8
        entropy_grid = np.array(entropies).reshape(3, 3)

        plt.figure(figsize=(8, 7))
        plt.imshow(entropy_grid, cmap='viridis', origin='upper', vmin=0, vmax=1)
        plt.colorbar(label='Von Neumann Entropy (bits)')
        plt.title("Figure 4: Entanglement Entropy Heatmap (9-Qubit Grid)")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.xticks([0, 1, 2], ['Q0/Q3/Q6', 'Q1/Q4/Q7', 'Q2/Q5/Q8']) # Example labels
        plt.yticks([0, 1, 2], ['Row 0 (Q0-2)', 'Row 1 (Q3-5)', 'Row 2 (Q6-8)'])
        
        # Annotate with entropy values
        for i in range(3):
            for j in range(3):
                plt.text(j, i, f'{entropy_grid[i, j]:.2f}', ha='center', va='center', color='white', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("figure4_9qubit_grid_entropy_heatmap.png")
        print("Figure 4 (9-qubit grid entropy heatmap) saved as figure4_9qubit_grid_entropy_heatmap.png")
        plt.close()
    else:
        logging.warning("Skipping entropy heatmap generation for 9-qubit grid due to prior errors.")


    # --- Part 2: QASM Simulation (for Bitstring Counts) ---
    logging.info("\n--- Running QASM Simulation for 9-Qubit Grid (Bitstring Counts) ---")
    qc_with_measurements = create_grid_circuit_with_measurements()

    # --- Backend Selection (for QASM simulation) ---
    backend = None
    try:
        logging.info("Attempting to find an active real quantum device with at least 9 qubits for QASM run...")
        
        active_backends = [
            b for b in service.backends(simulator=False, operational=True, min_num_qubits=9) 
            if b.status().operational 
        ]
        
        if active_backends:
            active_backends_with_jobs = []
            for bknd in active_backends:
                try:
                    status_info = bknd.status()
                    pending_jobs = getattr(status_info, 'pending_jobs', 9999) 
                    active_backends_with_jobs.append((pending_jobs, bknd))
                except Exception as e:
                    logging.warning(f"Could not get status for backend {bknd.name}: {e}. Skipping.")
                    continue

            if active_backends_with_jobs:
                active_backends_with_jobs.sort(key=lambda x: x[0]) 
                backend = active_backends_with_jobs[0][1] 
                logging.info(f"Running on backend (least busy real device): {backend.name}")
            else:
                logging.info("No active real quantum devices found that met criteria after full status check.")
        else:
            logging.info("No real quantum devices found that are operational with at least 9 qubits.")


        if backend is None:
            logging.info("Falling back to a local AerSimulator (QASM method) for bitstring counts if no real device could be found.")
            backend = AerSimulator() # Default to QASM method for counts
            logging.info(f"Selected fallback simulator: {backend.name}")
        
        if backend is None: # Final check before exit
            logging.critical("Fatal Error: Could not select any backend for QASM simulation. Exiting.")
            exit()

        logging.info(f"Selected backend for QASM simulation: {backend.name}") 

        # --- Execution using SamplerV2 ---
        tqc_qasm = transpile(qc_with_measurements, backend)

        sampler = SamplerV2(mode=backend) 

        job = sampler.run([tqc_qasm], shots=shots_count) 
        job_id = job.job_id()
        print("Job ID:", job_id)

        # --- Job Monitoring Loop ---
        print("Monitoring job status...")
        while True:
            try:
                # Get the JobStatus enum object
                job_status_enum = job.status() 
                # Convert to string for display, handling cases where it might unexpectedly be a string itself
                current_status_name = str(job_status_enum) 
                queue_pos_info = ""

                # Check queue position only if QUEUED and attribute exists
                if job_status_enum == JobStatus.QUEUED and hasattr(job, 'queue_position') and job.queue_position() is not None:
                    queue_pos_info = f" (Queue position: {job.queue_position()})"
                
                print(f"Job Status: {current_status_name}{queue_pos_info}")

                # Compare with JobStatus enum members directly
                if job_status_enum in [JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR]:
                    print(f"Job {job_id} detected as {current_status_name}. Finalizing script exit...")
                    time.sleep(1) 
                    break # Exit loop once job is done, cancelled, or errored
                
                time.sleep(15) 

            except Exception as e:
                print(f"Error checking job status: {e}. Retrying in 15 seconds...")
                time.sleep(15)


        if job.status() == JobStatus.ERROR: # This comparison is already correct (enum vs enum)
            print(f"Job failed with error: {job.error_message()}")
            exit()
        elif job.status() == JobStatus.CANCELLED: # This comparison is already correct (enum vs enum)
            print(f"Job {job_id} was cancelled.")
            exit()
        else: # JobStatus.DONE (This comparison is already correct - enum vs enum)
            print(f"Job {job_id} successfully completed.")

        result_obj = job.result()

        # --- Result Processing for Measurement-Based Runs ---
        counts = {} 
        try:
            if hasattr(result_obj.data, 'meas') and hasattr(result_obj.data.meas, 'get_counts'):
                counts = result_obj.data.meas.get_counts()
                logging.info("Counts retrieved from 'meas' classical register.")
            elif hasattr(result_obj, 'quasi_dists') and result_obj.quasi_dists:
                quasi_dists = result_obj.quasi_dists
                quasi_counts = quasi_dists[0] 
                counts = {bitstring: int(round(prob * shots_count)) for bitstring, prob in quasi_counts.items()}
                logging.info("Counts retrieved from quasi-probabilities.")
            elif hasattr(result_obj.data, 'counts'):
                counts = result_obj.data.counts
                logging.info("Counts retrieved from result.data.counts.")
            else:
                raise ValueError("Could not determine counts format from SamplerV2 result.")
        except Exception as e:
            logging.error(f"Failed to retrieve counts from SamplerV2 result: {e}")
            logging.debug(f"Type of result_obj: {type(result_obj)}")
            logging.debug(f"Attributes of result_obj: {dir(result_obj)}")
            exit()

        if counts:
            print(f"\nRaw Counts: {counts}")
            print("Ordered measurement results:")
            # Sort by bitstring (keys) to ensure consistent order
            for outcome, count in sorted(counts.items()): 
                # Qiskit output is usually like 'Q8 Q7 ... Q0' if using measure_all()
                # You might want to reverse the string for intuitive [q0 q1 ... q8] order
                # if you print individual qubits. For raw string, it's usually fine as is.
                print(f"  {outcome}: {count}")

            # Plot histogram for the QASM results
            plt.figure(figsize=(12, 6))
            plot_histogram(counts, title=f"Figure 5: Measurement Counts for 3x3 Grid on {backend.name}")
            plt.tight_layout()
            plt.savefig("figure5_9qubit_grid_histogram.png")
            print("Figure 5 (9-qubit grid histogram) saved as figure5_9qubit_grid_histogram.png")
            plt.close()
        else:
            logging.warning("No counts data available to plot histogram for QASM simulation.")

    except Exception as e:
        logging.critical(f"Critical error during QASM simulation or plotting: {e}")
        # Final fallback to local AerSimulator if possible for counts, if the main QASM path failed
        print("Attempting final fallback to local AerSimulator (QASM method) for counts analysis if previous run failed.")
        try:
            fallback_backend = AerSimulator()
            logging.info(f"Running on local Aer simulator: {fallback_backend.name}")
            
            # Use the circuit with qc.measure_all() for fallback simulation
            fallback_tqc = transpile(qc_with_measurements, fallback_backend)
            fallback_job = fallback_backend.run(fallback_tqc, shots=shots_count)
            fallback_result = fallback_job.result()
            fallback_counts = fallback_result.get_counts(fallback_tqc)
            
            print(f"\nRaw Counts (Fallback): {fallback_counts}")
            print("Ordered measurement results (Fallback):")
            for outcome, count in sorted(fallback_counts.items()):
                print(f"  {outcome}: {count}")
            
            plt.figure(figsize=(12, 6))
            plot_histogram(fallback_counts, title="Figure 5 (Fallback): Measurement Counts for 3x3 Grid (AerSimulator)")
            plt.tight_layout()
            plt.savefig("figure5_9qubit_grid_histogram_fallback.png")
            print("Figure 5 (fallback) saved as figure5_9qubit_grid_histogram_fallback.png")
            plt.close()
        except ImportError:
            logging.error("Error: Qiskit Aer simulator not found locally. Cannot proceed with fallback.")
        except Exception as e_fallback:
            logging.error(f"Error during fallback AerSimulator run: {e_fallback}. Cannot proceed with fallback.")

    logging.info("\n9-Qubit Grid Experiment complete. Review the printed results and generated plots.")
