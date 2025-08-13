
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Counts # For robust counts handling

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Authentication Setup ---
# This script is configured for IBM Quantum instances provisioned on IBM Cloud (cloud.ibm.com).
# You MUST set the following environment variables BEFORE running this script:
# QISKIT_IBM_TOKEN, QISKIT_IBM_CHANNEL ("ibm_cloud"), QISKIT_IBM_INSTANCE (CRN)
# Example: set QISKIT_IBM_TOKEN=YOUR_TOKEN_HERE

try:
    api_token = os.environ.get("QISKIT_IBM_TOKEN").strip('\'"') if os.environ.get("QISKIT_IBM_TOKEN") else None
    api_channel = os.environ.get("QISKIT_IBM_CHANNEL", "ibm_cloud").strip('\'"')
    api_instance = os.environ.get("QISKIT_IBM_INSTANCE").strip('\'"') if os.environ.get("QISKIT_IBM_INSTANCE") else None

    logging.info(f"DEBUG: QISKIT_IBM_TOKEN (first 10 chars): {api_token[:10]}..." if api_token else "DEBUG: QISKIT_IBM_TOKEN: Not set")
    logging.info(f"DEBUG: QISKIT_IBM_CHANNEL: '{api_channel}'")
    logging.info(f"DEBUG: QISKIT_IBM_INSTANCE: '{api_instance}'")

    if not api_token:
        raise ValueError("QISKIT_IBM_TOKEN environment variable is not set. Please set your IBM Cloud API Key.")
    if not api_instance:
        raise ValueError("QISKIT_IBM_INSTANCE environment variable (CRN) is not set. Please set your IBM Cloud Quantum service instance CRN.")
    if api_channel != "ibm_cloud":
        raise ValueError(f"Invalid QISKIT_IBM_CHANNEL: '{api_channel}'. It must be 'ibm_cloud' for IBM Cloud instances.")

    service = QiskitRuntimeService(
        token=api_token,
        channel=api_channel,
        instance=api_instance
    )
    logging.info("IBM Quantum service initialized successfully using environment variables.")

except ValueError as e:
    logging.error(f"Authentication setup error: {e}")
    logging.error("Please ensure QISKIT_IBM_TOKEN (IBM Cloud API Key), QISKIT_IBM_CHANNEL ('ibm_cloud'), and QISKIT_IBM_INSTANCE (CRN) environment variables are set correctly.")
    exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during service initialization: {e}")
    exit(1)

# --- Job ID to retrieve ---
# IMPORTANT: Replace this with the ACTUAL Job ID obtained from the submission script output.
JOB_ID = 'd1b8pkk7tq0c73dc9s30' # Example: 'd19scd96rndc73fagvs0'

logging.info(f"Retrieving job with ID: {JOB_ID}")

try:
    job = service.job(JOB_ID)
    
    current_job_status_obj = job.status()
    current_job_status_name = current_job_status_obj.name if isinstance(current_job_status_obj, JobStatus) else str(current_job_status_obj)
    
    logging.info(f"Job status: {current_job_status_name}")

    if current_job_status_name == JobStatus.DONE.name:
        job_result = job.result()
        logging.debug(f"Type of job_result: {type(job_result)}")
        logging.debug(f"Attributes of job_result: {dir(job_result)}")

        counts = {}
        # Assuming 1024 shots from the submission script, or try to get from job options
        shots_count = 1024 
        try:
            runtime_options = job.options
            if runtime_options and hasattr(runtime_options, 'shots') and runtime_options.shots is not None:
                shots_count = runtime_options.shots
            elif runtime_options and isinstance(runtime_options, dict) and 'shots' in runtime_options:
                shots_count = runtime_options['shots']
            logging.info(f"Using {shots_count} shots for calculations.")
        except Exception as e:
            logging.warning(f"Could not retrieve shots from job options, defaulting to {shots_count}: {e}")

        # --- Result Processing for SamplerV2 ---
        try:
            data_bin = None
            if hasattr(job_result, 'pub_results') and job_result.pub_results:
                data_bin = job_result.pub_results[0].data
                logging.debug("Accessed data via pub_results[0].data")
            elif hasattr(job_result, '_pub_results') and job_result._pub_results: # Fallback for underscored private attribute
                data_bin = job_result._pub_results[0].data
                logging.debug("Accessed data via _pub_results[0].data")
            elif hasattr(job_result, 'data'): # This might be for older versions or specific simulator outputs
                data_bin = job_result.data
                logging.debug("Accessed data via result.data (direct access).")

            if data_bin:
                if hasattr(data_bin, 'c_reg') and hasattr(getattr(data_bin, 'c_reg'), 'get_counts'):
                    counts = getattr(data_bin, 'c_reg').get_counts()
                    logging.info("Counts retrieved from 'c_reg' classical register.")
                elif hasattr(data_bin, 'quasi_dists') and data_bin.quasi_dists:
                    quasi_dists = data_bin.quasi_dists
                    quasi_counts = quasi_dists[0]
                    counts = {bitstring: int(round(prob * shots_count)) for bitstring, prob in quasi_counts.items()}
                    logging.info("Counts retrieved from quasi-probabilities.")
                elif hasattr(data_bin, 'counts'):
                    counts = data_bin.counts
                    logging.info("Counts retrieved from data_bin.counts.")
                else:
                    raise ValueError("Could not determine counts format from SamplerV2 DataBin.")
            else:
                raise ValueError("Could not find relevant data_bin in SamplerV2 result.")

        except Exception as e:
            logging.error(f"Failed to retrieve counts from SamplerV2 result: {e}")
            logging.debug(f"Type of job_result: {type(job_result)}")
            logging.debug(f"Attributes of job_result: {dir(job_result)}")
            exit(1)

        # --- Print Raw Counts for Paper ---
        print("\n--- RAW COUNTS FOR PAPER (4-Qubit Chain) ---")
        print("Job ID:", JOB_ID)
        print("Backend Used:", job.backend().name if hasattr(job, 'backend') and job.backend() else "N/A")
        print("Shots:", shots_count)
        print("Counts:")
        print(counts)
        print("------------------------------------------------")

        # --- Calculate Shannon Entropy per Qubit ---
        # Assuming a 4-qubit circuit for this script
        num_qubits = 4 
        qubit_marginals = {f'q{i}': {'0': 0, '1': 0} for i in range(num_qubits)}
        
        for bitstring, count in counts.items():
            # Bitstring is MSB to LSB. For 4 qubits: 'b3b2b1b0'
            # bitstring[0] is q3, bitstring[1] is q2, bitstring[2] is q1, bitstring[3] is q0
            for i in range(num_qubits):
                # Mapping bitstring index to qubit index (q0-q3)
                qubit_index = num_qubits - 1 - i 
                measured_bit = bitstring[i]
                qubit_marginals[f'q{qubit_index}'][measured_bit] += count

        shannon_entropies = {}
        for q_label, marginal_counts in qubit_marginals.items():
            p0 = marginal_counts['0'] / shots_count
            p1 = marginal_counts['1'] / shots_count
            
            entropy_val = 0
            if p0 > 0:
                entropy_val -= p0 * np.log2(p0)
            if p1 > 0:
                entropy_val -= p1 * np.log2(p1)
            shannon_entropies[q_label] = entropy_val

        print("\n--- SHANNON ENTROPIES PER QUBIT (4-Qubit Chain) ---")
        for q, s in sorted(shannon_entropies.items()):
            print(f"{q}: {s:.4f}")
        print("------------------------------------------------")

        # --- Plotting Histogram (Figure 1) ---
        if counts:
            fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
            backend_name = job.backend().name if hasattr(job, 'backend') and job.backend() else "Unknown Backend"
            plot_histogram(counts, ax=ax_hist, title=f"Figure 1: Measurement Counts for 4-Qubit Chain on {backend_name}")
            plt.tight_layout()
            plt.savefig("figure1_4qubit_histogram.png") # Save plot for paper
            print("\nFigure 1 (4-qubit histogram) saved as figure1_4qubit_histogram.png")
            plt.close(fig_hist)
        else:
            logging.warning("No counts data available to plot histogram.")

    else:
        logging.warning(f"Job status is not DONE. Current status: {current_job_status_name}. Cannot retrieve results yet. Please wait or try a different Job ID.")

except Exception as e:
    logging.error(f"Error retrieving or processing job results for {JOB_ID}: {e}")
    logging.error("Ensure the job ID is correct and the job has completed successfully, or wait for it to complete.")
    exit(1)
