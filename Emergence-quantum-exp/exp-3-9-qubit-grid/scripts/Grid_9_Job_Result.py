
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import os
import logging
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.visualization import plot_histogram
from qiskit.result import Counts # For robust counts handling
from qiskit.providers.jobstatus import JobStatus # For checking job status

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Authentication Setup ---
try:
    api_token = os.environ.get("QISKIT_IBM_TOKEN")
    api_channel = os.environ.get("QISKIT_IBM_CHANNEL", "ibm_cloud")
    api_instance = os.environ.get("QISKIT_IBM_INSTANCE")

    if api_token: api_token = api_token.strip('\'"')
    if api_channel: api_channel = api_channel.strip('\'"')
    if api_instance: api_instance = api_instance.strip('\'"')

    if not api_token:
        raise ValueError("QISKIT_IBM_TOKEN environment variable is not set.")
    # Instance is optional for quantum.ibm.com, but mandatory for ibm_cloud
    if not api_instance and api_channel == "ibm_cloud":
        raise ValueError("QISKIT_IBM_INSTANCE environment variable is not set for 'ibm_cloud' channel.")

    service = QiskitRuntimeService(
        token=api_token,
        channel=api_channel,
        instance=api_instance if api_instance else None
    )
    logging.info("IBM Quantum service initialized successfully.")

except Exception as e:
    logging.error(f"Service initialization error: {e}")
    exit(1)

# --- Job ID to retrieve ---
# IMPORTANT: Replace this with the actual Job ID obtained from your submission script.
JOB_ID = 'd1cmvr6c0o9c73apvqq0' # Updated to your completed 9-qubit job ID

logging.info(f"Retrieving job with ID: {JOB_ID}")

try:
    job = service.job(JOB_ID)
    
    # Get the job status and convert it to its string name for reliable comparison
    current_job_status_obj = job.status()
    current_job_status_name = current_job_status_obj.name if isinstance(current_job_status_obj, JobStatus) else str(current_job_status_obj)
    
    logging.info(f"Job status: {current_job_status_name}")

    # Ensure the job is DONE before attempting to retrieve results
    if current_job_status_name == JobStatus.DONE.name:
        job_result = job.result() # This is a PrimitiveResult object from SamplerV2
        logging.debug(f"Type of job_result: {type(job_result)}")
        logging.debug(f"Attributes of job_result: {dir(job_result)}")

        counts = {} # Initialize counts dictionary
        shots_count = 1024 # This should match the shots used in your 9-qubit grid submission

        # Try to get shots from job options if available for quasi_dists conversion
        try:
            runtime_options = job.options
            if runtime_options and hasattr(runtime_options, 'shots') and runtime_options.shots is not None:
                shots_count = runtime_options.shots
            elif runtime_options and isinstance(runtime_options, dict) and 'shots' in runtime_options:
                shots_count = runtime_options['shots']
        except Exception as e:
            logging.debug(f"Could not retrieve shots from job options, defaulting to {shots_count}: {e}")

        # --- Inspect and extract SamplerV2 results (Updated based on user's hint) ---
        try:
            # SamplerV2 PrimitiveResult objects typically have a '_pub_results' attribute
            # which is a list of results, one for each circuit submitted.
            if hasattr(job_result, '_pub_results') and job_result._pub_results:
                single_circuit_result = job_result._pub_results[0]
                logging.debug(f"Attributes of single_circuit_result: {dir(single_circuit_result)}")

                if hasattr(single_circuit_result, 'data'):
                    result_data_payload = single_circuit_result.data
                    logging.debug(f"Attributes of result_data_payload: {dir(result_data_payload)}")
                    logging.debug(f"Content of single_circuit_result.data: {result_data_payload}")

                    # PRIMARY ATTEMPT: Based on user's hint and previous error output, try .data.c.get_counts()
                    if hasattr(result_data_payload, 'c') and hasattr(result_data_payload.c, 'get_counts'):
                        counts = result_data_payload.c.get_counts()
                        logging.info("Counts retrieved from 'c' classical register via .get_counts().")
                    # FALLBACKS:
                    # Fallback to 'meas' if 'c' not found or doesn't have get_counts
                    elif hasattr(result_data_payload, 'meas') and hasattr(result_data_payload.meas, 'get_counts'):
                        counts = result_data_payload.meas.get_counts()
                        logging.info("Counts retrieved from 'meas' classical register via .get_counts().")
                    # Fallback for quasi_dists if they exist in data payload (less common for direct counts from measure_all)
                    elif hasattr(result_data_payload, 'quasi_dists') and result_data_payload.quasi_dists:
                        quasi_dists = result_data_payload.quasi_dists
                        quasi_counts = quasi_dists[0]
                        counts = {bitstring: int(round(prob * shots_count)) for bitstring, prob in quasi_counts.items()}
                        logging.info("Counts retrieved from quasi-probabilities (result_data_payload.quasi_dists).")
                    # Fallback for direct 'counts' attribute on data payload
                    elif hasattr(result_data_payload, 'counts'):
                        counts = result_data_payload.counts
                        logging.info("Counts retrieved from result_data_payload.counts.")
                    # Fallback for other named classical registers if 'c_reg' (e.g. 'c_reg' might be a generic placeholder)
                    elif hasattr(result_data_payload, 'c_reg') and hasattr(getattr(result_data_payload, 'c_reg'), 'get_counts'):
                        counts = getattr(result_data_payload, 'c_reg').get_counts()
                        logging.info("Counts retrieved from classical register 'c_reg' in result_data_payload.")
                    else:
                        logging.error(f"Found 'data' in single_circuit_result, but no expected counts format (c, meas, quasi_dists, counts, c_reg).")
                        logging.debug(f"Content of single_circuit_result.data: {result_data_payload}")
                        raise ValueError("Could not determine counts format from SamplerV2 result data payload.")
                else:
                    logging.error("'_pub_results' found, but no 'data' attribute in its first element.")
                    raise ValueError("Missing 'data' in single_circuit_result.")
            # Fallback: Direct quasi_dists on top-level PrimitiveResult (less common with measure_all on hardware, but possible)
            elif hasattr(job_result, 'quasi_dists') and job_result.quasi_dists:
                quasi_dists = job_result.quasi_dists
                quasi_counts = quasi_dists[0]
                counts = {bitstring: int(round(prob * shots_count)) for bitstring, prob in quasi_counts.items()}
                logging.info("Counts retrieved from job_result.quasi_dists directly.")
            # Final fallback if no _pub_results or quasi_dists are found on the main object
            else:
                logging.error(f"No known SamplerV2 result structure matched for counts retrieval.")
                logging.debug(f"Full job_result object: {job_result}")
                raise ValueError("Could not determine counts format from SamplerV2 result.")

        except Exception as e:
            logging.error(f"Failed to retrieve counts from SamplerV2 result: {e}")
            logging.debug(f"Type of job_result: {type(job_result)}")
            logging.debug(f"Attributes of job_result: {dir(job_result)}")
            exit()

        # --- Print Measurement Results ---
        if counts: # Only plot if counts were successfully retrieved
            print(f"\nRaw Counts: {counts}")
            print("Ordered measurement results:")
            for outcome, count in sorted(counts.items()):
                # For 9-qubit grid, just print the full bitstring outcome
                print(f"  {outcome}: {count}")

            # --- Plotting ---
            plt.figure(figsize=(12, 6)) # Adjust figure size for 9 qubits
            plot_histogram(counts, title=f"Figure 5: Measurement Results for 9-Qubit Grid (Job ID: {JOB_ID})") # Updated title
            plt.tight_layout() # Adjust layout to prevent clipping
            plt.savefig("figure5_9qubit_grid_histogram.png") # Save histogram
            print("Figure 5 (9-qubit grid histogram) saved as figure5_9qubit_grid_histogram.png")
            plt.close() # Close plot to prevent display issues in some environments
        else:
            logging.warning("No counts data available to plot histogram.")

    else:
        logging.warning(f"Job status is not DONE. Current status: {current_job_status_name}. Cannot retrieve results yet. Please wait or try a different Job ID.")

except Exception as e:
    logging.error(f"Error retrieving or processing job results for {JOB_ID}: {e}")
    logging.error("Ensure the job ID is correct and the job has completed successfully, or wait for it to complete.")
    exit(1)
