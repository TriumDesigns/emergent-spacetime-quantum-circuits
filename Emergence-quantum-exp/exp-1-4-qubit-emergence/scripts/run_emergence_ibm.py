
##############################################################################

# Project: Exploration of emergent spacetime signatures in quantum circuits.
# Author: Ankitkumar Patel
# Institution: Trium Designs Pvt Ltd
# License: MIT

##############################################################################

import numpy as np
import matplotlib.pyplot as plt # Keep for potential fallback plotting, though main script won't plot
import os
import time
import logging

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram # Keep for potential fallback plotting
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Counts # Keep for potential fallback result processing
from qiskit_aer import AerSimulator # Keep for fallback simulator

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Authentication Configuration ---
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
    exit()
except Exception as e:
    logging.error(f"An unexpected error occurred during service initialization: {e}")
    exit()

# --- Quantum Circuit Definition (4-Qubit Emergence Circuit) ---
def build_emergence_circuit():
    qr = QuantumRegister(4, 'q')
    cr = ClassicalRegister(4, 'c_reg') # Classical register named 'c_reg'
    qc = QuantumCircuit(qr, cr)

    # L(G): gravitational entanglement
    qc.h(qr[0])
    qc.cx(qr[0], qr[1])
    # T(G): gauge dynamics
    qc.rx(np.pi/4, qr[1])
    qc.h(qr[2])
    qc.cx(qr[2], qr[3])
    # Prepare “matter-like” qubit
    qc.ry(np.pi/3, qr[3])
    # Controlled-Z (q1->q3)
    qc.h(qr[3])
    qc.cx(qr[1], qr[3])
    qc.h(qr[3])
    # Controlled-X (q2->q3)
    qc.cx(qr[2], qr[3])

    qc.measure(qr, cr) # Measure all quantum bits into classical register 'c_reg'
    return qc

# --- Main execution block ---
qc = build_emergence_circuit()
shots_count = 1024 # Define shots count for consistency

backend = None

try:
    logging.info("Attempting to find an active real quantum device with at least 4 qubits...")
    
    active_backends = [
        b for b in service.backends(simulator=False, operational=True, min_num_qubits=4)
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
                logging.warning(f"Could not get status for backend {bknd.name}: {e}. Skipping for least busy calculation.")
                continue

        if active_backends_with_jobs:
            active_backends_with_jobs.sort(key=lambda x: x[0])
            backend = active_backends_with_jobs[0][1]
            logging.info(f"Running on backend (least busy real device): {backend.name}")
        else:
            logging.info("No active real quantum devices found that met criteria after full status check.")
    else:
        logging.info("No real quantum devices found that are operational with at least 4 qubits.")

    if backend is None:
        logging.info("Falling back to a local AerSimulator (QASM method) if no real device could be found.")
        backend = AerSimulator()
        logging.info(f"Selected fallback simulator: {backend.name}")
            
    logging.info(f"Selected backend: {backend.name}")

    # --- Execution using SamplerV2 ---
    tqc = transpile(qc, backend)

    sampler = SamplerV2(mode=backend)

    job = sampler.run([tqc], shots=shots_count)
    job_id = job.job_id()
    print("Job ID:", job_id)
    print(f"Backend used: {backend.name}")
    print(f"Shots: {shots_count}")

    # --- Job Monitoring Loop ---
    print("Monitoring job status...")
    while True:
        try:
            job_status_obj = job.status()
            # Convert status object/string to its name string for robust comparison
            current_status_name = job_status_obj.name if isinstance(job_status_obj, JobStatus) else str(job_status_obj) 

            queue_pos_info = ""
            # Use isinstance(job_status_obj, JobStatus) for queue position check
            if isinstance(job_status_obj, JobStatus) and job_status_obj == JobStatus.QUEUED and hasattr(job, 'queue_position') and job.queue_position() is not None:
                queue_pos_info = f" (Queue position: {job.queue_position()})"
            
            logging.info(f"Job Status: {current_status_name}{queue_pos_info}")

            # Check for final status using the string name for robustness
            if current_status_name in [JobStatus.DONE.name, JobStatus.CANCELLED.name, JobStatus.ERROR.name]:
                logging.info(f"Job {job_id} detected as {current_status_name}.")
                # Check for actual JobStatus.ERROR enum if possible, or rely on string name
                if current_status_name == JobStatus.ERROR.name:
                    print(f"Job failed with error: {job.error_message()}")
                print(f"\nJob {job_id} has completed with status: {current_status_name}.")
                print(f"Please use the result fetching script with Job ID: {job_id} to retrieve and analyze results.")
                time.sleep(1) # Small sleep for message flush
                break # Exit loop and script
            
            time.sleep(15)

        except Exception as e:
            logging.error(f"Error checking job status: {e}. Retrying in 15 seconds...")
            time.sleep(15)

except Exception as e:
    logging.error(f"Critical error during backend selection or job submission: {e}")
    logging.error("The script encountered a fatal error and could not complete the 4-qubit experiment submission.")
    exit()

