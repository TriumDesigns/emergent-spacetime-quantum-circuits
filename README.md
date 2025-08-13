# emergent-spacetime-quantum-circuits

## Exploration of Emergent Spacetime Signatures from Minimal Quantum Circuits

This repository contains the complete Qiskit code, raw experimental data, and image generation scripts for a multi-experiment study investigating emergent physical phenomena in quantum systems. The research explores entropy signatures and bitstring distributions across various circuit architectures, with experiments conducted on IBM Quantum hardware and Qiskit Aer simulators.

This work serves as an experimental probe into the idea that fundamental physical phenomena, including spacetime and its properties, may emerge from underlying quantum information structures, such as entanglement. The findings are detailed in the associated arXiv paper: "[Your Paper Title Here]" (Link to arXiv paper will go here once published).

---

## Experiments Overview

This repository encompasses the code and data for three distinct quantum circuit experiments:

1.  **4-Qubit Emergence Circuit:**
    * Designed as a minimal toy model for emergent gravitational and gauge structures.
    * Executed on real IBM Quantum hardware.
    * Focuses on the interplay of entanglement, emergent structure, and hardware noise.

2.  **8-Qubit Linear Chain with Conditional Evolution:**
    * Simulated noiselessly using Qiskit Aer Simulator.
    * Probes ideal entanglement spread and the formation of low-entropy states under conditional evolution, providing a baseline for understanding ideal emergent behavior without hardware imperfections.

3.  **9-Qubit 2D Grid:**
    * Designed as a toy model for a spatial fabric or a complex quantum medium.
    * Executed on real IBM Quantum System hardware.
    * Explores multidimensional connectivity, spatial entanglement patterns, and their resilience to real machine noise.

---


### Folder Contents:

* **`exp-1-4-qubit-emergence/`**:
    * `images/`: Contains generated image files for the 4-qubit experiment.
    * `run_emergence_ibm.py`: Python script to define and run the 4-qubit quantum circuit.
    * `run_emergence_ibm_result.py`: Python script to fetch and process results from the IBM Quantum backend for the 4-qubit experiment.
    * `run_emergence_ibm  raw result counts and data.txt`: Text file containing the raw measurement counts from the 4-qubit experiment.
	* `4_qubit_emergence_circuit_diagram.py`: Python script to generate the circuit diagram of the 4-qubit circuit.

* **`exp-2-8-qubit-chain/`**:
    * `images/`: Contains generated image files for the 8-qubit experiment.
    * `exp_chain8_compare_conditional_vs_nocorrection.py`: Python script to define and run the 8-qubit quantum circuit (simulated).
    * `8_qubit_chain_conditional vs no correction raw results counts data.txt`: Text file containing the raw measurement counts from the 8-qubit experiment.

* **`exp-3-9-qubit-grid/`**:
    * `images/`: Contains generated image files for the 9-qubit experiment.
    * `exp_grid9_poke4_entropy.py`: Python script to define and run the 9-qubit quantum circuit.
    * `Grid_9_Job_Result.py`: Python script to fetch and process results from the IBM Quantum backend for the 9-qubit experiment.
    * `exp grid9 poke4 entropy raw result data.txt`: Text file containing the raw measurement counts from the 9-qubit experiment.

* **`7_fig_plot.py`**: A central Python script to regenerate all the plot images (`.png` files) for all three experiments from their respective raw data files.

---

## Getting Started

To run the scripts and regenerate the plots, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TriumDesigns/emergent-spacetime-quantum-circuits.git](https://github.com/TriumDesigns/emergent-spacetime-quantum-circuits.git)
    cd emergent-spacetime-quantum-circuits/emergence_quantum_exp
    ```

2.  **Install Dependencies:**
    Ensure you have Python installed. Then, install the necessary libraries:
    ```bash
    pip install qiskit qiskit-ibm-runtime matplotlib numpy
    ```

3.  **IBM Quantum Account (for real hardware experiments):**
    * If you wish to run the 4-qubit and 9-qubit experiments on real IBM Quantum hardware, you will need an IBM Quantum account and API token.
    * Save your API token:
        ```python
        from qiskit_ibm_runtime import QiskitRuntimeService
        # Replace 'YOUR_IBM_QUANTUM_TOKEN' with your actual token
        QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_IBM_QUANTUM_TOKEN', overwrite=True)
        ```
        (You only need to run this once on your system.)

4.  **Running Experiments (Optional - Raw data is already provided):**
    * To run the circuits and generate new raw data (this will submit jobs to IBM Quantum for hardware experiments, which may take time and consume credits):
        ```bash
        python 4_qubit_experiment/run_emergence_ibm.py
        python 8_qubit_experiment/exp_chain8_compare_conditional_vs_nocorrection.py
        python 9_qubit_experiment/exp_grid9_poke4_entropy.py
        ```
    * To fetch the results from the executed jobs (after they complete):
        ```bash
        python 4_qubit_experiment/run_emergence_ibm_result.py
        python 9_qubit_experiment/Grid_9_Job_Result.py
        ```
        These scripts will update the `*_raw_data.txt` files.

5.  **Generating Plots:**
    * To generate all plots from the existing (or newly fetched) raw data:
        ```bash
        python 7_fig_plot.py
        ```
    * The generated `.png` image files will be saved in their respective `plots/` subdirectories.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# MIT License

# Copyright (c) 2025 Ankitkumar Patel / Trium Designs Pvt Ltd

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


---

## Citation

If you use this code or data in your research, please cite the associated paper:

@article{Patel2025Emergent,
title={Emergent Spacetime Signatures from Minimal Quantum Circuits: Entanglement, Entropy, and Real-Hardware Evidence},
author={Patel, Ankitkumar},



---

**Contact:**
For any questions or collaborations, please open an issue in this repository or contact [Ankitkumar Patel/ankit.patel@triumdesigns.com if you wish to include it].