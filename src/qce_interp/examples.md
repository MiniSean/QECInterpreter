# QEC Interpreter

Repetition-5 Example
===


```python
import os
from qce_interp import (
    DataManager,
    QubitIDObj,
    ParityType,
    Surface17Layer,
    IErrorDetectionIdentifier,
    ILabeledErrorDetectionIdentifier,
    ErrorDetectionIdentifier,
    LabeledErrorDetectionIdentifier,
    ISyndromeDecoder,
    ILabeledSyndromeDecoder,
    Distance5LookupTableDecoder,
    LabeledSyndromeDecoder,
)
from qce_interp.definitions import UNITDATA_DIR
example_file_path = os.path.join(UNITDATA_DIR, 'example_repetition_code_distance_5.hdf5')
```

Construct Data Manager
===
This class is responsible to construct the experiment index kernel based on metadata such as the number of QEC-rounds, involved qubits, device layout, expected initial stabilizer parities (based on prepared data-qubit state) and whether heralded initialization is used.

Note: reading a large .hdf5 file might take a few minutes. (approx. 2 minutes for 6Gb file)


```python
data_manager: DataManager = DataManager.from_file_path(
    file_path=example_file_path,
    rounds=list(range(0, 8 + 1)),
    heralded_initialization=True,
    qutrit_calibration_points=True,
    involved_data_qubit_ids=[QubitIDObj('D7'), QubitIDObj('D4'), QubitIDObj('D5'), QubitIDObj('D6'), QubitIDObj('D3')],
    involved_ancilla_qubit_ids=[QubitIDObj('Z3'), QubitIDObj('Z1'), QubitIDObj('Z4'), QubitIDObj('Z2')],
    expected_parity_lookup={
        QubitIDObj('Z3'): ParityType.ODD,
        QubitIDObj('Z1'): ParityType.ODD, 
        QubitIDObj('Z4'): ParityType.ODD, 
        QubitIDObj('Z2'): ParityType.ODD,
    },
    device_layout=Surface17Layer(),
)
```

    Processing data file:   0%|          | 0/9 [00:00<?, ?it/s]

    Processing data file: 100%|██████████| 9/9 [00:00<00:00, 47.01it/s]
    

Construct Error Detection Identifier
===
This class follows the `IErrorDetectionIdentifier` interface and is responsible providing processed experimental data such as:
- Binary (or ternary) classification of qubit (qutrit) measurements. 
- Post-selecting (masking) data based on heralded initialization or leakage events (2nd-state readout).
- Stabilizer parities.
- Computed parities from final data-qubit measurements.
- Stabilizer defects.

Each method output follows the following convention:
Output arrays are 3D tensors (N, M, P) where,
- N is the number of measurement repetitions.
- M is the number of stabilizer repetitions.
- P is the number of qubit elements.
    Where:
    - S is the number of stabilizer qubits.
    - D is the number of data qubits.

Note: It is possible to wrap `IErrorDetectionIdentifier` with `LabeledErrorDetectionIdentifier` to obtain the same method outputs but formatted using `xarray.DataArray`.


```python
error_identifier: IErrorDetectionIdentifier = data_manager.get_error_detection_classifier(
    use_heralded_post_selection=True,
    use_computational_parity=True,
)

# Optional labeled wrapper
labeled_error_identifier: ILabeledErrorDetectionIdentifier = LabeledErrorDetectionIdentifier(
    error_identifier,
)
```

Construct Basic LUT Decoder
===
From this point we can write decoder functionality that uses `IErrorDetectionIdentifier` as an interface to the experiment data. The decoder requires no knowledge for example on post-selection and state classification.
The following LUT decoder class implements a general `ISyndromeDecoder`.


```python
decoder: ISyndromeDecoder = Distance5LookupTableDecoder(
    error_identifier,
)

# Optional labeled wrapper
labeled_decoder: ILabeledSyndromeDecoder = LabeledSyndromeDecoder(
    decoder
)
```

Visualization
===
Lets use this framework to visualize some commong metrics


```python
from qce_interp import StateAcquisitionContainer
from qce_interp.visualization import plot_state_classification

qubit_id = QubitIDObj('Z1')
state_classifier: StateAcquisitionContainer = data_manager.get_state_acquisition(qubit_id=qubit_id)
fig, ax = plot_state_classification(state_classifier=state_classifier)
ax.set_title(f'IQ plot qubit {qubit_id.id}')
```




    Text(0.5, 1.0, 'IQ plot qubit Z1')




    
![png](examples_files/examples_10_1.png)
    



```python
from qce_interp.visualization import plot_all_defect_rate

plot_all_defect_rate(
    error_identifier,
    included_rounds=data_manager.rounds[-1],
)
```




    (<Figure size 640x480 with 1 Axes>,
     <Axes: xlabel='QEC round', ylabel='Defect rate $\\langle d_i \\rangle$'>)




    
![png](examples_files/examples_11_1.png)
    



```python
from qce_interp.visualization import plot_state_evolution

plot_state_evolution(
    syndrome_decoder=decoder,
    target_state=[0, 1, 0, 1, 0],
    included_rounds=data_manager.rounds,
)
```




    (<Figure size 640x480 with 1 Axes>,
     <Axes: title={'center': 'Computational parity and refocusing corrected State'}, ylabel='Rounds'>)




    
![png](examples_files/examples_12_1.png)
    



```python
from qce_interp.visualization import plot_pij_matrix

plot_pij_matrix(
    error_identifier=error_identifier,
    included_rounds=data_manager.rounds,
)
```

    C:\Users\seanvandermeer\Documents\Repositories\QECInterpreter\src\qce_interp\visualization\plot_pij_matrix.py:21: RuntimeWarning: invalid value encountered in sqrt
      return 0.5 - np.sqrt(1 / 4 - numerator / denominator)
    




    (<Figure size 640x480 with 3 Axes>, <Axes: >)




    
![png](examples_files/examples_13_2.png)
    


## Quick Setup Guide

### Cloning the repository

1. Open your terminal or command prompt.
2. Clone the repository using:
```bash
git clone git@github.com:MiniSean/QECInterpreter.git
```
3. Navigate to the repository root:
```bash
cd QECInterpreter
```

### Setting up the virtual environment
1. Create a virtual environment in the repository root:
```bash
python -m venv .venv
```
2. Activate the virtual environment:
- On Windows:
```bash
.venv\Scripts\activate
```
- On Unix or MacOS:
```bash
source .venv/bin/activate
```

### Installing Dependencies
1. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running Jupyter Notebooks
1. If not already installed, install Jupyter Notebook:
```bash
pip install notebook
```
2. Launch Jupyter Notebook:
```bash
jupyter notebook
```
3. Your default web browser should open with the Jupyter interface. Navigate to  `QECInterpreter\src\qce_interp\examples.ipynb` file and open it.
4. Run the notebook cells as needed.


