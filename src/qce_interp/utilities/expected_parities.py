# -------------------------------------------
# Module containing functionality to construct (expected) parity from initial state.
# -------------------------------------------
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID
from qce_circuit.connectivity.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_circuit.language.intrf_declarative_circuit import InitialStateContainer
from qce_interp.interface_definitions.intrf_state_classification import ParityType
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier


def initial_state_to_expected_parity(initial_state: InitialStateContainer, parity_layout: ISurfaceCodeLayer, involved_data_qubit_ids: List[IQubitID], involved_ancilla_qubit_ids: List[IQubitID], inverse_parity: bool = False) -> Dict[IQubitID, ParityType]:
    # Data allocation
    result: Dict[IQubitID, ParityType] = {}
    parity_index_lookup: Dict[IQubitID, NDArray[np.int_]] = ErrorDetectionIdentifier.get_parity_index_lookup(
        parity_layout=parity_layout,
        involved_data_qubit_ids=involved_data_qubit_ids,
        involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
    )
    assert initial_state.distance == len(involved_data_qubit_ids), f"Expects initial state for all involved data qubits. Instead {initial_state.distance} out of {len(involved_data_qubit_ids)} are present."

    # Reshape to (N, D) array to fit staticmethod function
    initial_state_array = initial_state.as_array.reshape(1, -1)
    computed_parity: np.ndarray = ErrorDetectionIdentifier.calculate_computational_parity(
        array=initial_state_array,
        parity_index_lookup=parity_index_lookup,
        involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
    )
    n, s = computed_parity.shape
    assert n == 1, f"Expects initial state to be a single dimensional attribute."
    computed_parity = computed_parity.reshape(s)
    assert len(computed_parity) == len(involved_ancilla_qubit_ids), f"Expects parity to be defined for each involved ancilla qubit. Instead {len(computed_parity)} out of {len(involved_ancilla_qubit_ids)} are present."

    for qubit_id, parity in zip(involved_ancilla_qubit_ids, computed_parity):
        # Determine even/odd weight parity
        parity_group = parity_layout.get_parity_group(element=qubit_id)[0]
        odd_weight_stabilizer: bool = qubit_id == parity_group.ancilla_id and len(parity_group.data_ids) % 2 != 0
        # Switch parity
        switch_parity: bool = inverse_parity ^ odd_weight_stabilizer
        even_parity: bool = parity != +1 if switch_parity else parity == +1
        if even_parity:
            result[qubit_id] = ParityType.EVEN
        else:
            result[qubit_id] = ParityType.ODD
    return result
