# -------------------------------------------
# Module describing implementation of Minimum Weight Perfect Matching decoders.
# Based on blossom algorithm.
# -------------------------------------------
import numpy as np
from numpy.typing import NDArray
import pymatching
import stim
from typing import Callable
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import IDecoder
from qce_interp.interface_definitions.intrf_state_classification import IStateClassifierContainer


class MWPMDecoder(IDecoder):
    """
    Behaviour class, implementing ILookupDecoder interfaces.
    Uses following convention:
    Output arrays are 3D tensors (N, M, P) where,
    - N is the number of measurement repetitions.
    - M is the number of stabilizer repetitions.
    - P is the number of qubit elements.
        Where:
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
    """

    # region Class Constructor
    def __init__(self, error_identifier: IErrorDetectionIdentifier, circuit_func: Callable[[int], stim.Circuit]):
        self._error_identifier: IErrorDetectionIdentifier = error_identifier
        self._circuit_func: Callable[[int], stim.Circuit] = circuit_func
    # endregion

    # region Interface Methods
    def get_fidelity(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """
        Output shape: (1)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Fidelity value of target state at specific cycle.
        """

        # (N, 1, D)
        binary_projected_classification: NDArray[np.int_] = self._error_identifier.get_binary_projected_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        if cycle_stabilizer_count % 2 == 0 and cycle_stabilizer_count != 0:
            binary_projected_classification = binary_projected_classification ^ 1
        # (N, M(+1), S)
        syndromes: np.ndarray = IStateClassifierContainer.eigenvalue_to_binary(self._error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=cycle_stabilizer_count))
        n, m, s = syndromes.shape
        # (N, M(+1) * S)
        syndromes = np.reshape(syndromes, (n, s * m))
        # MWPM
        matching_decoder: pymatching.Matching = self.get_decoder(cycle_stabilizer_count=cycle_stabilizer_count)

        # Logical init
        logical_init = np.sum(target_state) % 2
        # Logical outcome
        logical_outcome = np.sum(binary_projected_classification, axis=2) % 2
        # Logical correction
        logical_correction = matching_decoder.decode_batch(syndromes).astype(int)
        # Logical corrected
        logical_corrected = logical_outcome ^ logical_correction
        # Logical error
        logical_error = logical_init ^ logical_corrected

        # (N, D)
        equal_rows_count = np.sum(np.all(logical_error == 0, axis=1))
        equal_fraction: float = equal_rows_count / len(logical_error)
        return equal_fraction
    # endregion

    # region Class Methods
    def get_decoder(self, cycle_stabilizer_count: int) -> pymatching.Matching:
        # Construct circuit from definition
        stim_circuit: stim.Circuit = self._circuit_func(cycle_stabilizer_count)
        noisy_circuit = stim_circuit  # add_noise(stim_circuit)

        dem = noisy_circuit.detector_error_model(
            decompose_errors=True,
            allow_gauge_detectors=False,
            approximate_disjoint_errors=True,
        )
        mwpm = pymatching.Matching(
            graph=dem,
        )
        return mwpm
    # endregion
