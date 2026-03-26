# -------------------------------------------
# Module describing implementation of majority voting (not really a decoder).
# https://arxiv.org/pdf/1703.04136.pdf
# -------------------------------------------
import numpy as np
from qce_circuit.connectivity.intrf_connectivity_surface_code import StabilizerType
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import IDecoder
from qce_interp.interface_definitions.intrf_state_classification import IStateClassifierContainer


class MajorityVotingDecoder(IDecoder):
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
    def __init__(self, error_identifier: IErrorDetectionIdentifier, initial_state_basis: StabilizerType = StabilizerType.STABILIZER_Z):
        self._error_identifier: IErrorDetectionIdentifier = error_identifier
        self._initial_state_basis: StabilizerType = initial_state_basis
    # endregion

    # region ILookupDecoder Interface Methods
    def get_fidelity(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """
        Output shape: (1)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Fidelity value of target state at specific cycle.
        """
        # (N, 1, D)
        binary_output: np.ndarray = self._error_identifier.get_binary_projected_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        n, _, d = binary_output.shape
        # (N, D)
        corrected_binary_output: np.ndarray = binary_output.reshape((n, d))
        # Correct for refocusing (bit-flips)
        if self._initial_state_basis == StabilizerType.STABILIZER_X:
            corrected_binary_output = corrected_binary_output ^ 1
        elif cycle_stabilizer_count % 2 == 0 and cycle_stabilizer_count != 0:
            corrected_binary_output = IStateClassifierContainer.binary_to_eigenvalue(corrected_binary_output) * -1
            corrected_binary_output = IStateClassifierContainer.eigenvalue_to_binary(corrected_binary_output)

        counter: int = 0
        for outcome in corrected_binary_output:
            majority_target: bool = np.sum(np.abs(outcome - target_state)) < d / 2.0
            if majority_target:
                counter += 1
        equal_fraction = counter / len(corrected_binary_output)
        return equal_fraction

    def get_fidelity_uncertainty(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """:return: Uncertainty in Logical fidelity based on target state and stabilizer round-count."""
        logical_fidelity: float = self.get_fidelity(
            cycle_stabilizer_count=cycle_stabilizer_count,
            target_state=target_state,
        )
        # (N, 1, D)
        binary_projected_classification: np.ndarray = self._error_identifier.get_binary_projected_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        number_of_samples: int = binary_projected_classification.shape[0]
        return float(np.sqrt(logical_fidelity * (1.0 - logical_fidelity) / number_of_samples))
    # endregion

