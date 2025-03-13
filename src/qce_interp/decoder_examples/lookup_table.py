# -------------------------------------------
# Module describing implementation of lookup table decoder.
# https://arxiv.org/pdf/1703.04136.pdf
# -------------------------------------------
from abc import abstractmethod, ABCMeta
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from xarray import DataArray
from typing import Dict, List
from qce_interp.utilities.custom_exceptions import InterfaceMethodException
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_error_identifier import (
    IErrorDetectionIdentifier,
    DataArrayLabels,
)
from qce_interp.interface_definitions.intrf_syndrome_decoder import (
    ISyndromeDecoder,
    ILabeledSyndromeDecoder,
)
from qce_interp.interface_definitions.intrf_state_classification import IStateClassifierContainer


class LookupTableDecoder(ISyndromeDecoder, metaclass=ABCMeta):
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

    # region Interface Properties
    @property
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's"""
        return self._error_identifier.involved_qubit_ids

    @property
    def involved_stabilizer_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved stabilizer (ancilla) qubit-ID's"""
        return self._error_identifier.involved_stabilizer_qubit_ids

    @property
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's"""
        return self._error_identifier.involved_data_qubit_ids
    # endregion

    # region Class Properties
    @property
    @abstractmethod
    def binary_syndrome_lookup(self) -> Dict[tuple, tuple]:
        """:return: Syndrome lookup from stabilizer syndrome to data correction."""
        raise InterfaceMethodException

    @property
    def eigenvalue_syndrome_lookup(self) -> Dict[tuple, tuple]:
        """:return: Syndrome lookup from stabilizer syndrome to data correction. In eigen basis."""
        return {
            tuple(IStateClassifierContainer.binary_to_eigenvalue(np.asarray(key))): tuple(
                IStateClassifierContainer.binary_to_eigenvalue(np.asarray(value)))
            for key, value in self.binary_syndrome_lookup.items()
        }
    # endregion

    # region Class Constructor
    def __init__(self, error_identifier: IErrorDetectionIdentifier):
        self._error_identifier: IErrorDetectionIdentifier = error_identifier
    # endregion

    # region Interface Methods
    @lru_cache(maxsize=None)
    def get_binary_syndrome_corrections(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M(+1), D)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
        :return: Tensor of binary-syndromes at specific cycle.
        """
        # (N, M(+1), S) Binary projected acquisition
        defect_stabilizer_classification: NDArray[np.int_] = self._error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        n, m, _ = defect_stabilizer_classification.shape
        eigenvalue_syndrome_lookup: Dict[tuple, tuple] = self.eigenvalue_syndrome_lookup

        # Convert defect_stabilizer_classification to a list of tuples
        eigenvalue_syndromes = list(map(tuple, defect_stabilizer_classification.reshape(-1, defect_stabilizer_classification.shape[2])))

        # Use a list comprehension with the lookup table to generate the eigenvalue corrections
        eigenvalue_corrections = np.array([eigenvalue_syndrome_lookup[es] for es in eigenvalue_syndromes])

        # Reshape the result to (N, M(+1), D)
        d = eigenvalue_corrections.shape[-1]
        result = eigenvalue_corrections.reshape((n, m, d))

        return IStateClassifierContainer.eigenvalue_to_binary(result)

    @lru_cache(maxsize=None)
    def get_binary_projected_corrected(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-corrected at specific cycle.
        """
        # (N, 1, D)
        binary_projected_classification: NDArray[np.int_] = self._error_identifier.get_binary_projected_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        binary_syndrome_correction: NDArray[np.int_] = self.get_binary_syndrome_correction(cycle_stabilizer_count=cycle_stabilizer_count)
        # Pre-process
        eigenvalue_projected_classification = IStateClassifierContainer.binary_to_eigenvalue(binary_projected_classification)
        eigenvalue_syndrome_correction = IStateClassifierContainer.binary_to_eigenvalue(binary_syndrome_correction)
        # (N, 1, D)
        eigenvalue_projected_corrected = eigenvalue_projected_classification * eigenvalue_syndrome_correction
        # Correct for refocusing (bit-flips)
        if cycle_stabilizer_count % 2 == 0 and cycle_stabilizer_count != 0:
            eigenvalue_projected_corrected = eigenvalue_projected_corrected * -1
        # Post-process
        binary_projected_corrected: NDArray[np.int_] = IStateClassifierContainer.eigenvalue_to_binary(eigenvalue_projected_corrected)
        return binary_projected_corrected

    def get_fidelity(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """
        Output shape: (1)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Fidelity value of target state at specific cycle.
        """
        # (N, 1, D)
        binary_output: np.ndarray = self.get_binary_projected_corrected(cycle_stabilizer_count=cycle_stabilizer_count)
        n, _, d = binary_output.shape
        # (N, D)
        corrected_binary_output: np.ndarray = binary_output.reshape((n, d))
        equal_rows_count = np.sum(np.all(corrected_binary_output == target_state, axis=1))
        equal_fraction: float = equal_rows_count / len(corrected_binary_output)
        return equal_fraction
    # endregion


class LabeledSyndromeDecoder(ILabeledSyndromeDecoder, metaclass=ABCMeta):
    """
    Behaviour class, implementing ILabeledSyndromeDecoder interfaces.
    Extends LookupTableDecoder to format its outputs as labeled xarray.DataArray objects.
    Each method mirrors those in LookupTableDecoder but returns data with additional
    contextual information (coordinates and dimensions), enhancing data interpretability.
    """

    # region Interface Properties
    @property
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's"""
        return self._syndrome_decoder.involved_qubit_ids

    @property
    def involved_stabilizer_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved stabilizer (ancilla) qubit-ID's"""
        return self._syndrome_decoder.involved_stabilizer_qubit_ids

    @property
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's"""
        return self._syndrome_decoder.involved_data_qubit_ids
    # endregion

    # region Class Constructor
    def __init__(self, syndrome_decoder: ISyndromeDecoder):
        self._syndrome_decoder: ISyndromeDecoder = syndrome_decoder
    # endregion

    # region Interface Methods
    def get_binary_syndrome_corrections(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M(+1), D)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
        :return: Tensor of binary-syndromes at specific cycle.
        """
        return self._syndrome_decoder.get_binary_syndrome_corrections(cycle_stabilizer_count=cycle_stabilizer_count)

    def get_labeled_binary_syndrome_corrections(self, cycle_stabilizer_count: int) -> DataArray:
        """
        :return: xarray.DataArray of binary-syndromes at specific cycle.
        """
        # (N, M(+1), D)
        result: NDArray[np.int_] = self.get_binary_syndrome_corrections(cycle_stabilizer_count=cycle_stabilizer_count)
        n, m, d = result.shape

        measurements = range(n)
        stabilizer_repetitions = range(m)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_data_qubit_ids]

        data_array = DataArray(
            result,
            coords={
                DataArrayLabels.MEASUREMENT.value: measurements,
                DataArrayLabels.STABILIZER_REPETITION.value: stabilizer_repetitions,
                DataArrayLabels.QUBIT_ID.value: qubit_ids,
            },
            dims=[
                DataArrayLabels.MEASUREMENT.value,
                DataArrayLabels.STABILIZER_REPETITION.value,
                DataArrayLabels.QUBIT_ID.value,
            ],
        )

        return data_array

    def get_binary_projected_corrected(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-corrected at specific cycle.
        """
        return self._syndrome_decoder.get_binary_projected_corrected(cycle_stabilizer_count=cycle_stabilizer_count)

    def get_labeled_binary_projected_corrected(self, cycle_stabilizer_count: int) -> DataArray:
        """
        :return: xarray.DataArray of binary-corrected at specific cycle.
        """
        # (N, 1, D)
        result: NDArray[np.int_] = self.get_binary_projected_corrected(cycle_stabilizer_count=cycle_stabilizer_count)
        n, one, d = result.shape

        measurements = range(n)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_data_qubit_ids]

        data_array = DataArray(
            result,
            coords={
                DataArrayLabels.MEASUREMENT.value: measurements,
                DataArrayLabels.VALUE.value: range(one),
                DataArrayLabels.QUBIT_ID.value: qubit_ids,
            },
            dims=[
                DataArrayLabels.MEASUREMENT.value,
                DataArrayLabels.VALUE.value,
                DataArrayLabels.QUBIT_ID.value,
            ],
        )

        return data_array

    def get_labeled_binary_syndrome_correction(self, cycle_stabilizer_count: int) -> DataArray:
        """
        :return: xarray.DataArray of binary-corrections at specific cycle.
        """
        # (N, 1, D)
        result: NDArray[np.int_] = self.get_binary_syndrome_correction(cycle_stabilizer_count=cycle_stabilizer_count)
        n, one, d = result.shape

        measurements = range(n)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_data_qubit_ids]

        data_array = DataArray(
            result,
            coords={
                DataArrayLabels.MEASUREMENT.value: measurements,
                DataArrayLabels.VALUE.value: range(one),
                DataArrayLabels.QUBIT_ID.value: qubit_ids,
            },
            dims=[
                DataArrayLabels.MEASUREMENT.value,
                DataArrayLabels.VALUE.value,
                DataArrayLabels.QUBIT_ID.value,
            ],
        )

        return data_array

    def get_fidelity(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """
        Output shape: (1)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Fidelity value of target state at specific cycle.
        """
        return self._syndrome_decoder.get_fidelity(
            cycle_stabilizer_count=cycle_stabilizer_count,
            target_state=target_state,
        )

    def get_fidelity_uncertainty(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """:return: Uncertainty in Logical fidelity based on target state and stabilizer round-count."""
        return self._syndrome_decoder.get_fidelity_uncertainty(
            cycle_stabilizer_count=cycle_stabilizer_count,
            target_state=target_state,
        )
    # endregion


class Distance3LookupTableDecoder(LookupTableDecoder, ISyndromeDecoder):
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

    # region Class Properties
    @property
    def binary_syndrome_lookup(self) -> Dict[tuple, tuple]:
        """:return: Syndrome lookup from stabilizer syndrome to data correction."""
        result: Dict[tuple, tuple] = {
            (0, 0): (0, 0, 0),  # No error
            (0, 1): (0, 0, 1),  # error on D5
            (1, 0): (1, 0, 0),  # error on D7
            (1, 1): (0, 1, 0),  # error on D4
        }
        return result
    # endregion


class Distance5LookupTableDecoder(LookupTableDecoder, ISyndromeDecoder):
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

    # region Class Properties
    @property
    def binary_syndrome_lookup(self) -> Dict[tuple, tuple]:
        """:return: Syndrome lookup from stabilizer syndrome to data correction."""
        result: Dict[tuple, tuple] = {
            (0, 0, 0, 0): (0, 0, 0, 0, 0),  # No error
            (0, 0, 0, 1): (0, 0, 0, 0, 1),  # error on D3
            (0, 0, 1, 0): (0, 0, 0, 1, 1),  # error on D6 and D3
            (0, 1, 0, 0): (1, 1, 0, 0, 0),  # error on D7 and D4
            (1, 0, 0, 0): (1, 0, 0, 0, 0),  # error on D7
            (1, 0, 0, 1): (1, 0, 0, 0, 1),  # error on D3 and D7
            (1, 0, 1, 0): (0, 1, 1, 0, 0),  # error on D4 and D5
            (1, 1, 0, 0): (0, 1, 0, 0, 0),  # error on D4
            (0, 1, 0, 1): (0, 0, 1, 1, 0),  # error on D5 and D6
            (0, 1, 1, 0): (0, 0, 1, 0, 0),  # error on D5
            (0, 0, 1, 1): (0, 0, 0, 1, 0),  # error on D6
            (0, 1, 1, 1): (0, 0, 1, 0, 1),  # error on D5 and D3
            (1, 0, 1, 1): (1, 0, 0, 1, 0),  # error on D6 and D7
            (1, 1, 0, 1): (0, 1, 0, 0, 1),  # error on D4 and D3
            (1, 1, 1, 0): (1, 0, 1, 0, 0),  # error on D7 and D5
            (1, 1, 1, 1): (0, 1, 0, 1, 0),  # error on D4 and D6
        }
        return result
    # endregion

