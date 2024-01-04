# -------------------------------------------
# Module describing implementation of indexing kernel.
# Specializes repetition-code kernel implementations.
# -------------------------------------------
from dataclasses import dataclass, field
from typing import List
import warnings
import numpy as np
from numpy.typing import NDArray
from qce_interp.interface_definitions.intrf_stabilizer_index_kernel import IIndexingKernel, IStabilizerIndexingKernel
from qce_interp.interface_definitions.intrf_stabilizer_index_kernel import (
    IIndexStrategy,
    FixedIndexStrategy,
    RelativeIndexStrategy,
)
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_state_classification import StateKey


@dataclass(frozen=True)
class QutritCalibrationIndexKernel(IIndexingKernel):
    """Data class, containing information about qutrit calibration-points (measurement acquisition) indexing."""
    heralded_initialization: bool
    """Boolean whether heralded initialization is performed at the start of the kernel."""
    index_offset_strategy: IIndexStrategy = field(repr=False)
    """Determines reference index by which internal indices are offset."""
    involved_qubit_ids: List[IQubitID] = field(repr=False)

    # region Interface Properties
    @property
    def start_index(self) -> int:
        """:return: Starting index."""
        return self.index_offset_strategy.get_index(self)  # Starts counting after previous index

    @property
    def _exclusive_start_index(self) -> int:
        """:return: Exclusive start index, used for counting Inclusive indices."""
        return self.start_index - 1  # -1 used to make sure all other indices are INCLUSIVE.

    @property
    def index_delta_heralded_initialization(self) -> int:
        """:return: Number of measurements performed during heralded initialization."""
        return 1 if self.heralded_initialization else 0

    @property
    def index_delta_state_0(self) -> int:
        """:return: Number of measurements performed during State-0 measurement."""
        return 1

    @property
    def index_delta_state_1(self) -> int:
        """:return: Number of measurements performed during State-1 measurement."""
        return 1

    @property
    def index_delta_state_2(self) -> int:
        """:return: Number of measurements performed during State-2 measurement."""
        return 1

    @property
    def stop_index(self) -> int:
        """:return: End index."""
        total_index_delta: int = 3 * self.index_delta_heralded_initialization + self.index_delta_state_0 + self.index_delta_state_1 + self.index_delta_state_2
        return self._exclusive_start_index + total_index_delta
    # endregion

    # region Interface Methods
    def contains(self, element: IQubitID) -> List[int]:
        """:return: Array-like of measurement indices corresponding to element within this indexing kernel."""
        heralded_0_measurement_indices: List[int] = self.get_heralded_state_0_measurement_index(element)
        heralded_1_measurement_indices: List[int] = self.get_heralded_state_1_measurement_index(element)
        heralded_2_measurement_indices: List[int] = self.get_heralded_state_2_measurement_index(element)
        state_0_measurement_indices: List[int] = self.get_state_0_measurement_index(element)
        state_1_measurement_indices: List[int] = self.get_state_1_measurement_index(element)
        state_2_measurement_indices: List[int] = self.get_state_2_measurement_index(element)
        return sorted(heralded_0_measurement_indices + heralded_1_measurement_indices + heralded_2_measurement_indices + state_0_measurement_indices + state_1_measurement_indices + state_2_measurement_indices)

    def get_heralded_state_0_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        If no heralded initialization is performed, return None.
        :return: (Optional) index corresponding to heralded measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        if not self.heralded_initialization:
            return []
        return [self._exclusive_start_index + self.index_delta_heralded_initialization]

    def get_heralded_state_1_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        If no heralded initialization is performed, return None.
        :return: (Optional) index corresponding to heralded measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        if not self.heralded_initialization:
            return []
        return [self._exclusive_start_index + 2 * self.index_delta_heralded_initialization + self.index_delta_state_0]

    def get_heralded_state_2_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        If no heralded initialization is performed, return None.
        :return: (Optional) index corresponding to heralded measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        if not self.heralded_initialization:
            return []
        return [self._exclusive_start_index + 3 * self.index_delta_heralded_initialization + self.index_delta_state_0 + self.index_delta_state_1]

    def get_state_0_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        :return: (Optional) index corresponding to State-0 measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        return [self._exclusive_start_index + self.index_delta_heralded_initialization + self.index_delta_state_0]

    def get_state_1_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        :return: (Optional) index corresponding to State-1 measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        return [self._exclusive_start_index + 2 * self.index_delta_heralded_initialization + self.index_delta_state_0 + self.index_delta_state_1]

    def get_state_2_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        :return: (Optional) index corresponding to State-2 measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        return [self._exclusive_start_index + 3 * self.index_delta_heralded_initialization + self.index_delta_state_0 + self.index_delta_state_1 + self.index_delta_state_2]
    # endregion


@dataclass(frozen=True)
class RepetitionIndexKernel(IIndexingKernel):
    """Data class, containing information about repeated-stabilizer (measurement acquisition) indexing."""
    nr_repeated_parities: int
    """Amount of repeated parity check rounds in this single kernel. Should be minimal number of 1"""
    heralded_initialization: bool
    """Boolean whether heralded initialization is performed at the start of the kernel."""
    index_offset_strategy: IIndexStrategy = field(repr=False)
    """Determines reference index by which internal indices are offset."""
    involved_data_qubit_ids: List[IQubitID] = field(repr=False)
    involved_ancilla_qubit_ids: List[IQubitID] = field(repr=False)

    # region Interface Properties
    @property
    def start_index(self) -> int:
        """:return: Starting index."""
        return self.index_offset_strategy.get_index(self)  # Starts counting after previous index

    @property
    def _exclusive_start_index(self) -> int:
        """:return: Exclusive start index, used for counting Inclusive indices."""
        return self.start_index - 1  # -1 used to make sure all other indices are INCLUSIVE.

    @property
    def index_delta_heralded_initialization(self) -> int:
        """:return: Number of measurements performed during heralded initialization."""
        return 1 if self.heralded_initialization else 0

    @property
    def index_delta_stabilizer_measurements(self) -> int:
        """:return: Number of measurements performed during stabilizer rounds."""
        return self.nr_repeated_parities - 1

    @property
    def index_delta_final_measurement(self) -> int:
        """:return: Number of measurements performed during final measurement."""
        return 1

    @property
    def stop_index(self) -> int:
        """:return: End index."""
        total_index_delta: int = self.index_delta_heralded_initialization + self.index_delta_stabilizer_measurements + self.index_delta_final_measurement
        return self._exclusive_start_index + total_index_delta

    @property
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of all involved qubit-ID's."""
        return self.involved_data_qubit_ids + self.involved_ancilla_qubit_ids

    # endregion

    # region Interface Methods
    def contains(self, element: IQubitID) -> List[int]:
        """:return: Array-like of measurement indices corresponding to element within this indexing kernel."""
        heralded_measurement_indices: List[int] = self.get_heralded_measurement_index(element)
        stabilizer_measurement_indices: List[int] = self.get_ordered_stabilizer_measurement_indices(element)
        final_measurement_indices: List[int] = self.get_final_measurement_index(element)
        return sorted(heralded_measurement_indices + stabilizer_measurement_indices + final_measurement_indices)
    # endregion

    # region Class Methods
    def get_heralded_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        If no heralded initialization is performed, return None.
        :return: (Optional) index corresponding to heralded measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        if not self.heralded_initialization:
            return []
        return [self._exclusive_start_index + self.index_delta_heralded_initialization]

    def get_ordered_stabilizer_measurement_indices(self, element: IQubitID) -> List[int]:
        """
        If element not part of stabilizers (ancilla qubits) in this kernel, return None.
        :return: (Optional) array-like of measurement indices corresponding to element within kernel.
        """
        if element not in self.involved_ancilla_qubit_ids:
            return []
        if self.nr_repeated_parities == 1:
            return []
        index_array: np.ndarray = np.asarray(range(1, self.nr_repeated_parities))
        return list(self._exclusive_start_index + self.index_delta_heralded_initialization + index_array)

    def get_final_measurement_index(self, element: IQubitID) -> List[int]:
        """
        If element not part of this kernel, return None.
        :return: (Optional) index corresponding to final measurement.
        """
        if element not in self.involved_qubit_ids:
            return []
        return [self._exclusive_start_index + self.index_delta_heralded_initialization + self.index_delta_stabilizer_measurements + self.index_delta_final_measurement]

    def __post_init__(self):
        if self.nr_repeated_parities < 1:
            warnings.warn(f"Expects number of repeated parities to be at least 1. Instead: {self.nr_repeated_parities}.")
    # endregion


class RepetitionExperimentKernel(IStabilizerIndexingKernel):
    """Behaviour class, containing gettable methods for various acquisition indexing."""

    # region Interface Properties
    @property
    def start_index(self) -> int:
        """:return: Starting index."""
        return self.indexing_kernels[0].start_index

    @property
    def stop_index(self) -> int:
        """:return: End index."""
        total_index_size: int = self.experiment_repetitions * self.kernel_cycle_length
        return self.start_index + total_index_size

    @property
    def kernel_cycle_length(self) -> int:
        """:return: Integer length of indexing kernel cycle."""
        exclusive_cycle_length: int = self.indexing_kernels[-1].stop_index - self.indexing_kernels[0].start_index
        inclusive_cycle_length: int = exclusive_cycle_length + 1
        return inclusive_cycle_length

    @property
    def experiment_repetitions(self) -> int:
        """Number of repetitions for this experiment."""
        return self._repetitions
    # endregion

    # region Class Properties
    @property
    def indexing_kernels(self) -> List[IIndexingKernel]:
        """:return: Array-like of ordered indexing kernels that describe self."""
        repetition_kernels: List[IIndexingKernel] = self._repetition_kernels
        calibration_kernel: List[IIndexingKernel] = [self._calibration_kernel]
        result: List[IIndexingKernel] = repetition_kernels + calibration_kernel
        return result
    # endregion

    # region Class Constructor
    def __init__(self, rounds: List[int], heralded_initialization: bool, qutrit_calibration_points: bool, involved_data_qubit_ids: List[IQubitID], involved_ancilla_qubit_ids: List[IQubitID], experiment_repetitions: int):
        self._rounds: List[int] = rounds
        """Array-like of integers corresponding to number of repetitions. Each integer element represents a separate RepetitionIndexKernel."""
        self._heralded_initialization: bool = heralded_initialization
        """Boolean whether heralded initialization is performed at the start of the kernel."""
        self._qutrit_calibration_points: bool = qutrit_calibration_points
        """Boolean whether (state-0, -1 and -2) calibration points are included."""
        self._involved_data_ids: List[IQubitID] = involved_data_qubit_ids
        self._involved_ancilla_ids: List[IQubitID] = involved_ancilla_qubit_ids
        """Array-like of involved data- and ancilla-qubit-ID's."""
        self._repetitions: int = experiment_repetitions
        """Total length of data indices, including all experiment repetitions. Size of dataset."""
        self._repetition_kernels: List[RepetitionIndexKernel] = []
        for nr_round in self._rounds:
            # Uses this index as excluded start to count forward.
            offset_strategy: IIndexStrategy = FixedIndexStrategy(index=0)
            if self._repetition_kernels:  # Not empty
                offset_strategy = RelativeIndexStrategy(reference_index_kernel=self._repetition_kernels[-1])
            # Append indexing kernel
            kernel: RepetitionIndexKernel = RepetitionIndexKernel(
                nr_repeated_parities=nr_round,
                heralded_initialization=self._heralded_initialization,
                index_offset_strategy=offset_strategy,
                involved_data_qubit_ids=self._involved_data_ids,
                involved_ancilla_qubit_ids=self._involved_ancilla_ids,
            )
            self._repetition_kernels.append(kernel)
        self._calibration_kernel: QutritCalibrationIndexKernel = QutritCalibrationIndexKernel(
            heralded_initialization=self._heralded_initialization,
            index_offset_strategy=RelativeIndexStrategy(reference_index_kernel=self._repetition_kernels[-1]),
            involved_qubit_ids=self._involved_data_ids + self._involved_ancilla_ids,
        )
    # endregion

    # region Interface Methods
    def contains(self, element: IQubitID) -> List[int]:
        """:return: Array-like of measurement indices corresponding to element within this indexing kernel."""
        raise NotImplemented

    def get_projected_calibration_acquisition_indices(self, qubit_id: IQubitID, state: StateKey) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param state: Identifier for state specific selectivity.
        :return: Tensor of indices pointing at all projection acquisition within calibration points.
        """
        if state == StateKey.STATE_0:
            single_cycle_indices: List[int] = self._calibration_kernel.get_state_0_measurement_index(element=qubit_id)
        elif state == StateKey.STATE_1:
            single_cycle_indices: List[int] = self._calibration_kernel.get_state_1_measurement_index(element=qubit_id)
        elif state == StateKey.STATE_2:
            single_cycle_indices: List[int] = self._calibration_kernel.get_state_2_measurement_index(element=qubit_id)
        else:
            raise NotImplemented(f"Calibration indices for state: {state}, is not implemented.")
        return self.create_sliced_array(single_cycle_indices, self.kernel_cycle_length, self.experiment_repetitions)

    def get_heralded_calibration_acquisition_indices(self, qubit_id: IQubitID, state: StateKey) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param state: Identifier for state specific selectivity.
        :return: Tensor of indices pointing at all heralded acquisition before calibration points.
        """
        if state == StateKey.STATE_0:
            single_cycle_indices: List[int] = self._calibration_kernel.get_heralded_state_0_measurement_index(element=qubit_id)
        elif state == StateKey.STATE_1:
            single_cycle_indices: List[int] = self._calibration_kernel.get_heralded_state_1_measurement_index(element=qubit_id)
        elif state == StateKey.STATE_2:
            single_cycle_indices: List[int] = self._calibration_kernel.get_heralded_state_2_measurement_index(element=qubit_id)
        else:
            raise NotImplemented(f"Calibration conditional indices for state: {state}, is not implemented.")
        return self.create_sliced_array(single_cycle_indices, self.kernel_cycle_length, self.experiment_repetitions)

    def get_heralded_cycle_acquisition_indices(self, qubit_id: IQubitID, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param cycle_stabilizer_count: Identifies the indices to only include cycles with this number of stabilizers.
        :return: Tensor of indices pointing at all heralded acquisition before stabilizer cycles.
        """
        for repetition_kernel in self._repetition_kernels:
            if repetition_kernel.nr_repeated_parities == cycle_stabilizer_count:
                heralded_indices = repetition_kernel.get_heralded_measurement_index(element=qubit_id)
                return self.create_sliced_arrays(heralded_indices, self.kernel_cycle_length, self.experiment_repetitions)
        return np.asarray([])  # If kernel with specific number of repeated parities is not found

    def get_stabilizer_acquisition_indices(self, qubit_id: IQubitID, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Note: Includes final acquisition.
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param cycle_stabilizer_count: Identifies the indices to only include cycles with this number of stabilizers.
        :return: Tensor of indices pointing at all stabilizer acquisition withing stabilizer cycles.
        """
        for repetition_kernel in self._repetition_kernels:
            if repetition_kernel.nr_repeated_parities == cycle_stabilizer_count:
                stabilizer_measurement_indices = repetition_kernel.get_ordered_stabilizer_measurement_indices(element=qubit_id)
                final_measurement_indices = repetition_kernel.get_final_measurement_index(element=qubit_id)
                return self.create_sliced_arrays(stabilizer_measurement_indices + final_measurement_indices, self.kernel_cycle_length, self.experiment_repetitions)
        return np.asarray([])  # If kernel with specific number of repeated parities is not found

    def get_projected_cycle_acquisition_indices(self, qubit_id: IQubitID, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param cycle_stabilizer_count: Identifies the indices to only include cycles with this number of stabilizers.
        :return: Tensor of indices pointing at all projection acquisition after stabilizer cycles.
        """
        for repetition_kernel in self._repetition_kernels:
            if repetition_kernel.nr_repeated_parities == cycle_stabilizer_count:
                final_measurement_indices = repetition_kernel.get_final_measurement_index(element=qubit_id)
                return self.create_sliced_arrays(final_measurement_indices, self.kernel_cycle_length,
                                                 self.experiment_repetitions)
        return np.asarray([])  # If kernel with specific number of repeated parities is not found
    # endregion

    # region Static Class Methods
    @staticmethod
    def estimate_experiment_repetitions(rounds: List[int], heralded_initialization: bool, qutrit_calibration_points: bool, dataset_size: int) -> int:
        """:return: (Calculation) estimation for the number of experiment repetition based on total dataset size."""
        repetition_kernels: List[RepetitionIndexKernel] = []
        for nr_round in rounds:
            # Uses this index as excluded start to count forward.
            offset_strategy: IIndexStrategy = FixedIndexStrategy(index=0)
            if repetition_kernels:  # Not empty
                offset_strategy = RelativeIndexStrategy(reference_index_kernel=repetition_kernels[-1])
            # Append indexing kernel
            kernel: RepetitionIndexKernel = RepetitionIndexKernel(
                nr_repeated_parities=nr_round,
                heralded_initialization=heralded_initialization,
                index_offset_strategy=offset_strategy,
                involved_data_qubit_ids=[],
                involved_ancilla_qubit_ids=[],
            )
            repetition_kernels.append(kernel)
        # Combined indexing kernels
        indexing_kernels: List[IIndexingKernel] = repetition_kernels

        if qutrit_calibration_points:
            calibration_kernel: QutritCalibrationIndexKernel = QutritCalibrationIndexKernel(
                heralded_initialization=heralded_initialization,
                index_offset_strategy=RelativeIndexStrategy(reference_index_kernel=repetition_kernels[-1]),
                involved_qubit_ids=[],
            )
            indexing_kernels += [calibration_kernel]

        # Integer length of indexing kernel cycle
        exclusive_cycle_length: int = indexing_kernels[-1].stop_index - indexing_kernels[0].start_index
        inclusive_cycle_length: int = exclusive_cycle_length + 1

        nr_experiment_repetitions: int = int(dataset_size / inclusive_cycle_length)
        assert dataset_size == nr_experiment_repetitions * inclusive_cycle_length, f"Expects cycle-length * repetition number to be equal to total index size. By definition."
        return nr_experiment_repetitions

    # TODO: Move this method to more general array transformation module
    @staticmethod
    def create_sliced_arrays(int_list: List[int], cycle_length: int, repetitions: int) -> NDArray[np.int_]:
        """Generate an array-like of numpy array by repeating and offsetting a list of integers based on cycle length and repetitions."""
        return np.asarray([np.array(int_list) + i * cycle_length for i in range(repetitions)])

    # TODO: Move this method to more general array transformation module
    @staticmethod
    def create_sliced_array(int_list: List[int], cycle_length: int, repetitions: int) -> NDArray[np.int_]:
        """Generate a numpy array by repeating and offsetting a list of integers based on cycle length and repetitions."""
        arrays: NDArray[np.int_] = RepetitionExperimentKernel.create_sliced_arrays(int_list, cycle_length, repetitions)
        return np.concatenate(arrays)
    # endregion
