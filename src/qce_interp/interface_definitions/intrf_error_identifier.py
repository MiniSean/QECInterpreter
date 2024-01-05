# -------------------------------------------
# Module describing interface for error (detection/correction) identification
# -------------------------------------------
from abc import ABC, abstractmethod, ABCMeta
from functools import lru_cache
from collections import OrderedDict
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict
from xarray import DataArray
from enum import Enum, unique
from qce_interp.custom_exceptions import InterfaceMethodException
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_interp.interface_definitions.intrf_stabilizer_index_kernel import IStabilizerIndexingKernel
from qce_interp.interface_definitions.intrf_state_classification import IStateClassifierContainer


class IErrorDetectionIdentifier(ABC):
    """
    Interface class, describing methods for commonly used higher level abstractions of stabilizer codes.
    Uses following convention:
    Output arrays are 3D tensors (N, M, P) where,
    - N is the number of measurement repetitions.
    - M is the number of stabilizer repetitions.
    - P is the number of qubit elements.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def include_heralded_post_selection(self) -> bool:
        raise InterfaceMethodException

    @property
    @abstractmethod
    def include_leakage_post_selection(self) -> bool:
        raise InterfaceMethodException

    @property
    @abstractmethod
    def include_computation_parity(self) -> bool:
        raise InterfaceMethodException

    @property
    @abstractmethod
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's"""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def involved_stabilizer_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved stabilizer (ancilla) qubit-ID's"""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's"""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_binary_heralded_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, P)
        - N is the number of measurement repetitions.
        - P is the number of qubit elements.
        :return: Tensor of binary-classification at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_parity_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M, S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of parity-classification at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_parity_computation_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, S)
        - N is the number of measurement repetitions.
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
        :return: Tensor of parity-classification at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_defect_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M(+1), S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of defect-classification at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_defect_stabilizer_lookup(self, cycle_stabilizer_count: int) -> Dict[IQubitID, NDArray[np.int_]]:
        """
        Output shape: S: (N, M(+1))
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Dictionary with stabilizer qubits as keys and tensor of defect-classification at specific cycle as values.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_binary_projected_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-classification at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_ternary_projected_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of ternary-classification of projected data qubits at specific cycle.
        """
        raise InterfaceMethodException
    # endregion


class ILabeledErrorDetectionIdentifier(IErrorDetectionIdentifier, metaclass=ABCMeta):
    """
    Interface class, describing methods for commonly used higher level abstractions of stabilizer codes.
    Extends IErrorDetectionIdentifier by changing return type to xarray.DataArray for additional information.
    Defines methods for obtaining various classifications in the form of labeled xarray.DataArrays.
    These methods provide enhanced data interpretability with added contextual information.
    """

    # region Interface Methods
    @abstractmethod
    def get_labeled_binary_heralded_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves binary classification data for heralded qubits in a labeled, structured format.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'binary_value', and 'qubit_id'.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_parity_stabilizer_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves parity classification data for stabilizer qubits as a labeled data array.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'stabilizer_repetition', and 'qubit_id'.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_parity_computation_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves parity computation classification data for qubits in a structured format.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'value', and 'qubit_id'.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_defect_stabilizer_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves defect classification data for stabilizer qubits as a labeled data array.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'stabilizer_repetition', and 'qubit_id'.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_defect_stabilizer_lookup(self, cycle_stabilizer_count: int) -> Dict[IQubitID, DataArray]:
        """
        Retrieves defect classifications for stabilizer qubits, indexed by qubit ID.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: Dictionary mapping qubit IDs to their labeled defect classification data as xarray.DataArray.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_binary_projected_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves binary classification data for projected data qubits in a labeled format.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'value', and 'data_qubit_id'.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_ternary_projected_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves ternary classification data for projected data qubits as a structured array.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'value', and 'data_qubit_id'.
        """
        raise InterfaceMethodException
    # endregion


class ErrorDetectionIdentifier(IErrorDetectionIdentifier):
    """
    Behaviour class, implementing IErrorDetectionIdentifier interfaces.
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
    def include_heralded_post_selection(self) -> bool:
        return self._use_post_selection

    @property
    def include_leakage_post_selection(self) -> bool:
        return self.include_projected_leakage_post_selection or self.include_stabilizer_leakage_post_selection

    @property
    def include_computation_parity(self) -> bool:
        return self._use_computational_parity

    @property
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's"""
        return self._involved_qubit_ids

    @property
    def involved_stabilizer_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved stabilizer (ancilla) qubit-ID's"""
        return [qubit_id for qubit_id in self._involved_qubit_ids if qubit_id in self._device_layout.ancilla_qubit_ids]

    @property
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's"""
        return [qubit_id for qubit_id in self._involved_qubit_ids if qubit_id in self._device_layout.data_qubit_ids]
    # endregion

    # region Class Properties
    @property
    def include_projected_leakage_post_selection(self) -> bool:
        return self._use_projected_leakage_post_selection

    @property
    def include_stabilizer_leakage_post_selection(self) -> bool:
        return self._use_stabilizer_leakage_post_selection
    # endregion

    # region Class Constructor
    def __init__(
            self,
            classifier_lookup: Dict[IQubitID, IStateClassifierContainer],
            index_kernel: IStabilizerIndexingKernel, involved_qubit_ids: List[IQubitID],
            device_layout: ISurfaceCodeLayer,
            use_heralded_post_selection: bool = False,
            use_projected_leakage_post_selection: bool = False,
            use_stabilizer_leakage_post_selection: bool = False,
            use_computational_parity: bool = False,
    ):
        self._classifier_lookup: Dict[IQubitID, IStateClassifierContainer] = classifier_lookup
        self._index_kernel: IStabilizerIndexingKernel = index_kernel
        self._involved_qubit_ids: List[IQubitID] = involved_qubit_ids
        self._device_layout: ISurfaceCodeLayer = device_layout
        self._use_post_selection: bool = use_heralded_post_selection
        self._use_computational_parity: bool = use_computational_parity
        self._use_projected_leakage_post_selection: bool = use_projected_leakage_post_selection
        self._use_stabilizer_leakage_post_selection: bool = use_stabilizer_leakage_post_selection
    # endregion

    # region Interface Methods
    @lru_cache(maxsize=None)
    def get_binary_heralded_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, P)
        - N is the number of measurement repetitions.
        - P is the number of qubit elements.
        :return: Tensor of binary-classification at specific cycle.
        """
        # Data allocation
        any_qubit_id: IQubitID = self.involved_qubit_ids[0]
        heralded_acquisition_indices: NDArray[np.int_] = self._index_kernel.get_heralded_cycle_acquisition_indices(qubit_id=any_qubit_id, cycle_stabilizer_count=cycle_stabilizer_count)
        post_selection_mask = self.get_post_selection_mask(cycle_stabilizer_count=cycle_stabilizer_count)
        heralded_acquisition_indices = heralded_acquisition_indices[post_selection_mask]

        # Prepare output shape
        n, one = heralded_acquisition_indices.shape
        p: int = len(self.involved_qubit_ids)
        result: NDArray[np.int_] = np.zeros(shape=(p, n, one), dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            state_classifier: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=heralded_acquisition_indices,
            )
            result[i, :, :] = state_classifier.get_binary_classification()
        # (N, 1, P) Transpose
        result = result.transpose((1, 2, 0))
        return result

    @lru_cache(maxsize=None)
    def get_binary_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M, S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of binary-classification at specific cycle.
        """
        # Data allocation
        any_qubit_id: IQubitID = self.involved_stabilizer_qubit_ids[0]
        stabilizer_acquisition_indices: NDArray[np.int_] = self._index_kernel.get_stabilizer_acquisition_indices(qubit_id=any_qubit_id, cycle_stabilizer_count=cycle_stabilizer_count)
        post_selection_mask = self.get_post_selection_mask(cycle_stabilizer_count=cycle_stabilizer_count)
        stabilizer_acquisition_indices = stabilizer_acquisition_indices[post_selection_mask]

        # Prepare output shape
        n, m = stabilizer_acquisition_indices.shape
        s: int = len(self.involved_stabilizer_qubit_ids)
        result: NDArray[np.int_] = np.zeros(shape=(s, n, m), dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_stabilizer_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            state_classifier: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=stabilizer_acquisition_indices,
            )
            result[i, :, :] = state_classifier.get_binary_classification()
        # (N, M, S) Transpose
        result = result.transpose((1, 2, 0))
        return result

    @lru_cache(maxsize=None)
    def get_parity_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M, S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of parity-classification at specific cycle.
        """
        # Data allocation
        any_qubit_id: IQubitID = self.involved_stabilizer_qubit_ids[0]
        stabilizer_acquisition_indices: NDArray[np.int_] = self._index_kernel.get_stabilizer_acquisition_indices(qubit_id=any_qubit_id, cycle_stabilizer_count=cycle_stabilizer_count)
        post_selection_mask = self.get_post_selection_mask(cycle_stabilizer_count=cycle_stabilizer_count)
        stabilizer_acquisition_indices = stabilizer_acquisition_indices[post_selection_mask]

        # Prepare output shape
        n, m = stabilizer_acquisition_indices.shape
        s: int = len(self.involved_stabilizer_qubit_ids)
        result: NDArray[np.int_] = np.zeros(shape=(s, n, m), dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_stabilizer_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            state_classifier: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=stabilizer_acquisition_indices,
            )
            result[i, :, :] = state_classifier.get_parity_classification()
        # (N, M, S) Transpose
        result = result.transpose((1, 2, 0))
        return result

    @lru_cache(maxsize=None)
    def get_parity_computation_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, S)
        - N is the number of measurement repetitions.
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
        Note: returns empty tensor if self.include_computation_parity is False.
        :return: Tensor of parity-classification at specific cycle.
        """
        # (N, 1, D) Binary projected acquisition
        binary_projection: NDArray[np.int_] = self.get_binary_projected_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        # (N, D) Pre-process
        n, one, d = binary_projection.shape

        # Guard clause, if computational parity should not be computed, return empty array
        if not self.include_computation_parity:
            # What to return here
            return np.asarray([])

        binary_projection = binary_projection.reshape(n, d)
        computational_parity: NDArray[np.int_] = self.calculate_computational_parity(binary_projection)
        # (N, 1, D-1) Post-process
        computational_parity = computational_parity.reshape((n, one, d - 1))
        return computational_parity

    @lru_cache(maxsize=None)
    def get_defect_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M(+1), S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of defect-classification at specific cycle.
        """
        # (N, M, S)
        parity_classification: NDArray[np.int_] = self.get_parity_stabilizer_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        # Optionally include computational parity
        if self.include_computation_parity:
            parity_classification: NDArray[np.int_] = np.concatenate(
                (
                    parity_classification,
                    self.get_parity_computation_classification(cycle_stabilizer_count=cycle_stabilizer_count),
                ),
                axis=1,
            )
        # (N, M(+1), S) Iterate over involved stabilizers to convert from parity to defect
        result: NDArray[np.int_] = np.zeros(shape=parity_classification.shape, dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_stabilizer_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            # (N, M(+1), 1)
            sub_parity_classification: NDArray[np.int_] = parity_classification[:, :, i]
            sub_defect_classification: NDArray[np.int_] = IStateClassifierContainer.calculate_defect(
                m=sub_parity_classification,
                initial_condition=state_classifier.expected_parity.value,
            )
            result[:, :, i] = sub_defect_classification
        # (N, M(+1), S)
        return result

    def get_defect_stabilizer_lookup(self, cycle_stabilizer_count: int) -> Dict[IQubitID, NDArray[np.int_]]:
        """
        Output shape: S: (N, M(+1))
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Dictionary with stabilizer qubits as keys and tensor of (binary)defect-classification at specific cycle as values.
        """
        # Data allocation
        result: OrderedDict[IQubitID, NDArray[np.int_]] = OrderedDict()

        for qubit_id in self.involved_stabilizer_qubit_ids:
            # (Post selected) acquisition indices
            stabilizer_acquisition_indices: NDArray[np.int_] = self._index_kernel.get_stabilizer_acquisition_indices(qubit_id=qubit_id, cycle_stabilizer_count=cycle_stabilizer_count)
            post_selection_mask = self.get_post_selection_mask(cycle_stabilizer_count=cycle_stabilizer_count)
            stabilizer_acquisition_indices = stabilizer_acquisition_indices[post_selection_mask]

            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            state_classifier: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=stabilizer_acquisition_indices,
            )
            result[qubit_id] = IStateClassifierContainer.eigenvalue_to_binary(state_classifier.get_defect_classification())
        return result

    @lru_cache(maxsize=None)
    def get_binary_projected_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-classification at specific cycle.
        """
        # Data allocation
        any_qubit_id: IQubitID = self.involved_data_qubit_ids[0]
        projected_acquisition_indices: NDArray[np.int_] = self._index_kernel.get_projected_cycle_acquisition_indices(qubit_id=any_qubit_id, cycle_stabilizer_count=cycle_stabilizer_count)
        post_selection_mask = self.get_post_selection_mask(cycle_stabilizer_count=cycle_stabilizer_count)
        projected_acquisition_indices = projected_acquisition_indices[post_selection_mask]

        # Prepare output shape
        n, one = projected_acquisition_indices.shape
        d: int = len(self.involved_data_qubit_ids)
        result: NDArray[np.int_] = np.zeros(shape=(d, n, one), dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_data_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            state_classifier: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=projected_acquisition_indices,
            )
            result[i, :, :] = state_classifier.get_binary_classification()
        # (N, 1, D) Transpose
        result = result.transpose((1, 2, 0))
        return result

    @lru_cache(maxsize=None)
    def get_ternary_projected_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of ternary-classification of projected data qubits at specific cycle.
        """
        # Data allocation
        any_qubit_id: IQubitID = self.involved_data_qubit_ids[0]
        projected_acquisition_indices: NDArray[np.int_] = self._index_kernel.get_projected_cycle_acquisition_indices(qubit_id=any_qubit_id, cycle_stabilizer_count=cycle_stabilizer_count)
        post_selection_mask = self.get_post_selection_mask(cycle_stabilizer_count=cycle_stabilizer_count)
        projected_acquisition_indices = projected_acquisition_indices[post_selection_mask]

        # Prepare output shape
        n, one = projected_acquisition_indices.shape
        d: int = len(self.involved_data_qubit_ids)
        result: NDArray[np.int_] = np.zeros(shape=(d, n, one), dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_data_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            state_classifier: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=projected_acquisition_indices,
            )
            result[i, :, :] = state_classifier.get_ternary_classification()
        # (N, 1, D) Transpose
        result = result.transpose((1, 2, 0))
        return result
    # endregion

    # region Class Methods
    def get_heralded_post_selection_mask(self, cycle_stabilizer_count: int) -> NDArray[np.bool_]:
        """
        Output shape: (N,)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - P is the number of qubit elements.
        :return: Tensor of boolean mask at specific cycle.
        """
        # (P, N, M) Heralded acquisition index slices
        index_slices: NDArray[np.int_] = np.asarray([
            self._index_kernel.get_heralded_cycle_acquisition_indices(qubit_id=qubit_id,
                                                                      cycle_stabilizer_count=cycle_stabilizer_count)
            for qubit_id in self.involved_qubit_ids
        ])
        # (P, N, M) Binary classification of heralded acquisition
        heralded_binary_tensor: np.ndarray = np.zeros(index_slices.shape, dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            index_sub_slices: NDArray[np.int_] = index_slices[i]
            reshaped_container: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=index_sub_slices,
            )
            heralded_binary_tensor[i] = reshaped_container.get_binary_classification()
        # (N, P * M) Reshape
        p, n, m = heralded_binary_tensor.shape
        heralded_binary_tensor = heralded_binary_tensor.transpose((1, 0, 2))
        heralded_binary_tensor = heralded_binary_tensor.reshape((n, p * m))
        # (N, 1) Condition on product of binary states equals to 0 (ground state heralded)
        result: NDArray[np.bool_] = np.all(heralded_binary_tensor == 0, axis=1)
        return result

    def get_projected_leakage_post_selection_mask(self, cycle_stabilizer_count: int) -> NDArray[np.bool_]:
        """
        Output shape: (N,)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - D is the number of data qubits.
        :return: Tensor of boolean mask for leakage during projected acquisition at specific cycle.
        """
        # (D, N, 1) Projected acquisition index slices
        index_slices: NDArray[np.int_] = np.asarray([
            self._index_kernel.get_projected_cycle_acquisition_indices(qubit_id=qubit_id,
                                                                       cycle_stabilizer_count=cycle_stabilizer_count)
            for qubit_id in self.involved_data_qubit_ids
        ])
        # (D, N, 1) Ternary classification of projected acquisition
        projected_ternary_tensor: np.ndarray = np.zeros(index_slices.shape, dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_data_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            index_sub_slices: NDArray[np.int_] = index_slices[i]
            reshaped_container: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=index_sub_slices,
            )
            projected_ternary_tensor[i] = reshaped_container.get_ternary_classification()
        # (N, D * 1) Reshape
        d, n, one = projected_ternary_tensor.shape
        projected_ternary_tensor = projected_ternary_tensor.transpose((1, 0, 2))
        projected_ternary_tensor = projected_ternary_tensor.reshape((n, d * one))
        # (N, 1) Condition on product of ternary states lower than 2 (leakage state)
        result: NDArray[np.bool_] = np.all(projected_ternary_tensor < 2, axis=1)
        return result

    def get_stabilizer_leakage_post_selection_mask(self, cycle_stabilizer_count: int) -> NDArray[np.bool_]:
        """
        Output shape: (N,)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of data qubits.
        :return: Tensor of boolean mask for leakage during projected acquisition at specific cycle.
        """
        # (S, N, M) Projected acquisition index slices
        index_slices: NDArray[np.int_] = np.asarray([
            self._index_kernel.get_stabilizer_acquisition_indices(qubit_id=qubit_id,
                                                                  cycle_stabilizer_count=cycle_stabilizer_count)
            for qubit_id in self.involved_stabilizer_qubit_ids
        ])
        # (S, N, M) Ternary classification of projected acquisition
        stabilizer_ternary_tensor: np.ndarray = np.zeros(index_slices.shape, dtype=np.int_)
        for i, qubit_id in enumerate(self.involved_stabilizer_qubit_ids):
            state_classifier: IStateClassifierContainer = self._classifier_lookup[qubit_id]
            index_sub_slices: NDArray[np.int_] = index_slices[i]
            reshaped_container: IStateClassifierContainer = state_classifier.reshape(
                container=state_classifier,
                index_slices=index_sub_slices,
            )
            stabilizer_ternary_tensor[i] = reshaped_container.get_ternary_classification()
        # (N, S * M) Reshape
        s, n, m = stabilizer_ternary_tensor.shape
        stabilizer_ternary_tensor = stabilizer_ternary_tensor.transpose((1, 0, 2))
        stabilizer_ternary_tensor = stabilizer_ternary_tensor.reshape((n, s * m))
        # (N, 1) Condition on product of ternary states lower than 2 (leakage state)
        result: NDArray[np.bool_] = np.all(stabilizer_ternary_tensor < 2, axis=1)
        return result

    @lru_cache(maxsize=None)
    def get_post_selection_mask(self, cycle_stabilizer_count: int) -> NDArray[np.bool_]:
        """
        Output shape: (N,)
        - N is the number of measurement repetitions.
        :return: Tensor of boolean mask based on post-selection conditions (at specific cycle).
        """
        full_pass_mask: NDArray[np.bool_] = np.full(shape=(self._index_kernel.experiment_repetitions,), fill_value=True, dtype=np.bool_)
        result: NDArray[np.bool_] = full_pass_mask
        # (Optionally) add heralded post-selection
        if self.include_heralded_post_selection:
            heralded_selection_mask = self.get_heralded_post_selection_mask(
                cycle_stabilizer_count=cycle_stabilizer_count)
            result = np.logical_and(result, heralded_selection_mask)
        # (Optionally) add leakage (during data-qubit projection) post-selection
        if self.include_projected_leakage_post_selection:
            projected_leakage_selection_mask = self.get_projected_leakage_post_selection_mask(
                cycle_stabilizer_count=cycle_stabilizer_count)
            result = np.logical_and(result, projected_leakage_selection_mask)
        # (Optionally) add leakage (during stabilizer-qubit cycle) post-selection
        if self.include_stabilizer_leakage_post_selection:
            stabilizer_leakage_selection_mask = self.get_stabilizer_leakage_post_selection_mask(
                cycle_stabilizer_count=cycle_stabilizer_count)
            result = np.logical_and(result, stabilizer_leakage_selection_mask)
        return result
    # endregion

    # region Static Class Methods
    # TODO: This should depend on surface-code device layout
    @staticmethod
    def calculate_computational_parity(array: np.ndarray) -> np.ndarray:
        """
        Maps binary to parity.
        Mapping (0, 0) and (1, 1) -> +1.
        Mapping (0, 1) and (1, 0) -> -1.
        Input shape: (N, X)
        Output shape: (N, X-1)
        - N is the number of measurement repetitions.
        - X arbitrary number.
        :return: Tensor of parity-classification at specific cycle.
        """
        # Shift the array to the left and compare with the original array
        shifted_array = np.roll(array, -1, axis=-1)

        # The last column of shifted_array is no longer valid after rolling
        # It should not be used in comparison
        comparison = array[:, :-1] == shifted_array[:, :-1]

        # Map True to +1 and False to -1
        parity_array = np.where(comparison, 1, -1)
        return parity_array
    # endregion


@unique
class DataArrayLabels(Enum):
    MEASUREMENT = 'measurement_repetition'
    BINARY_VALUE = 'binary_value'
    QUBIT_ID = 'qubit_id'
    STABILIZER_REPETITION = 'stabilizer_repetition'
    VALUE = 'value'
    DATA_QUBIT_ID = 'data_qubit_id'


class LabeledErrorDetectionIdentifier(ILabeledErrorDetectionIdentifier):
    """
    Behaviour class, implementing ILabeledErrorDetectionIdentifier interfaces.
    Extends ErrorDetectionIdentifier to format its outputs as labeled xarray.DataArray objects.
    Each method mirrors those in ErrorDetectionIdentifier but returns data with additional
    contextual information (coordinates and dimensions), enhancing data interpretability.
    """

    # region Interface Properties
    @property
    def include_heralded_post_selection(self) -> bool:
        return self._error_detection_identifier.include_heralded_post_selection

    @property
    def include_leakage_post_selection(self) -> bool:
        return self._error_detection_identifier.include_leakage_post_selection

    @property
    def include_computation_parity(self) -> bool:
        return self._error_detection_identifier.include_computation_parity

    @property
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's"""
        return self._error_detection_identifier.involved_qubit_ids

    @property
    def involved_stabilizer_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved stabilizer (ancilla) qubit-ID's"""
        return self._error_detection_identifier.involved_stabilizer_qubit_ids

    @property
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's"""
        return self._error_detection_identifier.involved_data_qubit_ids
    # endregion

    # region Class Constructor
    def __init__(self, error_detection_identifier: IErrorDetectionIdentifier):
        self._error_detection_identifier: IErrorDetectionIdentifier = error_detection_identifier
    # endregion

    # region Interface Methods
    def get_binary_heralded_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, P)
        - N is the number of measurement repetitions.
        - P is the number of qubit elements.
        :return: Tensor of binary-classification at specific cycle.
        """
        return self._error_detection_identifier.get_binary_heralded_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_binary_heralded_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves binary classification data for heralded qubits in a labeled, structured format.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'binary_value', and 'qubit_id'.
        """
        # (N, 1, P) Transpose
        result: NDArray[np.int_] = self.get_binary_heralded_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        n, one, p = result.shape

        # Convert to xarray.DataArray with meaningful dimensions and coordinates
        measurements = range(1, n + 1)
        binary_values = range(one)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_qubit_ids]

        data_array = DataArray(
            result,
            coords={
                DataArrayLabels.MEASUREMENT.value: measurements,
                DataArrayLabels.BINARY_VALUE.value: binary_values,
                DataArrayLabels.QUBIT_ID.value: qubit_ids,
            },
            dims=[
                DataArrayLabels.MEASUREMENT.value,
                DataArrayLabels.BINARY_VALUE.value,
                DataArrayLabels.QUBIT_ID.value,
            ],
        )

        return data_array

    def get_parity_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M, S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of parity-classification at specific cycle.
        """
        return self._error_detection_identifier.get_parity_stabilizer_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_parity_stabilizer_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves parity classification data for stabilizer qubits as a labeled data array.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'stabilizer_repetition', and 'qubit_id'.
        """
        # (N, M, S) Transpose
        result: NDArray[np.int_] = self.get_parity_stabilizer_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        n, m, s = result.shape

        measurements = range(1, n + 1)
        stabilizer_repetitions = range(1, m + 1)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_stabilizer_qubit_ids]

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

    def get_parity_computation_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, S)
        - N is the number of measurement repetitions.
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
        :return: Tensor of parity-classification at specific cycle.
        """
        return self._error_detection_identifier.get_parity_computation_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_parity_computation_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves parity computation classification data for qubits in a structured format.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'value', and 'qubit_id'.
        """
        # (N, 1, S) Post-process
        result: NDArray[np.int_] = self.get_parity_computation_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        n, one, s = result.shape

        measurements = range(1, n + 1)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_stabilizer_qubit_ids]

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

    def get_defect_stabilizer_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, M(+1), S)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Tensor of defect-classification at specific cycle.
        """
        return self._error_detection_identifier.get_defect_stabilizer_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_defect_stabilizer_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves defect classification data for stabilizer qubits as a labeled data array.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'stabilizer_repetition', and 'qubit_id'.
        """
        # (N, M(+1), S)
        result: NDArray[np.int_] = self.get_defect_stabilizer_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        n, m_plus_one, s = result.shape

        measurements = range(1, n + 1)
        stabilizer_repetitions = range(1, m_plus_one + 1)
        qubit_ids = [qubit_id.id for qubit_id in self.involved_stabilizer_qubit_ids]

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

    def get_defect_stabilizer_lookup(self, cycle_stabilizer_count: int) -> Dict[IQubitID, NDArray[np.int_]]:
        """
        Output shape: S: (N, M(+1))
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        :return: Dictionary with stabilizer qubits as keys and tensor of defect-classification at specific cycle as values.
        """
        return self._error_detection_identifier.get_defect_stabilizer_lookup(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_defect_stabilizer_lookup(self, cycle_stabilizer_count: int) -> Dict[IQubitID, DataArray]:
        """
        Retrieves defect classifications for stabilizer qubits, indexed by qubit ID.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: Dictionary mapping qubit IDs to their labeled defect classification data as xarray.DataArray.
        """
        result_dict: Dict[IQubitID, NDArray[np.int_]] = self.get_defect_stabilizer_lookup(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        data_array_dict: Dict[IQubitID, DataArray] = {}

        for qubit_id, result in result_dict.items():
            n, m_plus_one = result.shape
            measurements = range(1, n + 1)
            stabilizer_repetitions = range(1, m_plus_one + 1)

            data_array = DataArray(
                result,
                coords={
                    DataArrayLabels.MEASUREMENT.value: measurements,
                    DataArrayLabels.STABILIZER_REPETITION.value: stabilizer_repetitions,
                },
                dims=[
                    DataArrayLabels.MEASUREMENT.value,
                    DataArrayLabels.STABILIZER_REPETITION.value,
                ],
            )

            data_array_dict[qubit_id] = data_array

        return data_array_dict

    def get_binary_projected_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-classification at specific cycle.
        """
        return self._error_detection_identifier.get_binary_projected_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_binary_projected_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves binary classification data for projected data qubits in a labeled format.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'value', and 'data_qubit_id'.
        """
        # (N, 1, D) Transpose
        result: NDArray[np.int_] = self.get_binary_projected_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        n, one, d = result.shape

        measurements = range(1, n + 1)
        data_qubit_ids = [qubit_id.id for qubit_id in self.involved_data_qubit_ids]

        data_array = DataArray(
            result,
            coords={
                DataArrayLabels.MEASUREMENT.value: measurements,
                DataArrayLabels.VALUE.value: range(one),
                DataArrayLabels.DATA_QUBIT_ID.value: data_qubit_ids,
            },
            dims=[
                DataArrayLabels.MEASUREMENT.value,
                DataArrayLabels.VALUE.value,
                DataArrayLabels.DATA_QUBIT_ID.value,
            ],
        )

        return data_array

    def get_ternary_projected_classification(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of ternary-classification of projected data qubits at specific cycle.
        """
        return self._error_detection_identifier.get_ternary_projected_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )

    def get_labeled_ternary_projected_classification(self, cycle_stabilizer_count: int) -> DataArray:
        """
        Retrieves ternary classification data for projected data qubits as a structured array.

        :param cycle_stabilizer_count: int, the count of stabilizer cycles
        :return: xarray.DataArray with dimensions 'measurement', 'value', and 'data_qubit_id'.
        """
        # (N, 1, D) Transpose
        result: NDArray[np.int_] = self.get_ternary_projected_classification(
            cycle_stabilizer_count=cycle_stabilizer_count
        )
        n, one, d = result.shape

        measurements = range(1, n + 1)
        data_qubit_ids = [qubit_id.id for qubit_id in self.involved_data_qubit_ids]

        data_array = DataArray(
            result,
            coords={
                DataArrayLabels.MEASUREMENT.value: measurements,
                DataArrayLabels.VALUE.value: range(one),
                DataArrayLabels.DATA_QUBIT_ID.value: data_qubit_ids,
            },
            dims=[
                DataArrayLabels.MEASUREMENT.value,
                DataArrayLabels.VALUE.value,
                DataArrayLabels.DATA_QUBIT_ID.value,
            ],
        )

        return data_array
    # endregion

