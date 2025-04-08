# -------------------------------------------
# Module describing interface for error (detection/correction) identification
# -------------------------------------------
from abc import abstractmethod, ABCMeta
from functools import lru_cache
import numpy as np
from numpy.typing import NDArray
from xarray import DataArray
from typing import List
from qce_interp.utilities.custom_exceptions import InterfaceMethodException
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_state_classification import IStateClassifierContainer


class IDecoder(metaclass=ABCMeta):
    """
    Interface class, describing methods required for lookup-table decoding.
    Uses following convention:
    Output arrays are 3D tensors (N, M, P) where,
    - N is the number of measurement repetitions.
    - M is the number of stabilizer repetitions.
    - P is the number of qubit elements.
        Where:
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
    """

    # region Interface Methods
    @abstractmethod
    def get_fidelity(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """:return: Logical fidelity based on target state and stabilizer round-count."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_fidelity_uncertainty(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """:return: Uncertainty in Logical fidelity based on target state and stabilizer round-count."""
        raise InterfaceMethodException
    # endregion


class ISyndromeDecoder(IDecoder, metaclass=ABCMeta):
    """
    Interface class, describing methods required for defect-to-syndrome decoding.
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
    def get_binary_syndrome_corrections(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Array-like of binary syndrome corrections to be applied to the data qubit outcomes.
        Output shape: (N, M(+1), D)
        - N is the number of measurement repetitions.
        - M is the number of stabilizer repetitions.
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
        :return: Tensor of binary-syndromes at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_binary_projected_corrected(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Array-like of combined binary syndrome corrections to be applied to the data qubit outcomes.
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-corrected at specific cycle.
        """
        raise InterfaceMethodException
    # endregion

    # region Class Methods
    @lru_cache(maxsize=None)
    def get_binary_syndrome_correction(self, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        Output shape: (N, 1, D)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Tensor of binary-corrections at specific cycle.
        """
        # (N, M(+1), D)
        syndrome_corrections: NDArray[np.int_] = self.get_binary_syndrome_corrections(cycle_stabilizer_count=cycle_stabilizer_count)
        n, m, d = syndrome_corrections.shape
        # Pre-process
        eigenvalue_syndrome_corrections = IStateClassifierContainer.binary_to_eigenvalue(syndrome_corrections)
        # (N, 1, D)
        eigenvalue_syndrome_correction: np.ndarray = np.prod(eigenvalue_syndrome_corrections, axis=1)  # Along 'M + 1' axis
        eigenvalue_syndrome_correction = eigenvalue_syndrome_correction.reshape((n, 1, d))
        # Post-process
        syndrome_correction = IStateClassifierContainer.eigenvalue_to_binary(eigenvalue_syndrome_correction)
        return syndrome_correction
    # endregion


class ILabeledSyndromeDecoder(ISyndromeDecoder, metaclass=ABCMeta):
    """
    Interface class, describing methods required for defect-to-syndrome decoding.
    Extends ISyndromeDecoder by changing return type to xarray.DataArray for additional information.
    Defines methods for obtaining various classifications in the form of labeled xarray.DataArrays.
    These methods provide enhanced data interpretability with added contextual information.
    """

    # region Interface Methods
    @abstractmethod
    def get_labeled_binary_syndrome_corrections(self, cycle_stabilizer_count: int) -> DataArray:
        """
        :return: xarray.DataArray of binary-syndromes at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_binary_projected_corrected(self, cycle_stabilizer_count: int) -> DataArray:
        """
        :return: xarray.DataArray of binary-corrected at specific cycle.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_labeled_binary_syndrome_correction(self, cycle_stabilizer_count: int) -> DataArray:
        """
        :return: xarray.DataArray of binary-corrections at specific cycle.
        """
        raise InterfaceMethodException
    # endregion
