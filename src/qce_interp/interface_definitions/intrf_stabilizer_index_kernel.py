# -------------------------------------------
# Module describing interface for stabilizer (error detection/correction) indexing.
# -------------------------------------------
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import List
from qce_interp.custom_exceptions import InterfaceMethodException
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_state_classification import StateKey


class IIndexingKernel(ABC):
    """
    Interface class, describing qubit acquisition indexing.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def start_index(self) -> int:
        """:return: Starting index."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def stop_index(self) -> int:
        """:return: End index."""
        raise InterfaceMethodException

    # endregion

    # region Interface Methods
    @abstractmethod
    def contains(self, element: IQubitID) -> List[int]:
        """:return: Array-like of measurement indices corresponding to element within this indexing kernel."""
        raise InterfaceMethodException
    # endregion


class IStabilizerIndexingKernel(IIndexingKernel, metaclass=ABCMeta):
    """
    Interface class, describing qubit acquisition indexing.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def kernel_cycle_length(self) -> int:
        """:return: Integer length of indexing kernel cycle."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def experiment_repetitions(self) -> int:
        """Number of repetitions for this experiment."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_heralded_cycle_acquisition_indices(self, qubit_id: IQubitID, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param cycle_stabilizer_count: Identifies the indices to only include cycles with this number of stabilizers.
        :return: Tensor of indices pointing at all heralded acquisition before stabilizer cycles.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_stabilizer_acquisition_indices(self, qubit_id: IQubitID, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param cycle_stabilizer_count: Identifies the indices to only include cycles with this number of stabilizers.
        :return: Tensor of indices pointing at all stabilizer acquisition within stabilizer cycles.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_projected_cycle_acquisition_indices(self, qubit_id: IQubitID, cycle_stabilizer_count: int) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param cycle_stabilizer_count: Identifies the indices to only include cycles with this number of stabilizers.
        :return: Tensor of indices pointing at all projection acquisition after stabilizer cycles.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_heralded_calibration_acquisition_indices(self, qubit_id: IQubitID, state: StateKey) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param state: Identifier for state specific selectivity.
        :return: Tensor of indices pointing at all heralded acquisition before calibration points.
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_projected_calibration_acquisition_indices(self, qubit_id: IQubitID, state: StateKey) -> NDArray[np.int_]:
        """
        :param qubit_id: Identifier to which these acquisition indices correspond to.
        :param state: Identifier for state specific selectivity.
        :return: Tensor of indices pointing at all projection acquisition within calibration points.
        """
        raise InterfaceMethodException
    # endregion


class IIndexStrategy(ABC):
    """
    Interface class, describing strategy for (measurement) index offset.
    """

    # region Interface Properties
    @abstractmethod
    def get_index(self, task: IIndexingKernel) -> int:
        """:return: Index [a.u.]."""
        raise InterfaceMethodException
    # endregion


@dataclass(frozen=True)
class FixedIndexStrategy(IIndexStrategy):
    """
    Data class, implementing IIndexStrategy interface.
    Forces fixed indexing.
    """
    index: int = field(default=0)

    # region Interface Properties
    def get_index(self, task: IIndexingKernel) -> int:
        """:return: Index [a.u.]."""
        return self.index
    # endregion


@dataclass(frozen=True)
class RelativeIndexStrategy(IIndexStrategy):
    """
    Data class, implementing IIndexStrategy interface.
    Forces relative indexing.
    """
    reference_index_kernel: IIndexingKernel

    # region Interface Properties
    def get_index(self, task: IIndexingKernel) -> int:
        """:return: Index [a.u.]."""
        return self.reference_index_kernel.stop_index + 1
    # endregion
