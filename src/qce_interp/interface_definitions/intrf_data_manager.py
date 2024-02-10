# -------------------------------------------
# Module containing interface for data manager.
# -------------------------------------------
from abc import ABC, abstractmethod
from typing import List, Optional
from qce_interp.utilities.custom_exceptions import InterfaceMethodException
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_stabilizer_index_kernel import IIndexingKernel
from qce_interp.interface_definitions.intrf_state_classification import (
    IStateClassifierContainer,
    StateAcquisitionContainer,
)
from qce_interp.interface_definitions.intrf_error_identifier import (
    IErrorDetectionIdentifier,
    LabeledErrorDetectionIdentifier,
)


class IDataManager(ABC):
    """
    Interface class, describing data entrypoints management.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def involved_ancilla_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved ancilla qubit-ID's."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def rounds(self) -> List[int]:
        """:return: Array-like of number of QEC-rounds per experiment."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def index_kernel(self) -> IIndexingKernel:
        """:return: Index kernel used for indexing data."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_error_detection_classifier(self, **kwargs) -> IErrorDetectionIdentifier:
        """
        :param kwargs: Additional keyword arguments passed to class constructor.
        :return: Instance that exposes high-level get-methods which can be used to construct error decoders, Pij-matrix, etc.
        """
        raise InterfaceMethodException

    def get_labeled_error_detection_classifier(self, **kwargs) -> LabeledErrorDetectionIdentifier:
        """
        :param kwargs: Additional keyword arguments passed to class constructor.
        :return: Instance that exposes high-level get-methods + xarray formatting,
            which can be used to construct error decoders, Pij-matrix, etc.
        """
        return LabeledErrorDetectionIdentifier(
            error_detection_identifier=self.get_error_detection_classifier(**kwargs),
        )

    @abstractmethod
    def get_state_classifier(self, qubit_id: IQubitID) -> Optional[IStateClassifierContainer]:
        """:return: State classifier based on qubit-ID. Returns None if qubit-ID is not supported."""
        raise InterfaceMethodException
    # endregion
