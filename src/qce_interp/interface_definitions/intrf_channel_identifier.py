# -------------------------------------------
# Interface for unique channel references
# For example:
# Qubit identifier, Feedline identifier, Flux channel identifier, etc.
# -------------------------------------------
from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass
from typing import List
from qce_interp.custom_exceptions import InterfaceMethodException


class IChannelIdentifier(ABC):
    """
    Interface class, describing unique identifier.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def id(self) -> str:
        """:returns: Reference Identifier."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def __hash__(self):
        """:returns: Identifiable hash."""
        raise InterfaceMethodException

    @abstractmethod
    def __eq__(self, other):
        """:returns: Boolean if other shares equal identifier, else InterfaceMethodException."""
        raise InterfaceMethodException
    # endregion


class IQubitID(IChannelIdentifier, metaclass=ABCMeta):
    """
    Interface for qubit reference.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def name(self) -> str:
        """:returns: Reference name for qubit."""
        raise InterfaceMethodException
    # endregion


class IFeedlineID(IChannelIdentifier, metaclass=ABCMeta):
    """
    Interface for feedline reference.
    """
    pass


class IEdgeID(IChannelIdentifier, metaclass=ABCMeta):
    """
    Interface class, for qubit-to-qubit edge reference.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def qubit_ids(self) -> List[IQubitID]:
        """:return: Connected Qubit-ID's."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def contains(self, element: IQubitID) -> bool:
        """:return: Boolean, whether element is part of edge or not."""
        raise InterfaceMethodException

    @abstractmethod
    def get_connected_qubit_id(self, element: IQubitID) -> IQubitID:
        """:return: Qubit-ID, connected to the other side of this edge."""
        raise InterfaceMethodException
    # endregion


@dataclass(frozen=True)
class QubitIDObj(IQubitID):
    """
    Contains qubit label ID.
    """
    _id: str

    # region Interface Properties
    @property
    def id(self) -> str:
        """:returns: Reference ID for qubit."""
        return self._id

    @property
    def name(self) -> str:
        """:returns: Reference name for qubit."""
        return self.id
    # endregion

    # region Class Methods
    def __hash__(self):
        """:returns: Identifiable hash."""
        return self.id.__hash__()

    def __eq__(self, other):
        """:returns: Boolean if other shares equal identifier, else InterfaceMethodException."""
        if isinstance(other, IQubitID):
            return self.id.__eq__(other.id)
        # raise NotImplementedError('QubitIDObj equality check to anything other than IQubitID interface is not implemented.')
        return False

    def __repr__(self):
        return f'<Qubit-ID>{self.id}'
    # endregion


@dataclass(frozen=True)
class FeedlineIDObj(IFeedlineID):
    """
    Data class, implementing IFeedlineID interface.
    """
    name: str

    # region Interface Properties
    @property
    def id(self) -> str:
        """:returns: Reference ID for feedline."""
        return self.name
    # endregion

    # region Class Methods
    def __hash__(self):
        return self.id.__hash__()

    def __eq__(self, other):
        if isinstance(other, IFeedlineID):
            return self.id.__eq__(other.id)
        # raise NotImplementedError('FeedlineIDObj equality check to anything other than IFeedlineID interface is not implemented.')
        return False

    def __repr__(self):
        return f'<Feedline-ID>{self.id}'
    # endregion


@dataclass(frozen=True)
class EdgeIDObj(IEdgeID):
    """
    Data class, implementing IEdgeID interface.
    """
    qubit_id0: IQubitID
    """Arbitrary edge qubit-ID."""
    qubit_id1: IQubitID
    """Arbitrary edge qubit-ID."""

    # region Interface Properties
    @property
    def id(self) -> str:
        """:returns: Reference ID for edge."""
        return f"{self.qubit_id0.id}-{self.qubit_id1.id}"

    @property
    def qubit_ids(self) -> List[IQubitID]:
        """:return: Connected Qubit-ID's."""
        return [self.qubit_id0, self.qubit_id1]
    # endregion

    # region Interface Methods
    def contains(self, element: IQubitID) -> bool:
        """:return: Boolean, whether element is part of edge or not."""
        if element in [self.qubit_id0, self.qubit_id1]:
            return True
        return False

    def get_connected_qubit_id(self, element: IQubitID) -> IQubitID:
        """:return: Qubit-ID, connected to the other side of this edge."""
        if element == self.qubit_id0:
            return self.qubit_id1
        if element == self.qubit_id1:
            return self.qubit_id0
        # If element is not part of this edge
        raise ValueError(f"Element: {element} is not part of this edge: {self}")
    # endregion

    # region Class Methods
    def __hash__(self):
        """
        Sorts individual qubit hashes such that the order is NOT maintained.
        Making hash comparison independent of order.
        """
        return hash((min(self.qubit_id0.__hash__(), self.qubit_id1.__hash__()), max(self.qubit_id0.__hash__(), self.qubit_id1.__hash__())))

    def __eq__(self, other):
        if isinstance(other, IEdgeID):
            # Edge is equal if they share the same qubit identifiers, order does not matter
            return other.contains(self.qubit_id0) and other.contains(self.qubit_id1)
        # raise NotImplementedError('EdgeIDObj equality check to anything other than IEdgeID interface is not implemented.')
        return False

    def __repr__(self):
        return f'<Edge-ID>{self.id}'
    # endregion
