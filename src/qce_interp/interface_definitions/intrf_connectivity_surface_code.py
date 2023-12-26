# -------------------------------------------
# Module containing interface for surface-code connectivity structure.
# -------------------------------------------
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from multipledispatch import dispatch
from typing import List, Tuple, Union
from enum import Enum, unique, auto
from qce_interp.custom_exceptions import InterfaceMethodException
from qce_interp.interface_definitions.intrf_channel_identifier import (
    IFeedlineID,
    IQubitID,
    IEdgeID,
)


class IIdentifier(ABC):
    """
    Interface class, describing equality identifier method.
    """

    # region Interface Methods
    @abstractmethod
    def __eq__(self, other):
        """:return: Boolean, whether 'other' equals 'self'."""
        raise InterfaceMethodException
    # endregion


class INode(IIdentifier, metaclass=ABCMeta):
    """
    Interface class, describing the node in a connectivity layer.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def edges(self) -> List['IEdge']:
        """:return: (N) Edges connected to this node."""
        raise InterfaceMethodException
    # endregion


class IEdge(IIdentifier, metaclass=ABCMeta):
    """
    Interface class, describing a connection between two nodes.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def nodes(self) -> Tuple[INode, INode]:
        """:return: (2) Nodes connected by this edge."""
        raise InterfaceMethodException
    # endregion


class IConnectivityLayer(ABC):
    """
    Interface class, describing a connectivity (graph) layer containing nodes and edges.
    Note that a connectivity layer can include 'separated' graphs
    where not all nodes have a connection path to all other nodes.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def nodes(self) -> List[INode]:
        """:return: Array-like of nodes."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def edges(self) -> List[IEdge]:
        """:return: Array-like of edges."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @dispatch(node=INode)
    @abstractmethod
    def get_connected_nodes(self, node: INode, order: int) -> List[INode]:
        """
        :param node: (Root) node to base connectivity on.
            If node has no edges, return an empty list.
        :param order: Connectivity range.
            Order <=0: empty list, 1: first order connectivity, 2: second order connectivity, etc.
        :return: Array-like of nodes connected to 'node' within order of connection (excluding 'node' itself).
        """
        raise InterfaceMethodException

    @dispatch(edge=IEdge)
    @abstractmethod
    def get_connected_nodes(self, edge: IEdge, order: int) -> List[INode]:
        """
        :param edge: (Root) edge to base connectivity on.
        :param order: Connectivity range.
            Order <=0: empty list, 1: first order connectivity, 2: second order connectivity, etc.
        :return: Array-like of nodes connected to 'edge' within order of connection.
        """
        raise InterfaceMethodException
    # endregion


class IConnectivityStack(ABC):
    """
    Interface class, describing an array-like of connectivity layers.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def layers(self) -> List[IConnectivityLayer]:
        """:return: Array-like of connectivity layers."""
        raise InterfaceMethodException
    # endregion


class IDeviceLayer(ABC):
    """
    Interface class, describing relation based connectivity.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def feedline_ids(self) -> List[IFeedlineID]:
        """:return: All feedline-ID's in device layer."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def qubit_ids(self) -> List[IQubitID]:
        """:return: (All) qubit-ID's in device layer."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def edge_ids(self) -> List[IEdgeID]:
        """:return: (All) edge-ID's in device layer."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_connected_qubits(self, feedline: IFeedlineID) -> List[IQubitID]:
        """:return: Qubit-ID's connected to feedline-ID."""
        raise InterfaceMethodException

    @abstractmethod
    def get_connected_feedline(self, qubit: IQubitID) -> IFeedlineID:
        """:return: Feedline-ID's connected to qubit-ID."""
        raise InterfaceMethodException

    @abstractmethod
    def get_neighbors(self, qubit: IQubitID, order: int = 1) -> List[IQubitID]:
        """
        Requires :param order: to be higher or equal to 1.
        :return: qubit neighbors separated by order. (order=1, nearest neighbors).
        """
        raise InterfaceMethodException

    @abstractmethod
    def get_edges(self, qubit: IQubitID) -> List[IEdgeID]:
        """:return: All qubit-to-qubit edges from qubit-ID."""
        raise InterfaceMethodException

    @abstractmethod
    def contains(self, element: Union[IFeedlineID, IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of device layer or not."""
        raise InterfaceMethodException
    # endregion


@unique
class ParityType(Enum):
    STABILIZER_X = 0
    STABILIZER_Z = 1


class IParityGroup(ABC):
    """
    Interface class, describing qubit (nodes) and edges related to the parity group.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def parity_type(self) -> ParityType:
        """:return: Parity type (X or Z type stabilizer)."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def ancilla_id(self) -> IQubitID:
        """:return: (Main) ancilla-qubit-ID from parity."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def data_ids(self) -> List[IQubitID]:
        """:return: (All) data-qubit-ID's from parity."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def edge_ids(self) -> List[IEdgeID]:
        """:return: (All) edge-ID's between ancilla and data qubit-ID's."""
        raise InterfaceMethodException

    # endregion

    # region Interface Methods
    @abstractmethod
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of parity group or not."""
        raise InterfaceMethodException
    # endregion


class IGateGroup(ABC):
    """
    Interface class, describing 2-qubit gate (edge) and corresponding 'spectator' qubits (nodes).
    Spectators are nearest neighbors around both qubits involved in the 2-qubit gate.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def gate_id(self) -> IEdgeID:
        """:return: Edge involved in the 2-qubit gate."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def involved_ids(self) -> List[IQubitID]:
        """:return: (All) qubit-ID's (directly) involved with 2-qubit gate."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def spectator_ids(self) -> List[IQubitID]:
        """:return: (All) qubit-ID's (indirectly) involved with 2-qubit gate."""
        raise InterfaceMethodException

    # endregion

    # region Interface Methods
    @abstractmethod
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of gate group or not."""
        raise InterfaceMethodException

    @abstractmethod
    def get_spectator_edge(self, spectator: IQubitID) -> IEdgeID:
        """
        Checks if spectator is part of group-spectators, if not raise ElementNotIncludedException.
        :return: Edge that links spectator to one of the involved qubits.
        """
        raise InterfaceMethodException
    # endregion


@unique
class FrequencyGroup(Enum):
    LOW = auto()
    MID = auto()
    HIGH = auto()


@dataclass(frozen=True)
class FrequencyGroupIdentifier:
    """
    Data class, representing (qubit) frequency group identifier.
    """
    _id: FrequencyGroup

    # region Class Properties
    @property
    def id(self) -> FrequencyGroup:
        """:return: Self identifier."""
        return self._id

    # endregion

    # region Class Methods
    def is_equal_to(self, other: 'FrequencyGroupIdentifier') -> bool:
        """:return: Boolean, whether other frequency group identifier is equal self."""
        return self.id == other.id

    def is_higher_than(self, other: 'FrequencyGroupIdentifier') -> bool:
        """:return: Boolean, whether other frequency group identifier is 'lower' than self."""
        # Guard clause, if frequency groups are equal, return False
        if self.is_equal_to(other):
            return False
        if self.id == FrequencyGroup.MID and other.id == FrequencyGroup.LOW:
            return True
        if self.id == FrequencyGroup.HIGH:
            return True
        return False

    def is_lower_than(self, other: 'FrequencyGroupIdentifier') -> bool:
        """:return: Boolean, whether other frequency group identifier is 'higher' than self."""
        # Guard clause, if frequency groups are equal, return False
        if self.is_equal_to(other):
            return False
        if self.is_higher_than(other):
            return False
        return True
    # endregion


class ISurfaceCodeLayer(IDeviceLayer, metaclass=ABCMeta):
    """
    Interface class, describing surface-code relation based connectivity.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def parity_group_x(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of X-stabilizers."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def parity_group_z(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of Z-stabilizers."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def data_qubit_ids(self) -> List[IQubitID]:
        """:return: (Data) qubit-ID's in device layer."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def ancilla_qubit_ids(self) -> List[IQubitID]:
        """:return: (Ancilla) qubit-ID's in device layer."""
        raise InterfaceMethodException

    # endregion

    # region Interface Methods
    @abstractmethod
    def get_parity_group(self, element: Union[IQubitID, IEdgeID]) -> IParityGroup:
        """:return: Parity group of which element (edge- or qubit-ID) is part of."""
        raise InterfaceMethodException

    @abstractmethod
    def get_frequency_group_identifier(self, element: IQubitID) -> FrequencyGroupIdentifier:
        """:return: Frequency group identifier based on qubit-ID."""
        raise InterfaceMethodException
    # endregion
