# -------------------------------------------
# Module containing implementation of surface-code connectivity structure.
# -------------------------------------------
from dataclasses import dataclass, field
from typing import List, Union, Dict
from definitions import SingletonABCMeta
from qce_interp.custom_exceptions import ElementNotIncludedException
from qce_interp.interface_definitions.intrf_channel_identifier import (
    IFeedlineID,
    IQubitID,
    IEdgeID,
    FeedlineIDObj,
    QubitIDObj,
    EdgeIDObj,
)
from qce_interp.interface_definitions.intrf_connectivity_surface_code import (
    ISurfaceCodeLayer,
    IParityGroup,
    IGateGroup,
    ParityType,
    FrequencyGroup,
    FrequencyGroupIdentifier,
)


@dataclass(frozen=True)
class ParityGroup(IParityGroup):
    """
    Data class, implementing IParityGroup interface.
    """
    _parity_type: ParityType = field(init=True)
    """X or Z type stabilizer."""
    _ancilla_qubit: IQubitID = field(init=True)
    """Ancilla qubit."""
    _data_qubits: List[IQubitID] = field(init=True)
    """Data qubits."""
    _edges: List[IEdgeID] = field(init=False)
    """Edges between ancilla and data qubits."""

    # region Interface Properties
    @property
    def parity_type(self) -> ParityType:
        """:return: Parity type (X or Z type stabilizer)."""
        return self._parity_type

    @property
    def ancilla_id(self) -> IQubitID:
        """:return: (Main) ancilla-qubit-ID from parity."""
        return self._ancilla_qubit

    @property
    def data_ids(self) -> List[IQubitID]:
        """:return: (All) data-qubit-ID's from parity."""
        return self._data_qubits

    @property
    def edge_ids(self) -> List[IEdgeID]:
        """:return: (All) edge-ID's between ancilla and data qubit-ID's."""
        return self._edges

    # endregion

    # region Interface Methods
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of parity group or not."""
        if element in self.data_ids + [self.ancilla_id]:
            return True
        if element in self.edge_ids:
            return True
        return False

    # endregion

    # region Class Methods
    def __post_init__(self):
        edges: List[IEdgeID] = [
            EdgeIDObj(
                qubit_id0=self.ancilla_id,
                qubit_id1=data_qubit_id,
            )
            for data_qubit_id in self.data_ids
        ]
        object.__setattr__(self, '_edges', edges)
    # endregion


@dataclass(frozen=True)
class GateGroup(IGateGroup):
    """
    Data class, implementing IGateGroup interface.
    """
    _gate_edge: IEdgeID
    """Edge that determines two-qubit gate."""

    # region Interface Properties
    @property
    def gate_id(self) -> IEdgeID:
        """:return: Edge involved in the 2-qubit gate."""
        return self._gate_edge

    @property
    def involved_ids(self) -> List[IQubitID]:
        """:return: (All) qubit-ID's (directly) involved with 2-qubit gate."""
        return self.gate_id.qubit_ids

    @property
    def spectator_ids(self) -> List[IQubitID]:
        """:return: (All) qubit-ID's (indirectly) involved with 2-qubit gate."""
        involved_qubit_ids: List[IQubitID] = self.involved_ids
        neighbor_qubit_ids: List[IQubitID] = Surface17Layer().get_neighbors(qubit=involved_qubit_ids[0], order=1) \
                                             + Surface17Layer().get_neighbors(qubit=involved_qubit_ids[1], order=1)
        return [element for element in set(neighbor_qubit_ids) if element not in involved_qubit_ids]

    # endregion

    # region Interface Methods
    def contains(self, element: Union[IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of gate group or not."""
        if element in self.spectator_ids + self.gate_id.qubit_ids:
            return True
        if element in [self.gate_id]:
            return True
        return False

    def get_spectator_edge(self, spectator: IQubitID) -> IEdgeID:
        """
        Checks if spectator is part of group-spectators, if not raise ElementNotIncludedException.
        :return: Edge that links spectator to one of the involved qubits.
        """
        # Guard clause, raise exception if spectator not part of group-spectator
        if spectator not in self.spectator_ids:
            raise ElementNotIncludedException(f"Element: {spectator} not part of {self.spectator_ids}.")

        # Guaranteed that one of the potential edges are correct
        involved_qubit_ids: List[IQubitID] = self.involved_ids
        for potential_edge in Surface17Layer().get_edges(qubit=spectator):
            if potential_edge.get_connected_qubit_id(element=spectator) in involved_qubit_ids:
                return potential_edge
    # endregion


class Surface17Layer(ISurfaceCodeLayer, metaclass=SingletonABCMeta):
    """
    Singleton class, implementing ISurfaceCodeLayer interface to describe a surface-17 layout.
    """
    _feedline_qubit_lookup: Dict[IFeedlineID, List[IQubitID]] = {
        FeedlineIDObj('FL1'): [QubitIDObj('D9'), QubitIDObj('D8'), QubitIDObj('X4'), QubitIDObj('Z4'), QubitIDObj('Z2'),
                               QubitIDObj('D6')],
        FeedlineIDObj('FL2'): [QubitIDObj('D3'), QubitIDObj('D7'), QubitIDObj('D2'), QubitIDObj('X3'), QubitIDObj('Z1'),
                               QubitIDObj('X2'), QubitIDObj('Z3'), QubitIDObj('D5'), QubitIDObj('D4')],
        FeedlineIDObj('FL3'): [QubitIDObj('D1'), QubitIDObj('X1')],
    }
    _qubit_edges: List[IEdgeID] = [
        EdgeIDObj(QubitIDObj('D1'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D1'), QubitIDObj('X1')),
        EdgeIDObj(QubitIDObj('D2'), QubitIDObj('X1')),
        EdgeIDObj(QubitIDObj('D2'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D2'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D3'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D3'), QubitIDObj('Z2')),
        EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z3')),
        EdgeIDObj(QubitIDObj('D4'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D4'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z1')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D5'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D6'), QubitIDObj('X2')),
        EdgeIDObj(QubitIDObj('D6'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D6'), QubitIDObj('Z2')),
        EdgeIDObj(QubitIDObj('D7'), QubitIDObj('Z3')),
        EdgeIDObj(QubitIDObj('D7'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D8'), QubitIDObj('X3')),
        EdgeIDObj(QubitIDObj('D8'), QubitIDObj('X4')),
        EdgeIDObj(QubitIDObj('D8'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D9'), QubitIDObj('Z4')),
        EdgeIDObj(QubitIDObj('D9'), QubitIDObj('X4')),
    ]
    _parity_group_x: List[IParityGroup] = [
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X1'),
            _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X2'),
            _data_qubits=[QubitIDObj('D2'), QubitIDObj('D3'), QubitIDObj('D5'), QubitIDObj('D6')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X3'),
            _data_qubits=[QubitIDObj('D4'), QubitIDObj('D5'), QubitIDObj('D7'), QubitIDObj('D8')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X4'),
            _data_qubits=[QubitIDObj('D8'), QubitIDObj('D9')]
        ),
    ]
    _parity_group_z: List[IParityGroup] = [
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z1'),
            _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2'), QubitIDObj('D4'), QubitIDObj('D5')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z2'),
            _data_qubits=[QubitIDObj('D3'), QubitIDObj('D6')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z3'),
            _data_qubits=[QubitIDObj('D4'), QubitIDObj('D7')]
        ),
        ParityGroup(
            _parity_type=ParityType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z4'),
            _data_qubits=[QubitIDObj('D5'), QubitIDObj('D6'), QubitIDObj('D8'), QubitIDObj('D9')]
        ),
    ]
    _frequency_group_lookup: Dict[IQubitID, FrequencyGroupIdentifier] = {
        QubitIDObj('D1'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D2'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D3'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D4'): FrequencyGroupIdentifier(_id=FrequencyGroup.HIGH),
        QubitIDObj('D5'): FrequencyGroupIdentifier(_id=FrequencyGroup.HIGH),
        QubitIDObj('D6'): FrequencyGroupIdentifier(_id=FrequencyGroup.HIGH),
        QubitIDObj('D7'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D8'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('D9'): FrequencyGroupIdentifier(_id=FrequencyGroup.LOW),
        QubitIDObj('Z1'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('Z2'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('Z3'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('Z4'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X1'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X2'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X3'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
        QubitIDObj('X4'): FrequencyGroupIdentifier(_id=FrequencyGroup.MID),
    }

    # region IDeviceLayer Interface Properties
    @property
    def feedline_ids(self) -> List[IFeedlineID]:
        """:return: All feedline-ID's in device layer."""
        return list(self._feedline_qubit_lookup.keys())

    @property
    def qubit_ids(self) -> List[IQubitID]:
        """:return: (All) qubit-ID's in device layer."""
        return [qubit_id for qubit_ids in self._feedline_qubit_lookup.values() for qubit_id in qubit_ids]

    @property
    def edge_ids(self) -> List[IEdgeID]:
        """:return: (All) edge-ID's in device layer."""
        return self._qubit_edges

    # endregion

    # region ISurfaceCodeLayer Interface Properties
    @property
    def parity_group_x(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of X-stabilizers."""
        return self._parity_group_x

    @property
    def parity_group_z(self) -> List[IParityGroup]:
        """:return: (All) parity groups part of Z-stabilizers."""
        return self._parity_group_z

    @property
    def data_qubit_ids(self) -> List[IQubitID]:
        """:return: (Data) qubit-ID's in device layer."""
        all_qubit_ids: List[IQubitID] = self.qubit_ids
        ancilla_qubit_ids: List[IQubitID] = self.ancilla_qubit_ids
        return [qubit_id for qubit_id in all_qubit_ids if qubit_id not in ancilla_qubit_ids]

    @property
    def ancilla_qubit_ids(self) -> List[IQubitID]:
        """:return: (Ancilla) qubit-ID's in device layer."""
        return [parity_group.ancilla_id for parity_group in self.parity_group_x + self.parity_group_z]

    # endregion

    # region ISurfaceCodeLayer Interface Methods
    def get_parity_group(self, element: Union[IQubitID, IEdgeID]) -> IParityGroup:
        """:return: Parity group of which element (edge- or qubit-ID) is part of."""
        # Assumes element is part of only a single parity group
        for parity_group in self.parity_group_x + self.parity_group_z:
            if parity_group.contains(element=element):
                return parity_group
        raise ElementNotIncludedException(f"Element: {element} is not included in any parity group.")

    def get_frequency_group_identifier(self, element: IQubitID) -> FrequencyGroupIdentifier:
        """:return: Frequency group identifier based on qubit-ID."""
        return self._frequency_group_lookup[element]

    # endregion

    # region IDeviceLayer Interface Methods
    def get_connected_qubits(self, feedline: IFeedlineID) -> List[IQubitID]:
        """:return: Qubit-ID's connected to feedline-ID."""
        # Guard clause, if feedline not in lookup, raise exception
        if feedline not in self._feedline_qubit_lookup:
            raise ElementNotIncludedException(f"Element: {feedline} is not included in any feedline group.")
        return self._feedline_qubit_lookup[feedline]

    def get_connected_feedline(self, qubit: IQubitID) -> IFeedlineID:
        """:return: Feedline-ID's connected to qubit-ID."""
        for feedline_id in self.feedline_ids:
            if qubit in self.get_connected_qubits(feedline=feedline_id):
                return feedline_id
        raise ElementNotIncludedException(f"Element: {qubit} is not included in any feedline.")

    def get_neighbors(self, qubit: IQubitID, order: int = 1) -> List[IQubitID]:
        """
        Requires :param order: to be higher or equal to 1.
        :return: qubit neighbors separated by order. (order=1, nearest neighbors).
        """
        if order > 1:
            raise NotImplementedError("Apologies, so far there has not been a use for. But feel free to implement.")
        edges: List[IEdgeID] = self.get_edges(qubit=qubit)
        result: List[IQubitID] = []
        for edge in edges:
            result.append(edge.get_connected_qubit_id(element=qubit))
        return result

    def get_edges(self, qubit: IQubitID) -> List[IEdgeID]:
        """:return: All qubit-to-qubit edges from qubit-ID."""
        result: List[IEdgeID] = []
        for edge in self.edge_ids:
            if edge.contains(element=qubit):
                result.append(edge)
        return result

    def contains(self, element: Union[IFeedlineID, IQubitID, IEdgeID]) -> bool:
        """:return: Boolean, whether element is part of device layer or not."""
        if element in self.feedline_ids:
            return True
        if element in self.qubit_ids:
            return True
        if element in self.edge_ids:
            return True
        return False
    # endregion
