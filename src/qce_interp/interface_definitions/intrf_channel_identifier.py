# -------------------------------------------
# Interface for unique channel references
# For example:
# Qubit identifier, Feedline identifier, Flux channel identifier, etc.
# -------------------------------------------
from qce_circuit.connectivity.intrf_channel_identifier import (
    IChannelIdentifier,
    IQubitID,
    IFeedlineID,
    IEdgeID,
    QubitIDObj,
    FeedlineIDObj,
    EdgeIDObj,
)

__all__ = [
    "IChannelIdentifier",
    "IQubitID",
    "IFeedlineID",
    "IEdgeID",
    "QubitIDObj",
    "FeedlineIDObj",
    "EdgeIDObj",
]
