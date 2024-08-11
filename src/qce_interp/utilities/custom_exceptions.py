# -------------------------------------------
# Customized exceptions for better maintainability
# -------------------------------------------


class InterfaceMethodException(Exception):
    """
    Raised when the interface method is not implemented.
    """


class ElementNotIncludedException(Exception):
    """
    Raised when element (such as IQubitID, IEdgeID or IFeedlineID) is not included in the connectivity layer.
    """


class QECCycleNotIncludedException(Exception):
    """
    Raised when requested QEC-cycle is not included in the data.
    """


class ZeroClassifierShotsException(Exception):
    """
    Raised when state-classification is attempted with zero (0) classification shots.
    """



class InsufficientParityInformationException(Exception):
    """
    Raised when parity can not be computed due to insufficient data.
    """
