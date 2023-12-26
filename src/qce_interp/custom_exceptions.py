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
