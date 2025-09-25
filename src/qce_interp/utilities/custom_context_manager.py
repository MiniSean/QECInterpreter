# -------------------------------------------
# Customized context managers for better maintainability
# -------------------------------------------
import warnings


class WhileLoopSafetyExceededWarning(Warning):
    """
    Raised when while-loop safety counter exceeds the allowed number of iterations.
    """

    # region Class Methods
    @classmethod
    def warning_format(cls, max_iter: int) -> dict:
        return dict(
            message=f"Max iterations reached ({max_iter}/{max_iter}), exiting loop.",
            category=cls,
        )
    # endregion


class WhileLoopSafety:
    """
    Context manager class,
    """

    # region Class Constructor
    def __init__(self, max_iterations: int = 10):
        self.counter = 0
        self.max_iterations = max_iterations
    # endregion

    # region Class Methods
    def safety_condition(self):
        if self.counter >= self.max_iterations:
            warnings.warn(**WhileLoopSafetyExceededWarning.warning_format(max_iter=self.max_iterations))
            return False
        self.counter += 1
        return True

    def __enter__(self) -> 'WhileLoopSafety':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    # endregion
