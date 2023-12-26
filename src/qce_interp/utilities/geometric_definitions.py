# -------------------------------------------
# Module containing vector definitions.
# -------------------------------------------
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class Vec2D:
    """
    Data class, containing x- and y-coordinate vector.
    """
    x: float
    y: float

    # region Class Methods
    def to_vector(self) -> np.ndarray:
        return np.asarray([self.x, self.y])

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> 'Vec2D':
        return Vec2D(
            x=vector[0],
            y=vector[1],
        )

    def __add__(self, other):
        if isinstance(other, Vec2D):
            return Vec2D(x=self.x + other.x, y=self.y + other.y)
        raise NotImplemented(f"Addition with anything other than {Vec2D} is not implemented.")
    # endregion
