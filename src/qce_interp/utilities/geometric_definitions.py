# -------------------------------------------
# Module containing vector definitions.
# -------------------------------------------
from dataclasses import dataclass
from typing import Tuple, List
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

    def __mul__(self, other):
        return Vec2D(x=self.x.__mul__(other), y=self.y.__mul__(other))

    def __rmul__(self, other):
        return Vec2D(x=self.x.__rmul__(other), y=self.y.__rmul__(other))

    def __sub__(self, other):
        return self.__add__(-1 * other)
    # endregion


@dataclass(frozen=True)
class Polygon:
    """
    Data class, containing Vec2D describing a polygon.
    """
    vertices: List[Vec2D]

    # region Class Methods
    def get_convex_vertices(self) -> List[Vec2D]:
        """
        Orders a list of Vec2D points to form a non-self-intersecting polygon.
        :return: List of Vec2D vertices ordered to form a non-self-intersecting polygon.
        """

        def centroid(points: List[Vec2D]) -> Vec2D:
            """Calculates the centroid of a polygon given its vertices."""
            x = [p.x for p in points]
            y = [p.y for p in points]
            centroid_x = sum(x) / len(points)
            centroid_y = sum(y) / len(points)
            return Vec2D(centroid_x, centroid_y)

        def angle_from_centroid(point: Vec2D, centroid: Vec2D) -> float:
            """Calculates the angle between a point and the centroid of the polygon."""
            return np.arctan2(point.y - centroid.y, point.x - centroid.x)

        # Calculate the centroid of the polygon
        center = centroid(self.vertices)

        # Sort the vertices based on their angle from the centroid
        sorted_vertices = sorted(self.vertices, key=lambda point: angle_from_centroid(point, center))

        return sorted_vertices
    # endregion


def euclidean_distance(p1: Vec2D, p2: Vec2D):
    """
    Calculates Euclidean distance between two points.

    :param p1: First point.
    :param p2: Second point.
    :return: Euclidean distance between p1 and p2.
    """
    return np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
