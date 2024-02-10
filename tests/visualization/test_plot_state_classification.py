import unittest
from typing import List
import numpy as np
from numpy.testing import assert_array_equal
from qce_interp.visualization.plot_state_classification import (
    filter_vertices_within_smaller_angle,
)
from qce_interp.utilities.geometric_definitions import Vec2D, Polygon


class DecisionRegionTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_correct_assessment_of_decision_region(self):
        """Tests default True."""
        center: Vec2D = Vec2D(x=0.15142037213185658, y=-0.03860353964415292)
        intersection_a: Vec2D = Vec2D(x=0.20131056275101383, y=1.1)
        intersection_b: Vec2D = Vec2D(x=0.8941016810504318, y=-1.1)
        corner_a: Vec2D = Vec2D(x=-1.1000000021999998, y=-1.1)
        corner_b: Vec2D = Vec2D(x=1.1000000021999998, y=-1.1)
        corner_c: Vec2D = Vec2D(x=1.1000000021999998, y=1.1)
        corner_d: Vec2D = Vec2D(x=-1.1000000021999998, y=1.1)
        rectangle_vertices: List[Vec2D] = [corner_a, corner_b, corner_c, corner_d]

        filtered_vertices: List[Vec2D] = filter_vertices_within_smaller_angle(
            center=center,
            intersection1=intersection_a,
            intersection2=intersection_b,
            vertices=rectangle_vertices,
        )
        filtered_vertices_alternative: List[Vec2D] = filter_vertices_within_smaller_angle(
            center=center,
            intersection1=intersection_b,
            intersection2=intersection_a,
            vertices=rectangle_vertices,
        )
        assert_array_equal(
            np.asarray([vertex.to_vector() for vertex in filtered_vertices]),
            np.asarray([vertex.to_vector() for vertex in filtered_vertices_alternative]),
            err_msg="Order of intersection should not influence the final result."
        )
        self.assertTrue(
            corner_a not in filtered_vertices
        )
        self.assertTrue(
            corner_b in filtered_vertices
        )
        self.assertTrue(
            corner_c in filtered_vertices
        )
        self.assertTrue(
            corner_d not in filtered_vertices
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
