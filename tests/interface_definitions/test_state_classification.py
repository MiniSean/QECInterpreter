import unittest
import numpy as np
from numpy.testing import assert_array_equal
from qce_interp.interface_definitions.intrf_state_classification import (
    StateKey,
    StateAcquisition,
    StateBoundaryKey,
    DecisionBoundaries,
    StateAcquisitionContainer,
    StateClassifierContainer,
    ParityType,
)
from qce_interp.utilities.geometric_definitions import Vec2D


class StateAcquisitionTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        pass

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_center(self):
        """Tests center property of state acquisition class. 2D Gaussian data centered at (1 + 1j)"""
        shots = np.random.normal(1, 0.1, 1000) + 1j * np.random.normal(1, 0.1, 1000)
        acquisition = StateAcquisition(state=StateKey.STATE_0, shots=shots)
        center = acquisition.center
        self.assertAlmostEqual(center.x, 1, delta=0.1)
        self.assertAlmostEqual(center.y, 1, delta=0.1)

    def test_concatenated_shots(self):
        """Tests concatenate shots shape"""
        shots_0 = np.random.normal(1, 0.1, 1000) + 1j * np.random.normal(1, 0.1, 1000)
        shots_1 = np.random.normal(2, 0.1, 1000) + 1j * np.random.normal(2, 0.1, 1000)
        acquisitions = [StateAcquisition(StateKey.STATE_0, shots_0), StateAcquisition(StateKey.STATE_1, shots_1)]
        container = StateAcquisitionContainer.from_state_acquisitions(acquisitions)
        concatenated_shots = container.concatenated_shots
        self.assertEqual(
            concatenated_shots.shape,
            (2000,)
        )
        self.assertAlmostEqual(
            container.estimate_threshold, 1.5, delta=0.1
        )

    def test_equality(self):
        """Tests bi-directionality of state boundary key"""
        key1 = StateBoundaryKey(state_a=StateKey.STATE_0, state_b=StateKey.STATE_1)
        key2 = StateBoundaryKey(state_a=StateKey.STATE_1, state_b=StateKey.STATE_0)
        self.assertEqual(key1, key2)
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion


class DecisionBoundaryTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        shots_0 = np.random.normal(1, 0.1, 1000) + 1j * np.random.normal(1, 0.1, 1000)
        shots_1 = np.random.normal(2, 0.1, 1000) + 1j * np.random.normal(2, 0.1, 1000)
        acquisitions = [StateAcquisition(StateKey.STATE_0, shots_0), StateAcquisition(StateKey.STATE_1, shots_1)]
        container = StateAcquisitionContainer.from_state_acquisitions(acquisitions)
        cls.decision_boundaries: DecisionBoundaries = container.decision_boundaries

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_mean(self):
        """Tests the mean property of DecisionBoundaries."""
        mean = self.decision_boundaries.mean
        # Test the mean property
        self.assertAlmostEqual(mean.x, 1, delta=0.1)
        self.assertAlmostEqual(mean.y, 2, delta=0.1)
        self.assertIsInstance(mean, Vec2D, msg="Mean is not a Vec2D instance.")

    def test_state_prediction_index_lookup(self):
        """Tests the state_prediction_index_lookup property of DecisionBoundaries."""
        lookup = self.decision_boundaries.state_prediction_index_lookup
        # Test the lookup property
        self.assertIsInstance(lookup, dict, msg="Lookup is not a dictionary.")
        self.assertTrue(
            StateKey.STATE_0 in lookup,
            msg=f"State key 0 is not in lookup. instead: {list(lookup.keys())}"
        )
        self.assertTrue(
            StateKey.STATE_1 in lookup,
            msg=f"State key 1 is not in lookup. instead: {list(lookup.keys())}"
        )

    def test_prediction_index_to_state_lookup(self):
        """Tests the prediction_index_to_state_lookup property of DecisionBoundaries."""
        lookup = self.decision_boundaries.prediction_index_to_state_lookup
        # Test the lookup property
        values = list(lookup.values())
        self.assertTrue(
            StateKey.STATE_0 in values,
            msg=f"State key 0 is not in lookup values. instead: {values}"
        )
        self.assertTrue(
            StateKey.STATE_1 in values,
            msg=f"State key 1 is not in lookup values. instead: {values}"
        )

    def test_decision_boundaries(self):
        """Tests state prediction based on new data."""
        expected_state_0: complex = 1 + 1j
        expected_state_1: complex = 2 + 2j
        self.assertEqual(
            self.decision_boundaries.get_prediction(expected_state_0),
            StateKey.STATE_0,
            msg=f"Expects value to be classified as state 0."
        )
        self.assertEqual(
            self.decision_boundaries.get_prediction(expected_state_1),
            StateKey.STATE_1,
            msg=f"Expects value to be classified as state 1."
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion


class StateClassificationTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        shots_0 = np.random.normal(1, 0.1, 1000) + 1j * np.random.normal(1, 0.1, 1000)
        shots_1 = np.random.normal(2, 0.1, 1000) + 1j * np.random.normal(2, 0.1, 1000)
        acquisitions = [StateAcquisition(StateKey.STATE_0, shots_0), StateAcquisition(StateKey.STATE_1, shots_1)]
        container = StateAcquisitionContainer.from_state_acquisitions(acquisitions)
        decision_boundaries: DecisionBoundaries = container.decision_boundaries

        semi_random_shots = np.asarray([
            1.62971825+1.93007721j, 1.04820391+1.22129528j, 1.26596653+1.68535489j,
            1.18107149+1.51606292j, 1.72287545+1.68360079j, 1.4504699 +1.77819718j,
            1.76327133+1.45279363j, 1.55082547+1.55100055j, 1.17130812+1.6237312j,
            1.97898141+1.32935511j
        ])
        p: ParityType = ParityType.ODD
        cls.expected_binary = np.asarray([1, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        cls.expected_eigenvalue = np.asarray([-1,  1,  1,  1, -1, -1, -1, -1,  1, -1])
        e0: int = cls.expected_eigenvalue[0]  # First round we effectively measure the parity
        cls.expected_parity = np.asarray([e0, -1,  1,  1, -1,  1,  1,  1, -1, -1])
        p0: int = cls.expected_parity[0]
        p1: int = cls.expected_parity[1]
        cls.expected_defect = np.asarray([p.value * p0,  p0 * p1, -1,  1, -1, -1,  1,  1, -1,  1])

        cls.state_classifier: StateClassifierContainer = StateClassifierContainer(
            shots=semi_random_shots,
            decision_boundaries=decision_boundaries,
            expected_parity=p,
        )

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_conversion_eigenvalue_to_parity(self):
        """Tests correct conversion to parity."""
        eigenvalues = np.copy(self.expected_eigenvalue)
        eigenvalues[0] = -1
        parities = StateClassifierContainer.calculate_parity(
            m=eigenvalues,
        )
        self.assertEqual(
            eigenvalues[0],
            parities[0],
        )
        eigenvalues[0] = +1
        parities = StateClassifierContainer.calculate_parity(
            m=eigenvalues,
        )
        self.assertEqual(
            eigenvalues[0],
            parities[0],
        )

    def test_conversion_eigenvalue_to_parity_tensor(self):
        """Tests correct conversion to parity."""
        eigenvalues: np.ndarray = np.asarray([
            [+1, -1],
            [+1, -1],
            [-1, +1],
            [-1, +1],
        ])
        parities = StateClassifierContainer.calculate_parity(
            m=eigenvalues,
        )
        assert_array_equal(
            parities,
            np.asarray([
                [+1, -1],
                [+1, -1],
                [-1, -1],
                [-1, -1],
            ]),
            err_msg="Expects first column to be equal to observed ancilla outcomes."
        )
        eigenvalues: np.ndarray = np.asarray([
            [+1],
            [+1],
            [-1],
            [-1],
        ])
        parities = StateClassifierContainer.calculate_parity(
            m=eigenvalues,
        )
        assert_array_equal(
            parities,
            np.asarray([
                [+1],
                [+1],
                [-1],
                [-1],
            ]),
            err_msg="Expects first column to be equal to observed ancilla outcomes. Regardless of cycle length"
        )

    def test_conversion_parity_to_defect_tensor(self):
        """Tests correct conversion to parity."""
        parities: np.ndarray = np.asarray([
            [+1, -1],
            [+1, -1],
            [-1, -1],
            [-1, -1],
        ])
        defect = StateClassifierContainer.calculate_defect(
            m=parities,
            initial_condition=-1
        )
        assert_array_equal(
            defect,
            np.asarray([
                [-1, -1],
                [-1, -1],
                [+1, +1],
                [+1, +1],
            ]),
            err_msg="Expects first column to depend on both stabilizer parities and expected initial condition."
        )
        parities: np.ndarray = np.asarray([
            [+1],
            [+1],
            [-1],
            [-1],
        ])
        defect = StateClassifierContainer.calculate_defect(
            m=parities,
            initial_condition=-1
        )
        assert_array_equal(
            defect,
            np.asarray([
                [-1],
                [-1],
                [+1],
                [+1],
            ]),
            err_msg="Expects first column to depend on both stabilizer parities and expected initial condition. Regardless of cycle length"
        )

    def test_binary_classification(self):
        """Tests binary classification of complex shots."""
        assert_array_equal(
            self.state_classifier.get_binary_classification(),
            self.expected_binary,
        )

    def test_eigenvalue_classification(self):
        """Tests eigenvalue classification of complex shots."""
        # x = self.state_classifier.get_eigenvalue_classification()
        # y = self.expected_eigenvalue
        assert_array_equal(
            self.state_classifier.get_eigenvalue_classification(),
            self.expected_eigenvalue,
        )

    def test_parity_classification(self):
        """Tests parity classification of complex shots."""
        assert_array_equal(
            self.state_classifier.get_parity_classification(),
            self.expected_parity,
        )

    def test_defect_classification(self):
        """Tests defect classification of complex shots."""
        assert_array_equal(
            self.state_classifier.get_defect_classification(),
            self.expected_defect,
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
