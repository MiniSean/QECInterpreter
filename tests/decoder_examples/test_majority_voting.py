import unittest
from unittest.mock import MagicMock
import numpy as np
from qce_circuit.connectivity.intrf_connectivity_surface_code import StabilizerType
from qce_interp.decoder_examples.majority_voting import MajorityVotingDecoder


class MajorityVotingDecoderTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases."""
        pass

    def setUp(self) -> None:
        """Set up for every test case."""
        self.mock_error_identifier = MagicMock()
    # endregion

    # region Test Cases
    def test_majority_voting_z_basis_odd_cycle(self) -> None:
        """Tests majority voting in Z-basis for odd cycle count without refocusing."""
        # Shape: (N=4, M=1, D=5)
        # Sample 0: [0, 0, 0, 0, 0] (0 errors -> majority 0)
        # Sample 1: [1, 1, 0, 0, 0] (2 errors -> majority 0)
        # Sample 2: [1, 1, 1, 0, 0] (3 errors -> majority 1)
        # Sample 3: [1, 1, 1, 1, 1] (5 errors -> majority 1)
        raw_output = np.array([
            [[0, 0, 0, 0, 0]],
            [[1, 1, 0, 0, 0]],
            [[1, 1, 1, 0, 0]],
            [[1, 1, 1, 1, 1]],
        ], dtype=int)
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output

        decoder = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_Z,
            include_dynamical_decoupling_correction=True,
        )
        target_state = np.zeros(5, dtype=int)
        fidelity = decoder.get_fidelity(cycle_stabilizer_count=1, target_state=target_state)

        # 2 out of 4 samples match majority of target_state [0, 0, 0, 0, 0]
        self.assertEqual(fidelity, 0.5)

    def test_majority_voting_z_basis_even_cycle_refocusing(self) -> None:
        """Tests dynamical decoupling bit-flip correction for even cycle counts in Z-basis."""
        # For even cycle_stabilizer_count (e.g. 2), refocusing inverts eigenvalues.
        # Raw all 1s should be inverted to all 0s.
        raw_output = np.array([
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
        ], dtype=int)
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output

        decoder = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_Z,
            include_dynamical_decoupling_correction=True,
        )
        target_state = np.zeros(5, dtype=int)
        fidelity = decoder.get_fidelity(cycle_stabilizer_count=2, target_state=target_state)

        # Raw 1s inverted to 0s by refocusing correction -> matches target 0s
        self.assertEqual(fidelity, 1.0)

    def test_majority_voting_x_basis(self) -> None:
        """Tests X-basis majority voting where output is inverted."""
        # For X-basis, corrected_binary_output = corrected_binary_output ^ 1
        raw_output = np.array([
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 0, 0]],
        ], dtype=int)
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output

        decoder = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_X,
            include_dynamical_decoupling_correction=True,
        )
        target_state = np.zeros(5, dtype=int)
        # Cycle 1: X-basis inverts [1,1,1,1,1] to [0,0,0,0,0] (matches target)
        # and [1,1,1,0,0] to [0,0,0,1,1] (majority 0, matches target)
        fidelity = decoder.get_fidelity(cycle_stabilizer_count=1, target_state=target_state)
        self.assertEqual(fidelity, 1.0)

    def test_majority_voting_target_state_logical_parity(self) -> None:
        """Tests target logical state correction when target_state has odd parity."""
        # Target state with odd parity: sum([1, 0, 0, 0, 0]) % 2 == 1
        # target_logical_state = 1
        raw_output = np.array([
            [[1, 0, 0, 0, 0]],
        ], dtype=int)
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output

        decoder = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_Z,
            include_dynamical_decoupling_correction=False,
        )
        target_state = np.array([1, 0, 0, 0, 0], dtype=int)
        # raw [1,0,0,0,0] ^ target_logical_state(1) = [0,1,1,1,1]
        # target_state is [1,0,0,0,0]. Distance between [0,1,1,1,1] and [1,0,0,0,0] is 5 > 2.5
        fidelity = decoder.get_fidelity(cycle_stabilizer_count=1, target_state=target_state)
        self.assertEqual(fidelity, 0.0)

        # Conversely, if raw output is [0, 1, 1, 1, 1] ^ 1 = [1, 0, 0, 0, 0] -> matches target!
        raw_output_matching = np.array([
            [[0, 1, 1, 1, 1]],
        ], dtype=int)
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output_matching
        fidelity_matching = decoder.get_fidelity(cycle_stabilizer_count=1, target_state=target_state)
        self.assertEqual(fidelity_matching, 1.0)

    def test_majority_voting_dynamical_decoupling_toggle(self) -> None:
        """Tests that disabling dynamical decoupling correction bypasses both X-basis and even cycle flips."""
        raw_output = np.array([
            [[1, 1, 1, 1, 1]],
        ], dtype=int)
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output

        # Disabled toggle: should NOT invert X-basis or even-cycle
        decoder_no_dd = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_X,
            include_dynamical_decoupling_correction=False,
        )
        target_state = np.zeros(5, dtype=int)
        # Without DD correction, raw [1,1,1,1,1] is not inverted to 0 -> fails majority matching against 0
        fidelity = decoder_no_dd.get_fidelity(cycle_stabilizer_count=2, target_state=target_state)
        self.assertEqual(fidelity, 0.0)

        # With DD correction enabled, X-basis inverts to 0 -> passes
        decoder_with_dd = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_X,
            include_dynamical_decoupling_correction=True,
        )
        fidelity_with_dd = decoder_with_dd.get_fidelity(cycle_stabilizer_count=1, target_state=target_state)
        self.assertEqual(fidelity_with_dd, 1.0)

    def test_majority_voting_uncertainty(self) -> None:
        """Tests get_fidelity_uncertainty calculation based on binomial distribution."""
        # 100 samples with 50 matching -> fidelity = 0.5
        raw_output = np.zeros((100, 1, 5), dtype=int)
        # First 50 samples are [0,0,0,0,0] (matches target [0,0,0,0,0])
        # Last 50 samples are [1,1,1,1,1] (does not match)
        raw_output[50:] = 1
        self.mock_error_identifier.get_binary_projected_classification.return_value = raw_output

        decoder = MajorityVotingDecoder(
            error_identifier=self.mock_error_identifier,
            initial_state_basis=StabilizerType.STABILIZER_Z,
            include_dynamical_decoupling_correction=False,
        )
        target_state = np.zeros(5, dtype=int)
        uncertainty = decoder.get_fidelity_uncertainty(cycle_stabilizer_count=1, target_state=target_state)

        # sqrt(0.5 * 0.5 / 100) = sqrt(0.25 / 100) = sqrt(0.0025) = 0.05
        self.assertAlmostEqual(uncertainty, 0.05)
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any leftover processes after testing."""
        pass
    # endregion
