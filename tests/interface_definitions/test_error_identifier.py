import unittest
import os
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from numpy.testing import assert_array_equal
from qce_interp.definitions import UNITDATA_DIR
from qce_interp.interface_definitions.intrf_channel_identifier import QubitIDObj
from qce_interp.interface_definitions.intrf_stabilizer_index_kernel import IStabilizerIndexingKernel
from qce_interp.data_manager import DataManager
from qce_interp.interface_definitions.intrf_state_classification import ParityType, StateClassifierContainer
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import ISyndromeDecoder
from qce_interp.decoder_examples.lookup_table import Distance5LookupTableDecoder
from qce_circuit.connectivity.connectivity_surface_code import Surface17Layer
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import RepetitionExperimentKernel


class DefectIdentifierTestCase(unittest.TestCase):

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
    def test_computational_parity_calculation(self):
        """Tests calculation logic behind computational parity."""
        # Expects input (N, X)
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [0, 1, 0, 1, 0]
            ])),
            np.asarray([
                [-1, -1, -1, -1]
            ])
        )
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [0, 0, 0, 0, 0]
            ])),
            np.asarray([
                [+1, +1, +1, +1]
            ])
        )
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [1, 1, 1, 1, 1]
            ])),
            np.asarray([
                [+1, +1, +1, +1]
            ])
        )
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [0, 0, 0, 1, 1]
            ])),
            np.asarray([
                [+1, +1, -1, +1]
            ])
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion


class ErrorDetectionIdentifierTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        file_path: Path = Path(os.path.join(UNITDATA_DIR, 'example_repetition_code_distance_5.hdf5'))
        cls.qec_test_rounds: int = 3
        cls.target_state: np.ndarray = np.asarray([0, 1, 0, 1, 0])
        cls.off_target_state: np.ndarray = np.asarray([1, 0, 1, 0, 1])
        cls.data_manager: DataManager = DataManager.from_file_path(
            file_path=file_path,
            qec_rounds=list(range(0, 8 + 1)),  # [1, 5, 10, 15],
            heralded_initialization=True,
            qutrit_calibration_points=True,
            involved_data_qubit_ids=[
                QubitIDObj('D7'),
                QubitIDObj('D4'),
                QubitIDObj('D5'),
                QubitIDObj('D6'),
                QubitIDObj('D3')
            ],
            involved_ancilla_qubit_ids=[
                QubitIDObj('Z3'),
                QubitIDObj('Z1'),
                QubitIDObj('Z4'),
                QubitIDObj('Z2')
            ],
            expected_parity_lookup={
                QubitIDObj('Z3'): ParityType.ODD,
                QubitIDObj('Z1'): ParityType.ODD,
                QubitIDObj('Z4'): ParityType.ODD,
                QubitIDObj('Z2'): ParityType.ODD,
            },
            device_layout=Surface17Layer(),
        )

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_error_identifier_output_array_shape(self):
        """Tests output array shapes of error identifier methods."""
        computed_parity: bool = False
        error_identifier: ErrorDetectionIdentifier = self.data_manager.get_error_detection_classifier(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
            use_computational_parity=computed_parity,
        )
        index_kernel: IStabilizerIndexingKernel = self.data_manager.index_kernel
        qec_rounds: int = self.qec_test_rounds
        # Number of experiment repetitions
        n: int = index_kernel.experiment_repetitions
        # Number of QEC-rounds
        m: int = qec_rounds
        # Ancilla qubit elements
        s: int = len(error_identifier.involved_stabilizer_qubit_ids)
        # Data qubit elements
        d: int = len(error_identifier.involved_data_qubit_ids)
        # All qubit elements
        p: int = len(error_identifier.involved_qubit_ids)
        # Number of post-selected experiment repetitions
        n_post_selected: int = int(np.sum(error_identifier.get_post_selection_mask(qec_rounds) == True))
        self.assertLessEqual(
            n_post_selected,
            n,
            msg="Expects number of post-selected data is less or equal to original dataset size."
        )
        self.assertEqual(
            error_identifier.get_binary_stabilizer_classification(qec_rounds).shape,
            (n_post_selected, m, s),
        )
        self.assertEqual(
            error_identifier.get_binary_heralded_classification(qec_rounds).shape,
            (n_post_selected, 1, p),
        )
        self.assertEqual(
            error_identifier.get_binary_projected_classification(qec_rounds).shape,
            (n_post_selected, 1, d),
        )
        self.assertEqual(
            error_identifier.get_parity_stabilizer_classification(qec_rounds).shape,
            (n_post_selected, m, s),
        )
        self.assertEqual(
            error_identifier.get_parity_computation_classification(qec_rounds).shape,
            (n_post_selected, 1, s) if computed_parity else (0,),
            msg="Expects same return shape (but empty) if computed parity is calculated or not."
        )
        self.assertEqual(
            error_identifier.get_defect_stabilizer_classification(qec_rounds).shape,
            (n_post_selected, m + int(computed_parity), s),
        )

    def test_syndrome_decoder_output_array_shape(self):
        """Tests output array shapes of syndrome decoder methods."""
        computed_parity: bool = True
        error_identifier: ErrorDetectionIdentifier = self.data_manager.get_error_detection_classifier(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
            use_computational_parity=computed_parity,
        )
        decoder: ISyndromeDecoder = Distance5LookupTableDecoder(
            error_identifier=error_identifier,
        )
        index_kernel: IStabilizerIndexingKernel = self.data_manager.index_kernel
        qec_rounds: int = self.qec_test_rounds
        # Number of experiment repetitions
        n: int = index_kernel.experiment_repetitions
        # Number of QEC-rounds
        m: int = qec_rounds
        # Ancilla qubit elements
        s: int = len(error_identifier.involved_stabilizer_qubit_ids)
        # Data qubit elements
        d: int = len(error_identifier.involved_data_qubit_ids)
        # All qubit elements
        p: int = len(error_identifier.involved_qubit_ids)
        # Number of post-selected experiment repetitions
        n_post_selected: int = int(np.sum(error_identifier.get_post_selection_mask(qec_rounds) == True))
        self.assertEqual(
            decoder.get_binary_syndrome_corrections(qec_rounds).shape,
            (n_post_selected, m + int(computed_parity), d),
        )
        self.assertEqual(
            decoder.get_binary_syndrome_correction(qec_rounds).shape,
            (n_post_selected, 1, d),
        )
        self.assertEqual(
            decoder.get_binary_projected_corrected(qec_rounds).shape,
            (n_post_selected, 1, d),
        )

    def test_empty_computed_parity_array(self):
        """Tests to make sure the computed parity array is empty if it is not supposed to be calculated."""
        qec_rounds: int = self.qec_test_rounds

        computed_parity: bool = False
        error_identifier: ErrorDetectionIdentifier = self.data_manager.get_error_detection_classifier(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
            use_computational_parity=computed_parity,
        )
        self.assertEqual(
            error_identifier.get_parity_computation_classification(qec_rounds).size,
            0,
            msg=f"Expects array size to be empty if {computed_parity} == False."
        )

        computed_parity: bool = True
        error_identifier: ErrorDetectionIdentifier = self.data_manager.get_error_detection_classifier(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
            use_computational_parity=computed_parity,
        )
        qec_rounds: int = self.qec_test_rounds
        self.assertNotEqual(
            error_identifier.get_parity_computation_classification(qec_rounds).size,
            0,
            msg=f"Expects array size not to be empty if {computed_parity} == True."
        )

    def test_correct_stabilization_to_target_state(self):
        """Tests if after including computed parity, the output states are successfully stabilized to logical subspace."""
        qec_rounds: int = self.qec_test_rounds
        computed_parity: bool = True
        error_identifier: ErrorDetectionIdentifier = self.data_manager.get_error_detection_classifier(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
            use_computational_parity=computed_parity,
        )
        decoder: ISyndromeDecoder = Distance5LookupTableDecoder(
            error_identifier=error_identifier,
        )
        # Number of post-selected experiment repetitions
        n_post_selected: int = int(np.sum(error_identifier.get_post_selection_mask(qec_rounds) == True))

        corrected_binary_data_qubit_outcomes: NDArray[np.int_] = decoder.get_binary_projected_corrected(qec_rounds)
        unique_binary_outcomes: NDArray[np.int_] = np.unique(corrected_binary_data_qubit_outcomes, axis=0)

        self.assertTrue(
            len(unique_binary_outcomes) == 2,
            msg="Expects one of two logical outcome states. Warning this might fail if error-correction is perfect!"
        )
        self.assertTrue(
            self.target_state in unique_binary_outcomes,
        )
        self.assertTrue(
            self.off_target_state in unique_binary_outcomes,
        )

        fraction_target_state = np.all(corrected_binary_data_qubit_outcomes == self.target_state, axis=2)
        fraction_off_target_state = np.all(corrected_binary_data_qubit_outcomes == self.off_target_state, axis=2)
        self.assertEqual(
            np.sum(fraction_target_state) + np.sum(fraction_off_target_state),
            n_post_selected,
            msg=f"Expects the total number of detected target- and off-target states to be equal to the number of post-selected experiment repetitions.  Instead {np.sum(fraction_target_state)} + {np.sum(fraction_off_target_state)} != {n_post_selected}"
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion


class GeneralComputedParityTestCase(unittest.TestCase):
    """
    Covers both weight-2 and weight-4 computed parity calculations
    """

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        weight_two_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
            },
            index_kernel=RepetitionExperimentKernel(

            ),
            involved_qubit_ids=,
            device_layout=Surface17Layer(),
            qec_rounds=[5],
            use_computational_parity=True,
        )

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_default(self):
        """Tests default True."""
        self.assertTrue(True)
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion