import unittest
import os
from typing import List, Dict
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from numpy.testing import assert_array_equal
from qce_interp.definitions import UNITDATA_DIR
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID, QubitIDObj
from qce_interp.interface_definitions.intrf_stabilizer_index_kernel import IStabilizerIndexingKernel
from qce_interp.data_manager import DataManager
from qce_interp.interface_definitions.intrf_state_classification import ParityType, StateClassifierContainer
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import ISyndromeDecoder
from qce_interp.decoder_examples.lookup_table import Distance5LookupTableDecoder
from qce_circuit.connectivity.connectivity_surface_code import (
    Surface17Layer,
    IParityGroup,
    ParityGroup,
    ParityType as StabilizerType
)
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import RepetitionExperimentKernel
from qce_circuit.library.repetition_code.repetition_code_connectivity import Repetition9Round6Code as Repetition17Layer


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
        # Arbitrary ancilla IDs and mapping
        ancilla_ids: List[IQubitID] = [
            QubitIDObj('A'),
            QubitIDObj('B'),
            QubitIDObj('C'),
            QubitIDObj('D'),
        ]
        parity_index_lookup: Dict[IQubitID, NDArray[np.int_]] = {
            QubitIDObj('A'): np.asarray([0, 1]),
            QubitIDObj('B'): np.asarray([1, 2]),
            QubitIDObj('C'): np.asarray([2, 3]),
            QubitIDObj('D'): np.asarray([3, 4]),
        }

        # Expects input (N, X)
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [0, 1, 0, 1, 0],
            ]), parity_index_lookup=parity_index_lookup, involved_ancilla_qubit_ids=ancilla_ids),
            np.asarray([
                [-1, -1, -1, -1]
            ])
        )
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [0, 0, 0, 0, 0]
            ]), parity_index_lookup=parity_index_lookup, involved_ancilla_qubit_ids=ancilla_ids),
            np.asarray([
                [+1, +1, +1, +1]
            ])
        )
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [1, 1, 1, 1, 1]
            ]), parity_index_lookup=parity_index_lookup, involved_ancilla_qubit_ids=ancilla_ids),
            np.asarray([
                [+1, +1, +1, +1]
            ])
        )
        assert_array_equal(
            ErrorDetectionIdentifier.calculate_computational_parity(np.asarray([
                [0, 0, 0, 1, 1]
            ]), parity_index_lookup=parity_index_lookup, involved_ancilla_qubit_ids=ancilla_ids),
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
            device_layout=Repetition17Layer(),
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


class CustomSurfaceLayer(Surface17Layer):
    """
    Singleton class, overwrites Surface17Layer for testing purposes.
    """
    _parity_group_x: List[IParityGroup] = [
        ParityGroup(
            _parity_type=StabilizerType.STABILIZER_X,
            _ancilla_qubit=QubitIDObj('X1'),
            _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2')]
        ),
    ]
    _parity_group_z: List[IParityGroup] = [
        ParityGroup(
            _parity_type=StabilizerType.STABILIZER_Z,
            _ancilla_qubit=QubitIDObj('Z1'),
            _data_qubits=[QubitIDObj('D1'), QubitIDObj('D2'), QubitIDObj('D4'), QubitIDObj('D5')]
        ),
    ]


class GeneralComputedParityTestCase(unittest.TestCase):
    """
    Covers both weight-2 and weight-4 computed parity calculations
    """

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        cls.qec_round: int = 5
        cls.qec_rounds: List[int] = [cls.qec_round]
        cls.experiment_repetitions: int = 1
        cls.index_kernel_weight_two = RepetitionExperimentKernel(
            rounds=cls.qec_rounds,
            heralded_initialization=False,
            qutrit_calibration_points=False,
            involved_ancilla_qubit_ids=[
                QubitIDObj('Z1'),
            ],
            involved_data_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('D5'),
            ],
            experiment_repetitions=cls.experiment_repetitions,
        )
        cls.index_kernel_weight_two_double = RepetitionExperimentKernel(
            rounds=cls.qec_rounds,
            heralded_initialization=False,
            qutrit_calibration_points=False,
            involved_ancilla_qubit_ids=[
                QubitIDObj('Z1'),
                QubitIDObj('Z4'),
            ],
            involved_data_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('D5'),
                QubitIDObj('D6'),
            ],
            experiment_repetitions=cls.experiment_repetitions,
        )
        cls.index_kernel_weight_four = RepetitionExperimentKernel(
            rounds=cls.qec_rounds,
            heralded_initialization=False,
            qutrit_calibration_points=False,
            involved_ancilla_qubit_ids=[
                QubitIDObj('Z1'),
            ],
            involved_data_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('D5'),
                QubitIDObj('D1'),
                QubitIDObj('D2'),
            ],
            experiment_repetitions=cls.experiment_repetitions,
        )
        cls.surface_layout = Surface17Layer()
        cls.repetition_layout = Repetition17Layer()

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_single_weight_two_computed_parties(self):
        """Tests basic parity, computed-parity and defect of a weight two code."""
        weight_two_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D5'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
            },
            index_kernel=self.index_kernel_weight_two,
            involved_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('Z1'),
                QubitIDObj('D5'),
            ],
            device_layout=self.repetition_layout,
            qec_rounds=self.qec_rounds,
            use_computational_parity=True,
        )
        nr_ancilla: int = len(weight_two_error_identifier.involved_stabilizer_qubit_ids)

        computed_parity: np.ndarray = weight_two_error_identifier.get_parity_computation_classification(cycle_stabilizer_count=self.qec_round)
        parity_array: np.ndarray = weight_two_error_identifier.get_parity_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        defect_array: np.ndarray = weight_two_error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        self.assertEqual(
            computed_parity.shape,
            (self.experiment_repetitions, 1, nr_ancilla)
        )
        self.assertEqual(
            parity_array.shape,
            (self.experiment_repetitions, self.qec_round, nr_ancilla)
        )
        self.assertEqual(
            defect_array.shape,
            (self.experiment_repetitions, self.qec_round + 1, nr_ancilla)
        )
        assert_array_equal(
            computed_parity,
            np.asarray([[[1]]])
        )
        assert_array_equal(
            parity_array,
            np.asarray([[[1], [1], [1], [1], [1]]])
        )
        assert_array_equal(
            defect_array,
            np.asarray([[[1], [1], [1], [1], [1], [1]]])
        )

    def test_single_weight_two_computed_parties_variation(self):
        """Tests basic (error) parity, computed-parity and defect of a weight two code."""
        weight_two_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D5'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 1], dtype=np.int_),
                ),
            },
            index_kernel=self.index_kernel_weight_two,
            involved_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('Z1'),
                QubitIDObj('D5'),
            ],
            device_layout=self.repetition_layout,
            qec_rounds=self.qec_rounds,
            use_computational_parity=True,
        )
        nr_ancilla: int = len(weight_two_error_identifier.involved_stabilizer_qubit_ids)

        computed_parity: np.ndarray = weight_two_error_identifier.get_parity_computation_classification(cycle_stabilizer_count=self.qec_round)
        parity_array: np.ndarray = weight_two_error_identifier.get_parity_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        defect_array: np.ndarray = weight_two_error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        self.assertEqual(
            computed_parity.shape,
            (self.experiment_repetitions, 1, nr_ancilla)
        )
        self.assertEqual(
            parity_array.shape,
            (self.experiment_repetitions, self.qec_round, nr_ancilla)
        )
        self.assertEqual(
            defect_array.shape,
            (self.experiment_repetitions, self.qec_round + 1, nr_ancilla)
        )
        assert_array_equal(
            computed_parity,
            np.asarray([[[-1]]])
        )
        assert_array_equal(
            parity_array,
            np.asarray([[[1], [-1], [-1], [1], [1]]])
        )
        assert_array_equal(
            defect_array,
            np.asarray([[[1], [-1], [1], [-1], [1], [-1]]])
        )

    def test_double_weight_two_computed_parties(self):
        """Tests double parity, computed-parity and defect of a weight two code."""
        weight_two_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D5'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('Z4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D6'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
            },
            index_kernel=self.index_kernel_weight_two_double,
            involved_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('Z1'),
                QubitIDObj('D5'),
                QubitIDObj('Z4'),
                QubitIDObj('D6'),
            ],
            device_layout=self.repetition_layout,
            qec_rounds=self.qec_rounds,
            use_computational_parity=True,
        )
        nr_ancilla: int = len(weight_two_error_identifier.involved_stabilizer_qubit_ids)

        computed_parity: np.ndarray = weight_two_error_identifier.get_parity_computation_classification(cycle_stabilizer_count=self.qec_round)
        parity_array: np.ndarray = weight_two_error_identifier.get_parity_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        defect_array: np.ndarray = weight_two_error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        self.assertEqual(
            computed_parity.shape,
            (self.experiment_repetitions, 1, nr_ancilla)
        )
        self.assertEqual(
            parity_array.shape,
            (self.experiment_repetitions, self.qec_round, nr_ancilla)
        )
        self.assertEqual(
            defect_array.shape,
            (self.experiment_repetitions, self.qec_round + 1, nr_ancilla)
        )
        assert_array_equal(
            computed_parity,
            np.asarray([[[1, 1]]])
        )
        assert_array_equal(
            parity_array,
            np.asarray([[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]])
        )
        assert_array_equal(
            defect_array,
            np.asarray([[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]])
        )

    def test_parity_index_lookup(self):
        """Tests functionality for generating parity index lookup."""
        parity_layout = CustomSurfaceLayer()
        involved_ancilla_qubit_ids = [
            QubitIDObj('Z1'),
            QubitIDObj('X1'),
        ]
        involved_data_qubit_ids = [
            QubitIDObj('D5'),
            QubitIDObj('D2'),
            QubitIDObj('D1'),
            QubitIDObj('D6'),  # Extra entries
            QubitIDObj('D7'),  # Extra entries
            QubitIDObj('D4'),
        ]
        parity_index_lookup: Dict[IQubitID, NDArray[np.int_]] = ErrorDetectionIdentifier.get_parity_index_lookup(
            parity_layout=parity_layout,
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
        )
        expected_lookup: Dict[IQubitID, NDArray[np.int_]] = {
            QubitIDObj('X1'): np.asarray([2, 1], dtype=np.int_),
            QubitIDObj('Z1'): np.asarray([2, 1, 5, 0], dtype=np.int_),
        }

        # Check if both dictionaries have the same keys
        self.assertEqual(set(parity_index_lookup.keys()), set(expected_lookup.keys()))

        # Check if the values (NumPy arrays) for each key are equal
        for key in expected_lookup:
            assert_array_equal(parity_index_lookup[key], expected_lookup[key])

    def test_parity_index_lookup_missing_data_qubits(self):
        """Tests functionality for generating parity index lookup."""
        parity_layout = CustomSurfaceLayer()
        involved_ancilla_qubit_ids = [
            QubitIDObj('Z1'),  # Missing data qubit-IDs
            QubitIDObj('X1'),
        ]
        involved_data_qubit_ids = [
            QubitIDObj('D5'),
            QubitIDObj('D2'),
            QubitIDObj('D1'),
        ]
        parity_index_lookup: Dict[IQubitID, NDArray[np.int_]] = ErrorDetectionIdentifier.get_parity_index_lookup(
            parity_layout=parity_layout,
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
        )
        expected_lookup: Dict[IQubitID, NDArray[np.int_]] = {
            QubitIDObj('X1'): np.asarray([2, 1], dtype=np.int_),
        }

        # Check if both dictionaries have the same keys
        self.assertEqual(set(parity_index_lookup.keys()), set(expected_lookup.keys()))

        # Check if the values (NumPy arrays) for each key are equal
        for key in expected_lookup:
            assert_array_equal(parity_index_lookup[key], expected_lookup[key])

    def test_calculate_weight_two_computational_parity(self):
        """Tests staticmethod functionality for determining computational parity."""
        weight_two_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D5'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
            },
            index_kernel=self.index_kernel_weight_two,
            involved_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('Z1'),
                QubitIDObj('D5'),
            ],
            device_layout=self.repetition_layout,
            qec_rounds=self.qec_rounds,
            use_computational_parity=True,
        )
        nr_ancilla: int = len(weight_two_error_identifier.involved_stabilizer_qubit_ids)

        # (N, 1, D) Binary projected acquisition
        binary_projection: NDArray[np.int_] = weight_two_error_identifier.get_binary_projected_classification(cycle_stabilizer_count=self.qec_round)
        # (N, D) Pre-process
        n, one, d = binary_projection.shape
        binary_projection = binary_projection.reshape(n, d)

        computed_parity: NDArray[np.int_] = ErrorDetectionIdentifier.calculate_computational_parity_from_layout(
            array=binary_projection,
            parity_layout=self.repetition_layout,
            involved_data_qubit_ids=weight_two_error_identifier.involved_data_qubit_ids,
            involved_ancilla_qubit_ids=weight_two_error_identifier.involved_stabilizer_qubit_ids,
        )

        self.assertEqual(
            computed_parity.shape,
            (self.experiment_repetitions, nr_ancilla)
        )

        assert_array_equal(
            computed_parity,
            np.asarray([[+1]])
        )

    def test_calculate_weight_four_computational_parity(self):
        """Tests staticmethod functionality for determining computational parity."""
        weight_four_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('D5'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('D1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 1], dtype=np.int_),
                ),
                QubitIDObj('D2'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 1], dtype=np.int_),
                ),
            },
            index_kernel=self.index_kernel_weight_four,
            involved_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('Z1'),
                QubitIDObj('D5'),
                QubitIDObj('D1'),
                QubitIDObj('D2'),
            ],
            device_layout=self.surface_layout,
            qec_rounds=self.qec_rounds,
            use_computational_parity=True,
        )
        nr_ancilla: int = len(weight_four_error_identifier.involved_stabilizer_qubit_ids)

        # (N, 1, D) Binary projected acquisition
        binary_projection: NDArray[np.int_] = weight_four_error_identifier.get_binary_projected_classification(cycle_stabilizer_count=self.qec_round)
        # (N, D) Pre-process
        n, one, d = binary_projection.shape
        binary_projection = binary_projection.reshape(n, d)

        computed_parity: NDArray[np.int_] = ErrorDetectionIdentifier.calculate_computational_parity_from_layout(
            array=binary_projection,
            parity_layout=self.surface_layout,
            involved_data_qubit_ids=weight_four_error_identifier.involved_data_qubit_ids,
            involved_ancilla_qubit_ids=weight_four_error_identifier.involved_stabilizer_qubit_ids,
        )

        self.assertEqual(
            computed_parity.shape,
            (self.experiment_repetitions, nr_ancilla)
        )

        assert_array_equal(
            computed_parity,
            np.asarray([[+1]])
        )

    def test_single_weight_four_computed_parties(self):
        """Tests double parity, computed-parity and defect of a weight two code."""
        weight_four_error_identifier = ErrorDetectionIdentifier(
            classifier_lookup={
                QubitIDObj('Z1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 0, 0, 0, 0], dtype=np.int_),
                    _expected_parity=ParityType.EVEN,
                ),
                QubitIDObj('D4'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('D5'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 1], dtype=np.int_),
                ),
                QubitIDObj('D1'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 0], dtype=np.int_),
                ),
                QubitIDObj('D2'): StateClassifierContainer(
                    state_classification=np.asarray([0, 1, 0, 1, 1], dtype=np.int_),
                ),
            },
            index_kernel=self.index_kernel_weight_four,
            involved_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('Z1'),
                QubitIDObj('D5'),
                QubitIDObj('D1'),
                QubitIDObj('D2'),
            ],
            device_layout=self.surface_layout,
            qec_rounds=self.qec_rounds,
            use_computational_parity=True,
        )
        nr_ancilla: int = len(weight_four_error_identifier.involved_stabilizer_qubit_ids)

        # Computed parity assert
        computed_parity: np.ndarray = weight_four_error_identifier.get_parity_computation_classification(cycle_stabilizer_count=self.qec_round)
        self.assertEqual(
            nr_ancilla,
            1,
        )
        self.assertEqual(
            computed_parity.shape,
            (self.experiment_repetitions, 1, nr_ancilla)
        )
        assert_array_equal(
            computed_parity,
            np.asarray([[[1]]])
        )

        # Parity assert
        parity_array: np.ndarray = weight_four_error_identifier.get_parity_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        self.assertEqual(
            parity_array.shape,
            (self.experiment_repetitions, self.qec_round, nr_ancilla)
        )
        assert_array_equal(
            parity_array,
            np.asarray([[[1], [1], [1], [1], [1]]])
        )

        # Defect assert
        defect_array: np.ndarray = weight_four_error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=self.qec_round)
        self.assertEqual(
            defect_array.shape,
            (self.experiment_repetitions, self.qec_round + 1, nr_ancilla)
        )
        assert_array_equal(
            defect_array,
            np.asarray([[[1], [1], [1], [1], [1], [1]]])
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
