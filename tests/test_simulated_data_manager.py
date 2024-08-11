import unittest
from typing import List
from qce_circuit.connectivity.intrf_channel_identifier import QubitIDObj, IQubitID
from qce_circuit.language.intrf_declarative_circuit import (
    InitialStateContainer,
    InitialStateEnum,
)
from qce_circuit.library.repetition_code.repetition_code_connectivity import Repetition9Round6Code as Repetition17Layer
from qce_circuit.addon_stim.noise_factory_manager import NoiseFactoryManager
from qce_interp.simulated_data_manager import SimulatedDataManager, NoiselessFactoryManager
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.visualization import (
    plot_all_defect_rate,
    plot_pij_matrix,
)
import matplotlib.pyplot as plt


class SimulatedDataManagerTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        rounds: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        involved_qubit_ids: List[IQubitID] = [
            QubitIDObj('D7'), QubitIDObj('Z3'), QubitIDObj('D4'), QubitIDObj('Z1'),
            QubitIDObj('D5'), QubitIDObj('Z4'), QubitIDObj('D6'), QubitIDObj('Z2'),
            QubitIDObj('D3'),
        ]
        initial_state: InitialStateContainer = InitialStateContainer.from_ordered_list([
            InitialStateEnum.ZERO,
            InitialStateEnum.ONE,
            InitialStateEnum.ZERO,
            InitialStateEnum.ONE,
            InitialStateEnum.ZERO,
        ])
        cls.manager_noiseless = SimulatedDataManager.from_simulated_repetition_code(
            qec_rounds=rounds,
            involved_qubit_ids=involved_qubit_ids,
            initial_state=initial_state,
            device_layout=Repetition17Layer(),
            noise_factory=NoiselessFactoryManager(),
        )
        cls.manager_noisy = SimulatedDataManager.from_simulated_repetition_code(
            qec_rounds=rounds,
            involved_qubit_ids=involved_qubit_ids,
            initial_state=initial_state,
            device_layout=Repetition17Layer(),
            noise_factory=NoiseFactoryManager(),
        )

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_calculate_defect_rates(self):
        """Tests default True."""
        error_identifier_noiseless: IErrorDetectionIdentifier = self.manager_noiseless.get_error_detection_classifier(
            use_heralded_post_selection=True,
            use_computational_parity=True,
        )
        error_identifier_noisy: IErrorDetectionIdentifier = self.manager_noisy.get_error_detection_classifier(
            use_heralded_post_selection=True,
            use_computational_parity=True,
        )
        plot_all_defect_rate(
            error_identifier_noiseless,
            included_rounds=self.manager_noiseless.qec_rounds[-1],
        )
        plot_all_defect_rate(
            error_identifier_noisy,
            included_rounds=self.manager_noisy.qec_rounds[-1],
        )
        self.assertTrue(True)

    def test_calculate_pij_matrix(self):
        """Tests default True."""
        error_identifier_noiseless: IErrorDetectionIdentifier = self.manager_noiseless.get_error_detection_classifier(
            use_heralded_post_selection=True,
            use_computational_parity=True,
        )
        error_identifier_noisy: IErrorDetectionIdentifier = self.manager_noisy.get_error_detection_classifier(
            use_heralded_post_selection=True,
            use_computational_parity=True,
        )
        plot_pij_matrix(
            error_identifier_noiseless,
            included_rounds=self.manager_noiseless.qec_rounds,
        )
        plot_pij_matrix(
            error_identifier_noisy,
            included_rounds=self.manager_noisy.qec_rounds,
        )
        self.assertTrue(True)
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        plt.close('all')
    # endregion
