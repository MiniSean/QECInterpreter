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
from qce_interp.interface_definitions.intrf_state_classification import ParityType
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import ISyndromeDecoder
from qce_interp.decoder_examples.lookup_table import Distance5LookupTableDecoder
from qce_interp.utilities.connectivity_surface_code import Surface17Layer
from qce_interp.utilities.kernel_repetition_code import RepetitionExperimentKernel


class IndexingKernelPropertiesTestCase(unittest.TestCase):

    # region Setup
    @classmethod
    def setUpClass(cls) -> None:
        """Set up for all test cases"""
        rounds = list(range(1, 60 + 1))
        dataset_size = 10465920  # Experimental data
        experiment_repetitions: int = RepetitionExperimentKernel.estimate_experiment_repetitions(
            rounds=rounds,
            heralded_initialization=True,
            qutrit_calibration_points=True,
            dataset_size=dataset_size,
        )
        cls.index_kernel: RepetitionExperimentKernel = RepetitionExperimentKernel(
            rounds=rounds,
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
            experiment_repetitions=experiment_repetitions,
        )

    def setUp(self) -> None:
        """Set up for every test case"""
        pass
    # endregion

    # region Test Cases
    def test_bug_repeatedly_adding_calibration_kernel(self):
        """Tests default True."""
        repetition_kernel_length: int = len(self.index_kernel._repetition_kernels)
        calibration_kernel_length: int = len([self.index_kernel._calibration_kernel])

        self.assertEqual(
            len(self.index_kernel.indexing_kernels),
            repetition_kernel_length + calibration_kernel_length,
        )
        self.assertEqual(
            len(self.index_kernel.indexing_kernels),
            repetition_kernel_length + calibration_kernel_length,
            msg="Expected that second property call yields the same result."
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
