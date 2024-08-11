import unittest
from qce_circuit.language.intrf_declarative_circuit import InitialStateContainer, InitialStateEnum
from qce_circuit.connectivity.intrf_channel_identifier import QubitIDObj
from qce_interp.interface_definitions.intrf_state_classification import ParityType
from qce_interp.utilities.expected_parities import initial_state_to_expected_parity
from qce_circuit.library.repetition_code.repetition_code_connectivity import Repetition9Round6Code as Repetition17Layer
from qce_circuit.connectivity.connectivity_surface_code import Surface17Layer


class InitialStateToExpectedParityTestCase(unittest.TestCase):

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
    def test_weight_two_initial_state_conversion(self):
        """Tests initial state to weight-2 parities."""

        expected_parity = initial_state_to_expected_parity(
            initial_state=InitialStateContainer.from_ordered_list(
                initial_states=[
                    InitialStateEnum.ZERO,
                    InitialStateEnum.ONE,
                    InitialStateEnum.ONE,
                ]
            ),
            parity_layout=Repetition17Layer(),
            involved_data_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('D5'),
                QubitIDObj('D6'),
            ],
            involved_ancilla_qubit_ids=[
                QubitIDObj('Z1'),
                QubitIDObj('Z4'),
            ],
        )

        self.assertEqual(
            expected_parity[QubitIDObj('Z1')],
            ParityType.ODD,
        )
        self.assertEqual(
            expected_parity[QubitIDObj('Z4')],
            ParityType.EVEN,
        )

    def test_weight_four_initial_state_conversion(self):
        """Tests initial state to weight-4 and weight-2 parities."""

        expected_parity = initial_state_to_expected_parity(
            initial_state=InitialStateContainer.from_ordered_list(
                initial_states=[
                    InitialStateEnum.ZERO,
                    InitialStateEnum.ONE,
                    InitialStateEnum.ONE,
                    InitialStateEnum.ONE,
                ]
            ),
            parity_layout=Surface17Layer(),
            involved_data_qubit_ids=[
                QubitIDObj('D4'),
                QubitIDObj('D5'),
                QubitIDObj('D1'),
                QubitIDObj('D2'),
            ],
            involved_ancilla_qubit_ids=[
                QubitIDObj('Z1'),
                QubitIDObj('X1'),
            ],
        )

        self.assertEqual(
            expected_parity[QubitIDObj('Z1')],
            ParityType.ODD,
        )
        self.assertEqual(
            expected_parity[QubitIDObj('X1')],
            ParityType.EVEN,
        )
    # endregion

    # region Teardown
    @classmethod
    def tearDownClass(cls) -> None:
        """Closes any left over processes after testing"""
        pass
    # endregion
