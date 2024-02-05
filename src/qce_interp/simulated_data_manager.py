# -------------------------------------------
# Module containing functionality for formatting quantum error (detection/correction) experimental data.
# -------------------------------------------
import numpy as np
from typing import List, Dict, Optional
import stim
from qce_circuit import (
    IDeclarativeCircuit,
    InitialStateContainer,
)
from qce_circuit.library.repetition_code_circuit import (
    construct_repetition_code_circuit,
)
from qce_circuit.addon_stim import to_stim
from qce_circuit.addon_stim.intrf_noise_factory import (
    IStimNoiseDresserFactory,
    StimNoiseDresserFactoryManager,
)
from qce_circuit.addon_stim.noise_factory_manager import (
    apply_noise,
    NoiseFactoryManager,
)
from qce_circuit.addon_stim.noise_settings_manager import IndexedNoiseSettings
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_data_manager import IDataManager
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import (
    RepetitionExperimentKernel,
    IStabilizerIndexingKernel,
)
from qce_interp.interface_definitions.intrf_state_classification import (
    IStateClassifierContainer,
    StateClassifierContainer,
    ParityType,
)
from qce_interp.interface_definitions.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_interp.interface_definitions.intrf_error_identifier import (
    ErrorDetectionIdentifier,
)


FILL_VALUE: float = np.nan


class NoiselessFactoryManager(IStimNoiseDresserFactory):
    """
    Behaviour class, implementing IStimNoiseDresserFactory as an identity or noise-less factory.
    """

    # region Interface Properties
    @property
    def supported_factories(self) -> List[str]:
        """:return: Array-like of supported factory types."""
        return []
    # endregion

    # region Interface Methods
    def construct(self, circuit: stim.Circuit, settings: IndexedNoiseSettings) -> stim.Circuit:
        """:return: Noise dressed Stim circuit."""
        return circuit

    def contains(self, factory_key: str) -> bool:
        """:return: Boolean, whether factory key is included in the manager."""
        return False
    # endregion


class SimulatedDataManager(IDataManager):
    """
    Behaviour class, constructs data entrypoints based on provided measurement data.
    Currently, a bit of a loaded class as it is responsible for constructing (Labeled)ErrorDetectionIdentifier
    and responsible for exposing StateClassifierContainer and StateAcquisitionContainer.
    """

    # region Class Properties
    @property
    def involved_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved qubit-ID's."""
        return self._involved_qubit_ids

    @property
    def involved_ancilla_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved ancilla qubit-ID's."""
        return [qubit_id for qubit_id in self.involved_qubit_ids if qubit_id in self._device_layout.ancilla_qubit_ids]

    @property
    def involved_data_qubit_ids(self) -> List[IQubitID]:
        """:return: Array-like of involved data qubit-ID's."""
        return [qubit_id for qubit_id in self.involved_qubit_ids if qubit_id in self._device_layout.data_qubit_ids]

    @property
    def qec_rounds(self) -> List[int]:
        """:return: Array-like of number of QEC-rounds per experiment."""
        return self._cycle_stabilizer_counts

    @property
    def index_kernel(self) -> IStabilizerIndexingKernel:
        """:return: Index kernel used for indexing data."""
        return self._experiment_index_kernel
    # endregion

    # region Class Constructor
    def __init__(
            self,
            classifier_lookup: Dict[IQubitID, IStateClassifierContainer],
            experiment_index_kernel: IStabilizerIndexingKernel,
            involved_qubit_ids: List[IQubitID],
            device_layout: ISurfaceCodeLayer,
            qec_rounds: List[int],
    ) -> None:
        self._classifier_lookup: Dict[IQubitID, IStateClassifierContainer] = classifier_lookup
        self._experiment_index_kernel: IStabilizerIndexingKernel = experiment_index_kernel
        self._involved_qubit_ids: List[IQubitID] = involved_qubit_ids
        self._device_layout: ISurfaceCodeLayer = device_layout
        self._cycle_stabilizer_counts: List[int] = qec_rounds
    # endregion

    # region Class Methods
    def get_error_detection_classifier(self, **kwargs) -> ErrorDetectionIdentifier:
        """
        :param kwargs: Additional keyword arguments passed to class constructor.
        :return: Instance that exposes high-level get-methods which can be used to construct error decoders, Pij-matrix, etc.
        """
        return ErrorDetectionIdentifier(
            classifier_lookup=self._classifier_lookup,
            index_kernel=self._experiment_index_kernel,
            involved_qubit_ids=self._involved_qubit_ids,
            device_layout=self._device_layout,
            qec_rounds=self.qec_rounds,
            **kwargs,
        )

    def get_state_classifier(self, qubit_id: IQubitID) -> Optional[IStateClassifierContainer]:
        """:return: State classifier based on qubit-ID. Returns None if qubit-ID is not supported."""
        if qubit_id not in self._classifier_lookup:
            return None
        return self._classifier_lookup[qubit_id]

    @classmethod
    def from_simulated_repetition_code(cls, qec_rounds: List[int], involved_qubit_ids: List[IQubitID], initial_state: InitialStateContainer, device_layout: ISurfaceCodeLayer, noise_factory: IStimNoiseDresserFactory = NoiselessFactoryManager()) -> 'SimulatedDataManager':
        """
        Constructs simulated data.
        Constructs experiment specific index kernel (This case repetition-code experiment).
        Iterate through each qubit, construct state-classifier based on simulated data and initial pauli-frame.
        :param qec_rounds: Array-like of integers representing qec-rounds per sub-experiment.
        :param involved_qubit_ids: Array-like of (ordered) qubit-ID's corresponding to repetition-code qubit string.
        :param initial_state: Array-like of initial state descriptors for each of the data qubits.
            (Assumes all ancilla qubits to start in 0)
        :param device_layout: Instance implementing ISurfaceCodeLayer interface, holding physical connectivity details.
            This is passed on to construct IErrorDetectionIdentifier. Not critical for DataManager itself.
        :return: New DataManager instance containing state-classifier lookup and experiment index kernel.
        """
        # Data allocation
        involved_data_qubit_ids = [qubit_id for qubit_id in involved_qubit_ids if qubit_id in device_layout.data_qubit_ids]
        involved_ancilla_qubit_ids = [qubit_id for qubit_id in involved_qubit_ids if qubit_id in device_layout.ancilla_qubit_ids]
        qubit_index_map: Dict[int, IQubitID] = {i: qubit_id for i, qubit_id in enumerate(involved_qubit_ids)}
        experiment_repetitions: int = 10000
        experiment_index_kernel: RepetitionExperimentKernel = RepetitionExperimentKernel(
            rounds=qec_rounds,
            heralded_initialization=True,
            qutrit_calibration_points=True,
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
            experiment_repetitions=experiment_repetitions,
        )
        expected_parity_lookup = SimulatedDataManager.initial_state_to_expected_parity(
            initial_state=initial_state,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
        )
        simulated_data: np.ndarray = SimulatedDataManager.construct_simulated_repetition_code_data(
            initial_state=initial_state,
            qec_cycles=qec_rounds,
            nr_ancilla_qubits=len(involved_ancilla_qubit_ids),
            nr_data_qubits=len(involved_data_qubit_ids),
            experiment_repetitions=experiment_repetitions,
            noise_factory=noise_factory,
            qubit_index_map=qubit_index_map,
        )
        assert experiment_index_kernel.kernel_cycle_length == simulated_data.shape[1], f"Simulated cycle length and index-kernel cycle length should be equal by definition. Instead {experiment_index_kernel.kernel_cycle_length} != {simulated_data.shape[1]}"

        classifier_lookup: Dict[IQubitID, IStateClassifierContainer] = {}
        for i, qubit_id in enumerate(involved_qubit_ids):
            classifier: StateClassifierContainer = StateClassifierContainer(
                state_classification=simulated_data[:, :, i].flatten(),
                _expected_parity=expected_parity_lookup.get(qubit_id, ParityType.EVEN)
            )
            classifier_lookup[qubit_id] = classifier

        return SimulatedDataManager(
            classifier_lookup=classifier_lookup,
            experiment_index_kernel=experiment_index_kernel,
            involved_qubit_ids=involved_qubit_ids,
            device_layout=device_layout,
            qec_rounds=qec_rounds,
        )
    # endregion

    # region Static Class Methods
    @staticmethod
    def initial_state_to_expected_parity(initial_state: InitialStateContainer, involved_ancilla_qubit_ids: List[IQubitID]) -> Dict[IQubitID, ParityType]:
        assert initial_state.distance == len(involved_ancilla_qubit_ids) + 1, f"Expects N number of initial states and N-1 number of ancilla's. instead {initial_state.distance} != {len(involved_ancilla_qubit_ids) - 1}."
        result: Dict[IQubitID, ParityType] = {}

        for i, qubit_id in enumerate(involved_ancilla_qubit_ids):
            state_a: int = initial_state.as_array[i]
            state_b: int = initial_state.as_array[i + 1]
            even_parity: bool = state_a == state_b
            if even_parity:
                result[qubit_id] = ParityType.EVEN
            else:
                result[qubit_id] = ParityType.ODD
        return result

    @staticmethod
    def _construct_simulated_data(stim_circuit: stim.Circuit, qec_cycle: int, nr_ancilla_qubits: int, nr_data_qubits: int, experiment_repetitions: int) -> np.ndarray:
        """
        Output shape: (N, Heralded + QEC-Rounds, P)
        (N, QEC-Rounds(+1), P)
        :return: Formatted array that mimics real experiment.
        """
        # Data allocation
        sampler = stim_circuit.compile_sampler()
        nr_total_qubits: int = nr_ancilla_qubits + nr_data_qubits
        heralded_init: bool = True
        # (N, (P*Heralded) + QEC-Rounds * M + S)
        samples: np.ndarray = sampler.sample(shots=experiment_repetitions)

        # (N, Heralded + QEC-Rounds + 3*Calibration, P)
        final_measurement: bool = True
        depth: int = max(heralded_init + final_measurement, heralded_init + qec_cycle)
        result: np.ndarray = np.ones(shape=(experiment_repetitions, depth, nr_total_qubits)) * FILL_VALUE
        # Populate heralded
        index_start: int = 0
        index_end: int = nr_total_qubits
        heralded_shift: int = nr_total_qubits
        if heralded_init:
            result[:, 0, :] = samples[:, index_start:index_end]  # In simulation, heralded measurements are always 0
        # Populate qec rounds
        index_start: int = heralded_shift
        index_end: int = (heralded_shift + qec_cycle * nr_ancilla_qubits)
        sub_samples: np.ndarray = samples[:, index_start:index_end]
        result[:, heralded_init:(heralded_init + qec_cycle), 1::2] = sub_samples.reshape(experiment_repetitions, qec_cycle, nr_ancilla_qubits)
        # Populate final measurements
        index_start: int = (heralded_shift + qec_cycle * nr_ancilla_qubits)
        index_end: int = (heralded_shift + qec_cycle * nr_ancilla_qubits + nr_data_qubits)
        sub_samples: np.ndarray = samples[:, index_start:index_end]
        result[:, (depth - 1):(depth), 0::2] = sub_samples.reshape(experiment_repetitions, 1, nr_data_qubits)
        return result

    @staticmethod
    def _construct_simulated_calibration_points(nr_ancilla_qubits: int, nr_data_qubits: int, experiment_repetitions: int) -> np.ndarray:
        """
        Output shape: (N, 3*Calibration, P)
        (N, 3, P)
        :return: Formatted array that mimics real experiment.
        """
        # Data allocation
        nr_total_qubits: int = nr_ancilla_qubits + nr_data_qubits
        heralded_init: bool = True
        qutrit_calibration: bool = True
        # (N, 3*(Heralded +Calibration), P)
        result: np.ndarray = np.ones(shape=(experiment_repetitions, 3 * (heralded_init + qutrit_calibration), nr_total_qubits)) * FILL_VALUE
        return result

    @staticmethod
    def construct_simulated_repetition_code_data(initial_state: InitialStateContainer, qec_cycles: List[int], nr_ancilla_qubits: int, nr_data_qubits: int, experiment_repetitions: int, noise_factory: IStimNoiseDresserFactory, qubit_index_map: Dict[int, IQubitID]) -> np.ndarray:
        """
        Output shape: (N, SUM(QEC-Rounds(+1)), P)
        :param initial_state: Data structure containing an array-like of initial state enum's. Corresponds to qubit order.
        :param qec_cycles: Array-like of integers describing the number of QEC cycle per sub-experiment.
        :param nr_ancilla_qubits: Integer number of ancilla qubits present in repetition code. Used for simulation.
        :param nr_data_qubits: Integer number of data qubits present in repetition code. Used for simulation.
        :param experiment_repetitions: Integer number of experiment repetitions. Repeats all QEC-cycles that many times.
        :param noise_factory: IStimNoiseDresserFactory instance that appends noise operations to stim.Circuit.
        :param qubit_index_map: Dictionary lookup that maps stim.Circuit qubit-index to qubit-ID.
            Used by noise settings manager to find the specific noise parameters for each qubit.
        :return: Formatted array that mimics real experiment.
        """
        result: List[np.ndarray] = []
        for qec_cycle in qec_cycles:
            circuit_with_detectors: IDeclarativeCircuit = construct_repetition_code_circuit(
                initial_state=initial_state,
                qec_cycles=qec_cycle,
            )
            stim_circuit: stim.Circuit = to_stim(circuit_with_detectors)
            noisy_stim_circuit: stim.Circuit = apply_noise(
                circuit=stim_circuit,
                qubit_index_map=qubit_index_map,
                factory=noise_factory,
            )
            result.append(SimulatedDataManager._construct_simulated_data(
                stim_circuit=noisy_stim_circuit,
                qec_cycle=qec_cycle,
                nr_ancilla_qubits=nr_ancilla_qubits,
                nr_data_qubits=nr_data_qubits,
                experiment_repetitions=experiment_repetitions
            ))
        result.append(SimulatedDataManager._construct_simulated_calibration_points(
            nr_ancilla_qubits=nr_ancilla_qubits,
            nr_data_qubits=nr_data_qubits,
            experiment_repetitions=experiment_repetitions
        ))
        return np.concatenate(result, axis=1)
    # endregion


if __name__ == '__main__':
    from qce_interp.interface_definitions.intrf_channel_identifier import QubitIDObj
    from qce_circuit import (
        IDeclarativeCircuit,
        InitialStateContainer,
        InitialStateEnum,
    )
    from qce_interp.utilities.connectivity_surface_code import Surface17Layer
    from qce_interp.visualization import plot_pij_matrix
    import matplotlib.pyplot as plt

    manager = SimulatedDataManager.from_simulated_repetition_code(
        qec_rounds=[1, 2, 3, 4, 5],
        involved_qubit_ids=[QubitIDObj('D7'), QubitIDObj('Z3'), QubitIDObj('D4'), QubitIDObj('Z1'), QubitIDObj('D5'), QubitIDObj('Z4'), QubitIDObj('D6'), QubitIDObj('Z2'), QubitIDObj('D3')],
        initial_state=InitialStateContainer.from_ordered_list([
            InitialStateEnum.ZERO,
            InitialStateEnum.ONE,
            InitialStateEnum.ZERO,
            InitialStateEnum.ONE,
            InitialStateEnum.ZERO,
        ]),
        device_layout=Surface17Layer(),
        noise_factory=NoiseFactoryManager(),
    )
    print(manager)
    plot_pij_matrix(
        error_identifier=manager.get_labeled_error_detection_classifier(
            use_heralded_post_selection=True,
            use_computational_parity=True,
        ),
        included_rounds=manager.qec_rounds,
    )
    plt.show()
