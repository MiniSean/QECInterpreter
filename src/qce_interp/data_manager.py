# -------------------------------------------
# Module containing functionality for formatting quantum error (detection/correction) experimental data.
# -------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Any, Optional
from enum import Enum, unique
from pathlib import Path
from qce_interp.utilities.custom_exceptions import InterfaceMethodException
from qce_interp.utilities.readwrite_hdf5 import (
    extract_data,
    HDF5DataExtractor,
    ExtractionSpec,
    SpecType,
)
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_data_manager import IDataManager
from qce_circuit.structure.acquisition_indexing.kernel_repetition_code import (
    RepetitionExperimentKernel,
    IStabilizerIndexingKernel,
)
from qce_interp.interface_definitions.intrf_state_classification import (
    StateKey,
    StateAcquisition,
    StateAcquisitionContainer,
    IStateClassifierContainer,
    ShotsClassifierContainer,
    ParityType,
)
from qce_interp.interface_definitions.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_interp.interface_definitions.intrf_error_identifier import (
    ErrorDetectionIdentifier,
)


@unique
class AcquisitionType(Enum):
    ONE_CHANNEL = "single_channel"
    TWO_CHANNEL = "double_channel"


class IAcquisitionChannelIdentifier(ABC):
    """
    Interface class, describing acquisition channel property.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def acquisition_type(self) -> AcquisitionType:
        """:return: Type of acquisition (1-channel or 2-channel)."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def channel_indices(self) -> List[int]:
        """:return: Array-like of channel indices."""
        raise InterfaceMethodException
    # endregion


@dataclass(frozen=True)
class SingleAcquisitionChannelIdentifier(IAcquisitionChannelIdentifier):
    """Data class, containing reference to qubit-ID, channel index."""
    qubit_id: IQubitID
    channel_index: int

    # region Interface Properties
    @property
    def acquisition_type(self) -> AcquisitionType:
        """:return: Type of acquisition (1-channel or 2-channel)."""
        return AcquisitionType.ONE_CHANNEL

    @property
    def channel_indices(self) -> List[int]:
        """:return: Array-like of channel indices."""
        return [self.channel_index]
    # endregion


@dataclass(frozen=True)
class AcquisitionChannelIdentifier(IAcquisitionChannelIdentifier):
    """Data class, containing reference to qubit-ID, I-channel index, Q-channel index."""
    qubit_id: IQubitID
    channel_index_i: int
    channel_index_q: int

    # region Interface Properties
    @property
    def acquisition_type(self) -> AcquisitionType:
        """:return: Type of acquisition (1-channel or 2-channel)."""
        return AcquisitionType.TWO_CHANNEL

    @property
    def channel_indices(self) -> List[int]:
        """:return: Array-like of channel indices."""
        return [self.channel_index_i, self.channel_index_q]
    # endregion


class DataManager(IDataManager):
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

    @classmethod
    def data_key(cls) -> str:
        return 'data_key'

    @classmethod
    def channel_name_key(cls) -> str:
        return 'value_names_key'
    # endregion

    # region Class Constructor
    def __init__(
            self,
            classifier_lookup: Dict[IQubitID, IStateClassifierContainer],
            calibration_point_lookup: Dict[IQubitID, StateAcquisitionContainer],
            experiment_index_kernel: IStabilizerIndexingKernel,
            involved_qubit_ids: List[IQubitID],
            device_layout: ISurfaceCodeLayer,
            qec_rounds: List[int],
    ) -> None:
        self._classifier_lookup: Dict[IQubitID, IStateClassifierContainer] = classifier_lookup
        self._calibration_point_lookup: Dict[IQubitID, StateAcquisitionContainer] = calibration_point_lookup
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

    def get_state_acquisition(self, qubit_id: IQubitID) -> Optional[StateAcquisitionContainer]:
        """:return: State acquisition based on qubit-ID. Returns None if qubit-ID is not supported."""
        if qubit_id not in self._calibration_point_lookup:
            return None
        return self._calibration_point_lookup[qubit_id]

    @classmethod
    def from_file_path(cls, file_path: Path, qec_rounds: List[int], heralded_initialization: bool, qutrit_calibration_points: bool, involved_data_qubit_ids: List[IQubitID], involved_ancilla_qubit_ids: List[IQubitID], expected_parity_lookup: Dict[IQubitID, ParityType], device_layout: ISurfaceCodeLayer) -> 'DataManager':
        """
        Constructs data extraction instructions.
        Opens hdf5 file using file_path and default HDF5DataExtractor strategy.
        Constructs experiment specific index kernel (This case repetition-code experiment).
        Iterate through each qubit, construct state-classifier based on raw shots, decision-boundaries and initial pauli-frame.
        :param file_path: Absolute file path to hdf5 file.
        :param qec_rounds: Array-like of integers representing qec-rounds per sub-experiment.
        :param heralded_initialization: Boolean, whether heralded initialization is used during this experiment.
        :param qutrit_calibration_points: Boolean, whether qutrit (readout) calibration points are used during this experiment.
        :param involved_data_qubit_ids: Array-like of qubit-ID's corresponding to data-qubits.
        :param involved_ancilla_qubit_ids: Array-like of qubit-ID's corresponding to ancilla-qubits.
        :param expected_parity_lookup: Dictionary mapping qubit-ID to ParityType. Used for constructing state-classifier.
        :param device_layout: Instance implementing ISurfaceCodeLayer interface, holding physical connectivity details.
            This is passed on to construct IErrorDetectionIdentifier. Not critical for DataManager itself.
        :return: New DataManager instance containing state-classifier lookup and experiment index kernel.
        """
        specs = [
            ExtractionSpec(path='Experimental Data/Data', spec_type=SpecType.DATASET, key=cls.data_key()),
            ExtractionSpec(path='Experimental Data', spec_type=SpecType.ATTRIBUTE, attr_name='value_names', key=cls.channel_name_key())
        ]
        # Extract data dictionary from HDF5 file
        data_dict: Dict[str, Any] = extract_data(file_path, specs, HDF5DataExtractor)

        # Construct datastructures
        involved_qubit_ids: List[IQubitID] = involved_data_qubit_ids + involved_ancilla_qubit_ids
        classifier_lookup: Dict[IQubitID, IStateClassifierContainer] = {}
        calibration_point_lookup: Dict[IQubitID, StateAcquisitionContainer] = {}
        channel_identifier_lookup: Dict[
            IQubitID, IAcquisitionChannelIdentifier] = DataManager.get_channel_identifier_lookup(
            channel_names=[name.decode() if isinstance(name, bytes) else name for name in data_dict[cls.channel_name_key()]],
            qubit_ids=involved_qubit_ids,
        )
        experiment_repetitions: int = RepetitionExperimentKernel.estimate_experiment_repetitions(
            rounds=qec_rounds,
            heralded_initialization=heralded_initialization,
            qutrit_calibration_points=qutrit_calibration_points,
            dataset_size=len(data_dict[cls.data_key()])
        )
        experiment_index_kernel: RepetitionExperimentKernel = RepetitionExperimentKernel(
            rounds=qec_rounds,
            heralded_initialization=heralded_initialization,
            qutrit_calibration_points=qutrit_calibration_points,
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=involved_ancilla_qubit_ids,
            experiment_repetitions=experiment_repetitions,
        )

        for qubit_id in tqdm(involved_qubit_ids, desc='Processing data file'):
            channel_identifier: IAcquisitionChannelIdentifier = channel_identifier_lookup[qubit_id]
            raw_shots: NDArray[np.float32] = data_dict[cls.data_key()][:, channel_identifier.channel_indices]
            raw_complex_shots: NDArray[np.complex64] = StateAcquisitionContainer.real_imag_to_complex(raw_shots)

            # Qutrit calibration points
            calibration_points: StateAcquisitionContainer = StateAcquisitionContainer.from_state_acquisitions(
                acquisitions=[
                    StateAcquisition(StateKey.STATE_0, raw_complex_shots[
                        experiment_index_kernel.get_projected_calibration_acquisition_indices(qubit_id, StateKey.STATE_0)
                    ]),
                    StateAcquisition(StateKey.STATE_1, raw_complex_shots[
                        experiment_index_kernel.get_projected_calibration_acquisition_indices(qubit_id, StateKey.STATE_1)
                    ]),
                    StateAcquisition(StateKey.STATE_2, raw_complex_shots[
                        experiment_index_kernel.get_projected_calibration_acquisition_indices(qubit_id, StateKey.STATE_2)
                    ]),
                ]
            )

            state_classification: ShotsClassifierContainer = ShotsClassifierContainer(
                shots=raw_complex_shots,
                decision_boundaries=calibration_points.decision_boundaries,
                _expected_parity=expected_parity_lookup.get(qubit_id, ParityType.EVEN),
            )

            classifier_lookup[qubit_id] = state_classification
            calibration_point_lookup[qubit_id] = calibration_points

        return DataManager(
            classifier_lookup=classifier_lookup,
            calibration_point_lookup=calibration_point_lookup,
            experiment_index_kernel=experiment_index_kernel,
            involved_qubit_ids=involved_qubit_ids,
            device_layout=device_layout,
            qec_rounds=qec_rounds,
        )
    # endregion

    # region Static Class Methods
    @staticmethod
    def get_channel_identifier_lookup(channel_names: List[str], qubit_ids: List[IQubitID]) -> Dict[IQubitID, IAcquisitionChannelIdentifier]:
        """

        :param channel_names:
        :param qubit_ids:
        :return:
        """
        # Data allocation
        result: Dict[IQubitID, IAcquisitionChannelIdentifier] = {}

        def get_channel_identifier_defined(channel_names: List[str], channel_identifier: str) -> bool:
            for i, channel_name in enumerate(channel_names):
                if channel_identifier in channel_name:
                    return True
            return False

        # Map qubit id to (UHF acquisition) channel index. (Which can map to channel name)
        def get_channel_index(channel_names: List[str], channel_identifier: str) -> int:
            for i, channel_name in enumerate(channel_names):
                if channel_identifier in channel_name:
                    return i + 1

        for qubit_id in qubit_ids:
            single_channel_defined: bool = get_channel_identifier_defined(channel_names=channel_names, channel_identifier=f"{qubit_id.id}")
            double_channel_defined: bool = (get_channel_identifier_defined(channel_names=channel_names, channel_identifier=f"{qubit_id.id} I")
                                            and get_channel_identifier_defined(channel_names=channel_names, channel_identifier=f"{qubit_id.id} Q"))

            if double_channel_defined:
                result[qubit_id] = AcquisitionChannelIdentifier(
                    qubit_id=qubit_id,
                    channel_index_i=get_channel_index(channel_names=channel_names, channel_identifier=f"{qubit_id.id} I"),
                    channel_index_q=get_channel_index(channel_names=channel_names, channel_identifier=f"{qubit_id.id} Q"),
                )
                continue
            if single_channel_defined:
                result[qubit_id] = SingleAcquisitionChannelIdentifier(
                    qubit_id=qubit_id,
                    channel_index=get_channel_index(channel_names=channel_names, channel_identifier=f"{qubit_id.id}"),
                )
                continue

        return result
    # endregion
