# -------------------------------------------
# Module describing implementation of Minimum Weight Perfect Matching decoders.
# Based on blossom algorithm.
# -------------------------------------------
import numpy as np
from numpy.typing import NDArray
import pymatching
import stim
from qce_circuit.addon_stim.intrf_noise_factory import IStimNoiseDresserFactory
from qce_circuit.addon_stim.noise_factory_manager import (
    NoiseFactoryManager,
    apply_noise,
)
from qce_circuit.addon_stim.intrf_stim_factory import IStimCircuitFactory
from qce_circuit.addon_stim import (
    StimFactoryManager,
    to_stim,
)
from qce_circuit.library.repetition_code.circuit_components import (
    IRepetitionCodeDescription,
)
from qce_circuit.library.repetition_code.circuit_constructors import construct_repetition_code_circuit
from qce_circuit.language import (
    IDeclarativeCircuit,
    InitialStateContainer,
)
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import IDecoder
from qce_interp.interface_definitions.intrf_state_classification import IStateClassifierContainer


class MWPMDecoder(IDecoder):
    """
    Behaviour class, implementing ILookupDecoder interfaces.
    Uses following convention:
    Output arrays are 3D tensors (N, M, P) where,
    - N is the number of measurement repetitions.
    - M is the number of stabilizer repetitions.
    - P is the number of qubit elements.
        Where:
        - S is the number of stabilizer qubits.
        - D is the number of data qubits.
    """

    # region Class Constructor
    def __init__(self, error_identifier: IErrorDetectionIdentifier, circuit_description: IRepetitionCodeDescription, initial_state_container: InitialStateContainer = InitialStateContainer.empty(), stim_factory: IStimCircuitFactory = StimFactoryManager(), noise_factory: IStimNoiseDresserFactory = NoiseFactoryManager()):
        self._error_identifier: IErrorDetectionIdentifier = error_identifier
        self._stim_factory: IStimCircuitFactory = stim_factory
        self._noise_factory: IStimNoiseDresserFactory = noise_factory
        self._initial_state_container: InitialStateContainer = initial_state_container
        self._circuit_description: IRepetitionCodeDescription = circuit_description
    # endregion

    # region Interface Methods
    def get_fidelity(self, cycle_stabilizer_count: int, target_state: np.ndarray) -> float:
        """
        Output shape: (1)
        - N is the number of measurement repetitions.
        - D is the number of data qubits.
        :return: Fidelity value of target state at specific cycle.
        """

        # (N, 1, D)
        binary_projected_classification: NDArray[np.int_] = self._error_identifier.get_binary_projected_classification(cycle_stabilizer_count=cycle_stabilizer_count)
        if cycle_stabilizer_count % 2 == 0 and cycle_stabilizer_count != 0:
            binary_projected_classification = binary_projected_classification ^ 1
        # (N, M(+1), S)
        syndromes: np.ndarray = IStateClassifierContainer.eigenvalue_to_binary(self._error_identifier.get_defect_stabilizer_classification(cycle_stabilizer_count=cycle_stabilizer_count))
        n, m, s = syndromes.shape
        # (N, M(+1) * S)
        syndromes = np.reshape(syndromes, (n, s * m))
        # MWPM
        matching_decoder: pymatching.Matching = self.get_decoder(cycle_stabilizer_count=cycle_stabilizer_count)

        # Logical init
        logical_init = np.sum(target_state) % 2
        # Logical outcome
        logical_outcome = np.sum(binary_projected_classification, axis=2) % 2
        # Logical correction
        logical_correction = matching_decoder.decode_batch(syndromes).astype(int)
        # Logical corrected
        logical_corrected = logical_outcome ^ logical_correction
        # Logical error
        logical_error = logical_init ^ logical_corrected

        # (N, D)
        equal_rows_count = np.sum(np.all(logical_error == 0, axis=1))
        equal_fraction: float = equal_rows_count / len(logical_error)
        return equal_fraction
    # endregion

    # region Class Methods
    def get_decoder(self, cycle_stabilizer_count: int) -> pymatching.Matching:
        # Construct circuit from definition
        circuit: IDeclarativeCircuit = construct_repetition_code_circuit(
            qec_cycles=cycle_stabilizer_count,
            description=self._circuit_description,
            initial_state=self._initial_state_container,
        )
        stim_circuit: stim.Circuit = to_stim(circuit=circuit, factory=self._stim_factory)
        noisy_circuit = apply_noise(circuit=stim_circuit, factory=self._noise_factory)

        detector_error_model = noisy_circuit.detector_error_model(
            decompose_errors=True,
            allow_gauge_detectors=False,
            approximate_disjoint_errors=True,
        )
        matching_object = pymatching.Matching(
            graph=detector_error_model,
        )
        return matching_object
    # endregion
