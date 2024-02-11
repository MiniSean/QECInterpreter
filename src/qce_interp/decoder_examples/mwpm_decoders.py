# -------------------------------------------
# Module describing implementation of Minimum Weight Perfect Matching decoders.
# Based on blossom algorithm.
# -------------------------------------------
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import numpy as np
from numpy.typing import NDArray
import pymatching
import stim
import cma
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID, QubitIDObj
from qce_circuit.addon_stim.noise_settings_manager import QubitNoiseModelParameters, IndexedNoiseSettings
from qce_circuit.addon_stim.intrf_noise_factory import IStimNoiseDresserFactory
from qce_circuit.addon_stim.noise_factory_manager import (
    NoiseFactoryManager,
    apply_noise,
)
from qce_circuit.addon_stim.intrf_stim_factory import IStimCircuitFactory
from qce_circuit.addon_stim import (
    StimFactoryManager,
    to_stim,
    NoiseSettingManager,
    NoiseSettings,
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


@dataclass(frozen=True)
class CircuitNoiseParameters:
    """
    Data class, containing QubitNoiseModelParameters.
    """
    qubit_parameters: List[QubitNoiseModelParameters]

    # region Class Properties
    @classmethod
    def argument_count(cls) -> int:
        return 3   # t1, t2 and assignment error

    @property
    def to_array(self) -> np.ndarray:
        """:return: Parameters formatted as numpy array."""
        result: List[float] = []
        for parameters in self.qubit_parameters:
            result.extend([
                self.normalize_log_scale(parameters.t1),
                self.normalize_log_scale(parameters.t2),
                parameters.assignment_error,
            ])
        return np.asarray(result)

    @property
    def bounds_tuple(self) -> Tuple[List[float], List[float]]:
        """:return: Tuple of lower- and upper-bound values."""
        lower_bound: List[float] = []
        upper_bound: List[float] = []
        for _ in self.qubit_parameters:
            lower_bound.extend([
                0,
                0,
                0,
            ])
            upper_bound.extend([
                1.0,
                1.0,
                1.0,
            ])
        return lower_bound, upper_bound
    # endregion

    # region Class Methods
    @classmethod
    def from_array(cls, v: np.ndarray) -> 'CircuitNoiseParameters':
        # Data allocation
        result: List[QubitNoiseModelParameters] = []
        argument_length: int = cls.argument_count()
        n: int = v.size // argument_length
        for row in v.reshape((n, argument_length)):
            result.append(
                QubitNoiseModelParameters(
                    t1=cls.inverse_normalize_log_scale(row[0]),
                    t2=cls.inverse_normalize_log_scale(row[1]),
                    assignment_error=row[2],
                )
            )
        return CircuitNoiseParameters(
            qubit_parameters=result,
        )

    @classmethod
    def from_noise_settings(cls, settings: NoiseSettings, involved_qubit_ids: List[IQubitID]) -> 'CircuitNoiseParameters':
        return CircuitNoiseParameters(
            qubit_parameters=[settings.get_noise_settings(qubit_id=qubit_id) for qubit_id in involved_qubit_ids]
        )

    def apply_to_noise_settings(self, settings: NoiseSettings, involved_qubit_ids: List[IQubitID]) -> NoiseSettings:
        # Overwrite individual noise parameters
        individual_noise: Dict[IQubitID, QubitNoiseModelParameters] = settings.individual_noise
        individual_noise.update({
            qubit_id: parameters
            for qubit_id, parameters in zip(involved_qubit_ids, self.qubit_parameters)
        })

        return NoiseSettings(
            default_t1=settings.default_t1,
            default_t2=settings.default_t2,
            default_assignment_error=settings.default_assignment_error,
            default_single_qubit_gate_error=settings.default_single_qubit_gate_error,
            individual_noise=individual_noise,
            operation_durations=settings.operation_durations,
        )
    # endregion

    # region Static Class Methods
    @staticmethod
    def normalize_log_scale(x: float, max_value: float = 1e-3, offset: float = 1e-6) -> float:
        """
        Normalize a number between 0 and max_value to a range [0, 1] using logarithmic scaling.

        :param x: The number to normalize (assumed to be in the range [0, max_value]).
        :param max_value: The maximum value in the original range.
        :param offset: A small positive value to avoid logarithm of zero.

        :return: The normalized value between [0, 1].
        """
        # Apply logarithmic transformation and normalization
        normalized_x = (np.log(x + offset) - np.log(offset)) / (np.log(max_value + offset) - np.log(offset))

        return normalized_x

    @staticmethod
    def inverse_normalize_log_scale(y: float, max_value: float = 1e-3, offset: float = 1e-6) -> float:
        """
        Convert a normalized value in the range [0, 1] back to its original scale using logarithmic scaling.

        :param y: The normalized value to convert (in the range [0, 1]).
        :param max_value: The maximum value in the original range.
        :param offset: The small positive offset used to avoid the logarithm of zero.

        :return: The original value before normalization.
        """
        # Calculate the original value from the normalized value
        original_x = np.exp(y * (np.log(max_value + offset) - np.log(offset)) + np.log(offset)) - offset

        return original_x
    # endregion


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
    def __init__(
            self,
            error_identifier: IErrorDetectionIdentifier,
            circuit_description: IRepetitionCodeDescription,
            initial_state_container: InitialStateContainer = InitialStateContainer.empty(),
            stim_factory: IStimCircuitFactory = StimFactoryManager(),
            noise_factory: IStimNoiseDresserFactory = NoiseFactoryManager(),
            noise_settings: NoiseSettings = NoiseSettingManager.read_config(),
    ):
        self._error_identifier: IErrorDetectionIdentifier = error_identifier
        self._stim_factory: IStimCircuitFactory = stim_factory
        self._noise_factory: IStimNoiseDresserFactory = noise_factory
        self._noise_settings: NoiseSettings = noise_settings
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
        noisy_circuit = apply_noise(
            circuit=stim_circuit,
            qubit_index_map=self._circuit_description.circuit_channel_map,
            factory=self._noise_factory,
            noise_settings=self._noise_settings,
        )

        detector_error_model = noisy_circuit.detector_error_model(
            decompose_errors=True,
            allow_gauge_detectors=False,
            approximate_disjoint_errors=True,
        )
        matching_object = pymatching.Matching(
            graph=detector_error_model,
        )
        return matching_object

    def optimize_noise_model(self, at_qec_rounds: List[int], noise_settings: NoiseSettings = NoiseSettingManager.read_config(), max_iter: int = 100) -> Tuple[NoiseSettings, 'MWPMDecoder']:
        """:return: Optimized NoiseSettings to get the highest logical fidelity at chosen QEC-rounds."""
        involved_qubit_ids: List[IQubitID] = self._circuit_description.qubit_ids
        noise_parameters: CircuitNoiseParameters = CircuitNoiseParameters.from_noise_settings(
            settings=noise_settings,
            involved_qubit_ids=involved_qubit_ids
        )

        def get_fitness_at_round(qec_cycle: int, noise_parameters: NoiseSettings) -> float:
            _decoder = MWPMDecoder(
                error_identifier=self._error_identifier,
                circuit_description=self._circuit_description,
                initial_state_container=self._initial_state_container,
                stim_factory=self._stim_factory,
                noise_factory=self._noise_factory,
                noise_settings=noise_parameters,  # To optimize
            )
            fidelity: float = _decoder.get_fidelity(
                cycle_stabilizer_count=qec_cycle,
                target_state=self._initial_state_container.as_array,
            )
            return -fidelity

        def multi_round_fitness_function(v: np.ndarray) -> float:
            # Add noise
            _noise_parameters: CircuitNoiseParameters = CircuitNoiseParameters.from_array(v)
            _noise_settings: NoiseSettings = _noise_parameters.apply_to_noise_settings(
                settings=noise_settings,
                involved_qubit_ids=involved_qubit_ids,
            )
            rewards: List[float] = []
            for qec_cycle in at_qec_rounds:
                rewards.append(
                    get_fitness_at_round(
                        qec_cycle=qec_cycle,
                        noise_parameters=_noise_settings,
                    )
                )
            return float(np.mean(rewards))

        def optimize_with_cma(initial_parameters: CircuitNoiseParameters, fitness_function: Callable[[np.ndarray], float], max_iter: int) -> Tuple[np.ndarray, float]:
            # Define the lower and upper bounds for each dimension
            lower_bounds, upper_bounds = initial_parameters.bounds_tuple
            bounds = [lower_bounds, upper_bounds]
            initial_vector: np.ndarray = initial_parameters.to_array

            initial_stepsize: float = 0.25  # (works good)
            es = cma.CMAEvolutionStrategy(initial_vector, initial_stepsize, {'maxiter': max_iter, 'bounds': bounds})
            while not es.stop():
                solutions = es.ask()
                es.tell(solutions, [fitness_function(s) for s in solutions])
                es.logger.add()  # Add a data point to the log
                es.disp()  # Print current iteration data

            result = es.result
            return result.xbest, result.fbest

        # CMA optimize
        optimized_vector, optimized_fitness = optimize_with_cma(
            initial_parameters=noise_parameters,
            fitness_function=multi_round_fitness_function,
            max_iter=max_iter,
        )
        _noise_parameters: CircuitNoiseParameters = CircuitNoiseParameters.from_array(optimized_vector)
        optimized_parameters: NoiseSettings = _noise_parameters.apply_to_noise_settings(
            settings=noise_settings,
            involved_qubit_ids=involved_qubit_ids,
        )
        optimized_decoder: MWPMDecoder = MWPMDecoder(
            error_identifier=self._error_identifier,
            circuit_description=self._circuit_description,
            initial_state_container=self._initial_state_container,
            stim_factory=self._stim_factory,
            noise_factory=self._noise_factory,
            noise_settings=optimized_parameters,
        )

        return optimized_parameters, optimized_decoder
    # endregion
