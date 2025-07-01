# -------------------------------------------
# Module containing interface and implementation of state discrimination and classification.
# -------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
import itertools
import numpy as np
from warnings import warn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy.typing import NDArray
from typing import List, Dict, Optional, Callable, TypeVar, Sequence
from qce_circuit.utilities.array_manipulation import unique_in_order
from qce_interp.utilities.custom_exceptions import (
    InterfaceMethodException,
    ZeroClassifierShotsException,
)
from qce_interp.utilities.geometric_definitions import Vec2D
from qce_circuit.structure.acquisition_indexing.intrf_stabilizer_index_kernel import StateKey


# Acquisition (state and boundary) dataclasses
@unique
class ParityType(Enum):
    EVEN = +1
    ODD = -1


@dataclass(frozen=True)
class StateAcquisition:
    """Data class, containing (complex) acquisition information, together with a state key."""
    state: StateKey
    shots: NDArray[np.complex64]

    # region Class Properties
    @property
    def center(self) -> Vec2D:
        """:return: Mean center."""
        complex_mean: complex = complex(np.mean(self.shots))
        return Vec2D(
            x=complex_mean.real,
            y=complex_mean.imag,
        )
    # endregion


@dataclass(frozen=True)
class StateBoundaryKey:
    """Data class, containing un-ordered state-keys."""
    state_a: StateKey
    state_b: StateKey

    # region Class Methods
    def __contains__(self, item):
        if isinstance(item, StateKey):
            return item in [self.state_a, self.state_b]
        return False

    def __hash__(self):
        """
        Sorts individual state hashes such that the order is NOT maintained.
        Making hash comparison independent of order.
        """
        return hash((min(self.state_a.__hash__(), self.state_b.__hash__()),
                     max(self.state_a.__hash__(), self.state_b.__hash__())))

    def __eq__(self, other):
        if isinstance(other, StateBoundaryKey):
            # State boundary is equal if they share the same state keys, order does not matter
            return self.__hash__() == other.__hash__()
        return False

    def __repr__(self):
        return f'<BoundaryKey>{self.state_a}-{self.state_b}'
    # endregion


@dataclass(frozen=True)
class DirectedStateBoundaryKey(StateBoundaryKey):
    """Data class, containing ordered state-keys."""

    # region Class Methods
    def __hash__(self):
        """
        Sorts individual state hashes such that the order IS maintained.
        Making hash comparison dependent of order.
        """
        return hash((self.state_a.__hash__(), self.state_b.__hash__()))
    # endregion


@dataclass(frozen=True)
class DecisionBoundaries:
    """Data class, containing decision boundaries based on states."""
    boundary_lookup: Dict[StateBoundaryKey, Vec2D]
    _discriminator: LinearDiscriminantAnalysis
    _state_lookup: Dict[StateKey, int]
    """Lookup dictionary that maps state key to discriminator prediction index."""
    _mean: Optional[Vec2D] = field(default=None)
    """Explicit specification of boundary means, necessary when handling 2-state classification."""

    # region Class Properties
    @property
    def mean(self) -> Vec2D:
        """:return: Mean IQ-vector based on state boundaries."""
        if self._mean is not None:
            return self._mean
        boundary_points: np.ndarray = np.asarray([point.to_vector() for point in self.boundary_lookup.values()])
        mean_point: np.ndarray = np.mean(boundary_points, axis=0)
        return Vec2D.from_vector(mean_point)

    @property
    def state_prediction_index_lookup(self) -> Dict[StateKey, int]:
        """Lookup dictionary that maps state key to discriminator prediction index."""
        return self._state_lookup

    @property
    def prediction_index_to_state_lookup(self) -> Dict[int, StateKey]:
        """:return: Lookup dictionary that maps discriminator prediction index to state key."""
        return {index: state for state, index in self.state_prediction_index_lookup.items()}

    # endregion

    # region Class Methods
    def get_boundary(self, key: StateBoundaryKey) -> Optional[Vec2D]:
        """
        :return: Boundary point (2D) between state A and B.
        If state A == B or if state-boundary is not known, return None.
        """
        return self.get_boundary_between(key.state_a, key.state_b)

    def get_boundary_between(self, state_a: StateKey, state_b: StateKey) -> Optional[Vec2D]:
        """
        :return: Boundary point (2D) between state A and B.
        If state A == B or if state-boundary is not known, return None.
        """
        # Guard clause, if state A and B are equal, return None
        if state_a == state_b:
            return None
        boundary_key: StateBoundaryKey = StateBoundaryKey(state_a=state_a, state_b=state_b)
        if boundary_key not in self.boundary_lookup:
            return None
        return self.boundary_lookup[boundary_key]

    def get_binary_predictions(self, shots: NDArray[np.complex64]) -> NDArray[np.int_]:
        """
        NOTE: Forces classification of element in group 1 or 2, disregarding other groups.
        NOTE: Returns integer prediction value, can be mapped to state-enum using self.prediction_index_to_state_lookup.
        :return: Array-like of State key predictions based on shots discrimination.
        """
        shot_reshaped: NDArray[np.float32] = StateAcquisitionContainer.complex_to_real_imag(shots)
        # Step 1: Predict probabilities
        probabilities: np.ndarray = self._discriminator.predict_proba(shot_reshaped)
        # Step 2: Compare probabilities for groups 1 and 2
        # Assuming classes are labeled as 0, 1, 2 for groups 1, 2, 3 respectively
        prob_group_1: np.ndarray = probabilities[:, 0]
        prob_group_2: np.ndarray = probabilities[:, 1]
        # Step 3: Classify based on higher probability
        # Assign to group 1 if prob_group_1 > prob_group_2, else assign to group 2
        state_indices: NDArray[np.int_] = np.where(prob_group_1 > prob_group_2, 0, 1)  # 0 for group 1, 1 for group 2
        return state_indices

    def get_predictions(self, shots: NDArray[np.complex64]) -> NDArray[np.int_]:
        """
        NOTE: Returns integer prediction value, can be mapped to state-enum using self.prediction_index_to_state_lookup.
        :return: Array-like of State key predictions based on shots discrimination.
        """
        shot_reshaped: NDArray[np.float32] = StateAcquisitionContainer.complex_to_real_imag(shots)
        state_indices: NDArray[np.int_] = self._discriminator.predict(shot_reshaped)  # 1 indexed
        return state_indices

    def get_prediction(self, shot: np.complex64) -> StateKey:
        """:return: State key prediction based on shot discrimination."""
        state_indices: NDArray[np.int_] = self.get_predictions(shots=np.asarray([shot]))
        int_to_enum = self.prediction_index_to_state_lookup
        return int_to_enum[state_indices[0]]

    def get_fidelity(self, shots: NDArray[np.complex64], assigned_state: StateKey) -> float:
        """:return: Assignment fidelity defined as the probability of shots being part of assigned state."""
        shots_reshaped: NDArray[np.float32] = StateAcquisitionContainer.complex_to_real_imag(shots)
        state_indices: NDArray[np.int_] = self._discriminator.predict(shots_reshaped)  # 1 indexed
        return float(np.mean(state_indices == self._state_lookup[assigned_state]))

    def post_select_on(self, shots_to_filter: NDArray[np.complex64], conditional_shots: NDArray[np.complex64], conditional_state: StateKey) -> NDArray[np.complex64]:
        """:return: Filtered shots based on conditional shots (of same length) and conditional state."""
        # Guard clause, if conditional shots are empty, return shots without filtering
        if len(conditional_shots) == 0:
            return shots_to_filter

        conditional_shots_reshaped: NDArray[np.float32] = StateAcquisitionContainer.complex_to_real_imag(
            conditional_shots)
        state_indices: NDArray[np.int_] = self._discriminator.predict(conditional_shots_reshaped)  # 1 indexed
        conditional_index: int = self._state_lookup[conditional_state]
        mask: NDArray[np.int_] = np.array(
            [1 if state_index == conditional_index else np.nan for state_index in state_indices])
        return shots_to_filter[~np.isnan(mask)]

    @classmethod
    def from_acquisition_container(cls, container: 'StateAcquisitionContainer') -> 'DecisionBoundaries':
        """
        Calculates decision boundary points for a LinearDiscriminantAnalysis model.
        Assumes linear boundaries, working only with LDA classifiers.
        """
        # Data allocation
        discriminator = LinearDiscriminantAnalysis()
        concatenated_shots = container.concatenated_shots
        concatenated_shots_reshaped: np.ndarray = StateAcquisitionContainer.complex_to_real_imag(concatenated_shots)
        labels = container.concatenated_labels
        # Run linear discriminator fit
        discriminator.fit(
            concatenated_shots_reshaped,
            labels,
        )

        # Handling different number of classes
        num_classes = len(container.state_acquisition_lookup)
        # In multi-class classification, use the provided coefficients and intercepts
        coef_values = discriminator.coef_
        intercept_values = discriminator.intercept_

        # Map coefficients
        coef_lookup: Dict[StateKey, Vec2D] = {state: Vec2D.from_vector(value) for state, value in
                                              zip(container.state_acquisition_lookup.keys(), coef_values)}
        intercept_lookup: Dict[StateKey, float] = {state: value for state, value in
                                                   zip(container.state_acquisition_lookup.keys(), intercept_values)}
        state_lookup: Dict[StateKey, int] = {state: state.value for state in container.state_acquisition_lookup.keys()}

        # Create an iterator for all unique combinations of StateKey values, excluding same-key pairs
        intersection_lookup: Dict[StateBoundaryKey, Vec2D] = {}
        states: List[StateKey] = list(container.state_acquisition_lookup.keys())
        if num_classes == 2:
            state_a = states[0]
            state_b = states[1]
            boundary_key: StateBoundaryKey = StateBoundaryKey(state_a=state_a, state_b=state_b)
            intersection_lookup[boundary_key] = DecisionBoundaries._calculate_intersection_binary_case(
                coef1=coef_lookup[state_a],
                intercept1=intercept_lookup[state_a],
            )
            center: Vec2D = 0.5 * (container.state_acquisition_lookup[state_a].center + container.state_acquisition_lookup[state_b].center)
            return DecisionBoundaries(
                boundary_lookup=intersection_lookup,
                _discriminator=discriminator,
                _state_lookup=state_lookup,
                _mean=center,
            )

        for state_a, state_b in itertools.combinations(states, 2):
            boundary_key: StateBoundaryKey = StateBoundaryKey(state_a=state_a, state_b=state_b)
            intersection_lookup[boundary_key] = DecisionBoundaries._calculate_intersection(
                coef1=coef_lookup[state_a],
                intercept1=intercept_lookup[state_a],
                coef2=coef_lookup[state_b],
                intercept2=intercept_lookup[state_b],
            )
        return DecisionBoundaries(
            boundary_lookup=intersection_lookup,
            _discriminator=discriminator,
            _state_lookup=state_lookup,
        )

    # endregion

    # region Static Class Methods
    @staticmethod
    def _calculate_intersection(coef1: Vec2D, intercept1: float, coef2: Vec2D, intercept2: float) -> Vec2D:
        """
        :return: Intersection point of two linear equations defined by coefficients and intercepts.
        """
        denominator: float = (-coef1.x / coef1.y + coef2.x / coef2.y)
        numerator: float = (-intercept2 / coef2.y + intercept1 / coef1.y)
        # Deal with possible zero-division error
        if denominator != 0:
            _x: float = numerator / denominator
        elif denominator == 0 and numerator == 0:
            _x: float = 1.0
        else:
            warn(f"[ZeroDivisionError] During intersect calculation of {coef1} and {coef2}.")
            denominator = 1e-6
            _x: float = numerator / denominator

        _y: float = -coef1.x / coef1.y * _x - intercept1 / coef1.y
        return Vec2D(
            x=_x,
            y=_y,
        )

    @staticmethod
    def _calculate_intersection_binary_case(coef1: Vec2D, intercept1: float):
        """
        :return: Intersection point of single linear equations defined by coefficients and intercepts.
        """
        x_intercept = -intercept1 / coef1.x if coef1.x != 0 else np.inf
        y_intercept = -intercept1 / coef1.y if coef1.y != 0 else np.inf
        return Vec2D(x=x_intercept, y=y_intercept)
    # endregion


class IStateAcquisitionContainer(ABC):
    """
    Interface class, describing state acquisition and classification for state 0, 1 (and 2).
    """

    # region Interface Properties
    @property
    @abstractmethod
    def contained_states(self) -> List[StateKey]:
        """:return: Array-like of unique contained states."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def classification_boundaries(self) -> DecisionBoundaries:
        """:return: DecisionBoundaries."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def concatenated_shots(self) -> NDArray[np.complex64]:
        """:return: Array-like of (complex-valued) concatenated acquisition shots."""
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_state_acquisition(self, state: StateKey) -> StateAcquisition:
        """:return: StateAcquisition based on state key."""
        raise InterfaceMethodException
    # endregion


@dataclass(frozen=True)
class StateAcquisitionContainer(IStateAcquisitionContainer):
    """
    Data class, containing raw acquisition shots for state 0, 1 and 2.
    """
    state_acquisition_lookup: Dict[StateKey, StateAcquisition]
    decision_boundaries: DecisionBoundaries = field(init=False)

    # region Interface Properties
    @property
    def contained_states(self) -> List[StateKey]:
        """:return: Array-like of unique contained states."""
        return unique_in_order(self.state_acquisition_lookup.keys())

    @property
    def classification_boundaries(self) -> DecisionBoundaries:
        """:return: DecisionBoundaries."""
        return self.decision_boundaries
    # endregion

    # region Class Properties
    @property
    def estimate_threshold(self) -> float:
        """:return: Estimated 0-1 threshold based on x-axes projection."""
        return self.get_threshold_estimate(
            self.state_acquisition_lookup[StateKey.STATE_0].shots,
            self.state_acquisition_lookup[StateKey.STATE_1].shots,
        )

    @property
    def concatenated_shots(self) -> NDArray[np.complex64]:
        """:return: Array-like of (complex-valued) concatenated acquisition shots."""
        shots: List[NDArray[np.complex64]] = [value.shots for value in self.state_acquisition_lookup.values()]
        return np.concatenate(shots)

    @property
    def concatenated_labels(self) -> NDArray[np.int_]:
        """:return: Array-like of (arbitrary) integer labels, corresponding to self.concatenated_shots order."""
        labels = [[state.value] * len(acquisition.shots) for state, acquisition in
                  self.state_acquisition_lookup.items()]
        return np.concatenate(labels)
    # endregion

    # region Interface Methods
    def get_state_acquisition(self, state: StateKey) -> StateAcquisition:
        """:return: StateAcquisition based on state key."""
        return self.state_acquisition_lookup[state]
    # endregion

    # region Class Methods
    def __post_init__(self):
        object.__setattr__(self, 'decision_boundaries', DecisionBoundaries.from_acquisition_container(self))

    @classmethod
    def from_state_acquisitions(cls, acquisitions: List[StateAcquisition]) -> 'StateAcquisitionContainer':
        """:return: Class method constructor based on array-like of (state) acquisitions."""
        return StateAcquisitionContainer(
            state_acquisition_lookup={
                acquisition.state: acquisition
                for acquisition in acquisitions
            }
        )
    # endregion

    # region Static Class Methods
    # TODO: Move this functionality to general array transformation module
    @staticmethod
    def complex_to_real_imag(complex_data: np.ndarray) -> np.ndarray:
        """
        Converts an array of complex values to an N x 2 real-valued array.
        The first column contains the real parts, and the second contains the imaginary parts.

        :param complex_data: Array of complex numbers.
        :return: N x 2 array with real and imaginary parts as real values.
        """
        real_parts = complex_data.real
        imag_parts = complex_data.imag

        # Stack the real and imaginary parts horizontally
        result = np.column_stack((real_parts, imag_parts))

        return result

    # TODO: Move this functionality to general array transformation module
    @staticmethod
    def real_imag_to_complex(real_imag_data: np.ndarray) -> np.ndarray:
        """
        Converts an N x 2 real-valued array to an N x 1 complex-valued array, with the first column as the real part
        and the second as the imaginary part. If the input is an N x 1 array, it treats values as purely real.

        :param real_imag_data: N x 2 or N x 1 array with real and/or imaginary parts.
        :return: N x 1 array of complex numbers.
        """
        if real_imag_data.ndim != 2 or real_imag_data.shape[1] > 2:
            raise ValueError("Input array must be either N x 1 or N x 2")

        if real_imag_data.shape[1] == 2:
            # Create complex numbers from real and imaginary parts
            complex_data = real_imag_data[:, 0] + 1j * real_imag_data[:, 1]
        else:
            # Treat values as purely real
            complex_data = real_imag_data.ravel() + 0j

        return complex_data  # .reshape(-1, 1)

    @staticmethod
    def get_threshold_estimate(shots_0: NDArray[np.complex64], shots_1: NDArray[np.complex64]) -> float:
        """
        Estimate 0-1 threshold based on x-axes projection.

        :return: Estimated threshold.
        """
        # Extract real-part from shots arrays
        shots_0_projection = shots_0.real
        shots_1_projection = shots_1.real

        # Calculate bounds for histogram
        min_bound = np.min(np.concatenate([shots_0_projection, shots_1_projection]))
        max_bound = np.max(np.concatenate([shots_0_projection, shots_1_projection]))
        midpoint: float = (min_bound + max_bound) / 2

        return midpoint
    # endregion


@dataclass(frozen=True)
class AssignmentProbabilityMatrix:
    """Data class, containing assignment probability matrix and array-like of state-key."""
    state_keys: List[StateKey]
    matrix: NDArray[np.float32]

    # region Class Methods
    @classmethod
    def from_acquisition_container(cls, acquisition_container: IStateAcquisitionContainer) -> 'AssignmentProbabilityMatrix':
        """:return: Class method constructor based on decision boundaries."""
        # Data allocation
        decision_boundaries: DecisionBoundaries = acquisition_container.classification_boundaries
        states: List[StateKey] = list(acquisition_container.contained_states)
        probability_matrix: np.ndarray = np.zeros(shape=(len(states), len(states)))
        for i, _state_key in enumerate(states):
            state_acquisition = acquisition_container.get_state_acquisition(state=_state_key)
            for j, state in enumerate(states):
                probability_matrix[i][j] = decision_boundaries.get_fidelity(
                    shots=state_acquisition.shots,
                    assigned_state=state,
                )

        return AssignmentProbabilityMatrix(
            state_keys=states,
            matrix=probability_matrix,
        )

    def correct_readout_error(
        self,
        measured_probabilities: NDArray[np.float32],
        input_states: List[StateKey],
        clip_values: bool = True
    ) -> Optional[NDArray[np.float32]]:
        """
        Corrects a vector of measured probabilities using THIS instance as the calibration matrix.

        Assumes THIS instance's matrix represents P(measure i | true k) [COLUMNS sum to 1].
        Takes a measured distribution P(measure j) and estimates P(true k).

        :param measured_probabilities: 1D array of measured probabilities P(measure j).
                                       Order defined by input_states. Should sum to 1.
        :param input_states: List of StateKeys corresponding to measured_probabilities,
                             defining the subspace for correction relative to THIS matrix.
        :param clip_values: If True, clips results to [0, 1] and renormalizes.
        :return: Numpy array of corrected probabilities P(true k) in the same order as input_states,
                 cast to float32, or None if correction fails (e.g., singular matrix).
                 Returns an empty float32 array if input sequences are empty.
        :raises ValueError: If validation fails (length mismatch, unknown states in input_states).
        """
        # --- Input Validation ---
        if len(measured_probabilities) != len(input_states):
            raise ValueError(f"Length mismatch: len(measured_probabilities)={len(measured_probabilities)} != len(input_states)={len(input_states)}.")
        if not input_states:
            return np.array([], dtype=np.float32)  # Handle empty input

        # Check sum robustly before proceeding
        prob_sum = np.sum(measured_probabilities)
        if not np.isclose(prob_sum, 1.0, atol=1e-5):
            # Use warning instead of error? Depends on expected usage.
            warn(f"Input probabilities sum to {prob_sum:.6f}, not 1.0. Correction proceeds, but result validity depends on input.")
            # raise ValueError(f"Input probabilities must sum to 1.0, but sum to {prob_sum:.4f}.")

        missing_keys = [ts for ts in input_states if ts not in self.state_keys]
        if missing_keys:
            raise ValueError(f"Input states {missing_keys} are not present in the "
                             f"calibration matrix state keys {self.state_keys}.")
        if len(set(input_states)) != len(input_states):
            raise ValueError(f"Duplicate states found in input_states: {input_states}")
        # --- End Validation ---

        if len(self.state_keys) == 0:
            warn("Cannot correct, AssignmentProbabilityMatrix (calibration matrix) is empty.")
            return None

        # Get Correction Submatrix (from self.matrix, assumed P(measure|true))
        # Use float64 for the submatrix going into inversion
        sub_matrix_64 = self._get_correction_submatrix(input_states).astype(np.float64)

        # Apply Core Correction Math (using float64 for precision)
        raw_corrected_probs_64 = AssignmentProbabilityMatrix.apply_inverse_correction(
            sub_matrix_64,
            measured_probabilities.astype(np.float64) # Ensure input is float64
        )

        if raw_corrected_probs_64 is None:
            return None  # Failure (error already printed by static method)

        # Clip and Re-normalize (Optional) - operates on float64
        final_probs_64 = raw_corrected_probs_64
        if clip_values:
            final_probs_64 = AssignmentProbabilityMatrix.clip_and_normalize(final_probs_64)

        # Return Result Array (cast back to float32)
        return final_probs_64.astype(np.float32)

    def _get_correction_submatrix(self, target_states: Sequence[StateKey]) -> NDArray[np.float32]:
        """
        Extracts the relevant sub-matrix from THIS instance's matrix.

        Assumes self.matrix is P(measure i | true k). Extracts the part relevant
        to the subspace defined by target_states. The target_states define both
        the measured states (rows) and the true states (columns) of the submatrix.

        :param target_states: Sequence of StateKeys defining the subspace.
        :return: The square sub-matrix M_sub[a, b] = P(measure target_state[a] | true target_state[b]).
        """
        if not target_states:
            return np.empty((0, 0), dtype=np.float32)
        try:
            # Get indices corresponding to target_states within self.state_keys
            cal_indices = [self.state_keys.index(ts) for ts in target_states]
        except ValueError as e:
            # Should be caught by validation earlier, but handle defensively.
            raise ValueError(f"State key not found in calibration matrix state keys during submatrix extraction: {e}") from e
        # Extract rows and columns corresponding to these indices
        return self.matrix[np.ix_(cal_indices, cal_indices)]

    def apply_readout_correction(
            self,
            noisy_assignment: 'AssignmentProbabilityMatrix',
            clip_values: bool = True
    ) -> Optional['AssignmentProbabilityMatrix']:
        """
        Corrects a noisy assignment matrix using THIS instance as the calibration matrix.

        Corrects the input matrix where noisy_assignment.matrix[i, j] = P(assigned j | prepared i) [ROWS sum to 1].
        Uses THIS instance where self.matrix[i, k] = P(measure i | true k) [COLUMNS sum to 1] for calibration.
        The output matrix represents P(true k | prepared i) [ROWS sum to 1].

        :param noisy_assignment: An AssignmentProbabilityMatrix instance containing the noisy
                                 assignment matrix P(assigned j | prepared i). Its state_keys
                                 define the prepared/assigned states. ROWS should sum to 1.
        :param clip_values: Whether to clip and normalize the corrected probabilities row-wise.
        :return: A new AssignmentProbabilityMatrix instance containing the corrected matrix
                 P(true k | prepared i), where rows sum to 1, or None if correction fails for any row.
        :raises ValueError: If dimensions or state keys are inconsistent between the
                           noisy matrix and the calibration matrix (self).
        """
        # --- Input Validation ---
        noisy_matrix: NDArray[np.float32] = noisy_assignment.matrix
        matrix_states: List[StateKey] = noisy_assignment.state_keys
        calibration_matrix: AssignmentProbabilityMatrix = self  # Self is the calibrator

        n_states = len(matrix_states)
        if noisy_matrix.shape != (n_states, n_states):
            raise ValueError(f"Shape mismatch: noisy assignment matrix shape {noisy_matrix.shape} "
                             f"does not match its number of states {n_states}.")

        if n_states == 0:
            warn("Input noisy_assignment matrix is empty (0 states). Returning empty matrix.")
            return AssignmentProbabilityMatrix(state_keys=[], matrix=np.empty((0, 0), dtype=np.float32))

        # Verify row sums of the input noisy matrix
        row_sums = np.sum(noisy_matrix, axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            warn(f"Rows of input noisy assignment matrix do not all sum to 1.0 (sums: {row_sums}). "
                          f"Correction assumes rows represent measured probability distributions.")

        # Check if calibration matrix (self) supports the necessary states
        if not all(state in calibration_matrix.state_keys for state in matrix_states):
            missing = [s for s in matrix_states if s not in calibration_matrix.state_keys]
            raise ValueError(f"Calibration matrix (self) is missing states required by the "
                             f"noisy matrix: {missing}. Cannot correct over subspace {matrix_states}.")
        # --- End Validation ---

        # Create an empty matrix for the results
        # Output matrix rows are prepared states, columns are TRUE states
        corrected_matrix = np.zeros_like(noisy_matrix, dtype=np.float32)

        # Iterate through the ROWS of the noisy matrix
        for i, prepared_state in enumerate(matrix_states):
            # Row 'i' represents the measured probabilities P(assigned j | prepared i)
            measured_probs_for_row_i: NDArray[np.float32] = noisy_matrix[i, :]

            # The measured probabilities correspond to the 'matrix_states' (assigned states)
            # We use the 'calibration_matrix' (self) to perform the correction
            # The 'input_states' for correction are the possible measured/assigned states (matrix_states)
            corrected_probs_for_row_i: Optional[NDArray[np.float32]] = calibration_matrix.correct_readout_error(
                measured_probabilities=measured_probs_for_row_i,
                input_states=matrix_states,  # These are the states corresponding to measured_probs_for_row_i
                clip_values=clip_values
            )

            # Check if correction failed for this row
            if corrected_probs_for_row_i is None:
                # Error message already printed by correct_readout_error or its helpers
                warn(f"Readout correction failed for row {i} (prepared state {prepared_state}). Returning None.")
                return None  # Abort correction for the whole matrix

            # Store the corrected probability vector P(true k | prepared i) in row 'i'
            corrected_matrix[i, :] = corrected_probs_for_row_i

        # Optional: Verify row sums of the *output* matrix
        final_row_sums = np.sum(corrected_matrix, axis=1)
        if not np.allclose(final_row_sums, 1.0, atol=1e-5):
            warn(f"Rows of the *corrected* matrix do not all sum to 1.0 (sums: {final_row_sums}). This can happen if clipping was aggressive or input rows didn't sum to 1.")

        # Return a new AssignmentProbabilityMatrix instance
        # Note: The state_keys remain the same, representing the prepared states (rows)
        # and now the TRUE states (columns).
        return AssignmentProbabilityMatrix(
            state_keys=matrix_states,
            matrix=corrected_matrix,
        )
    # endregion

    # region Static Class Methods
    @staticmethod
    def apply_inverse_correction(sub_matrix: NDArray[np.float64], probability_array: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
        """
        Applies P_true = inv(M_sub) * P_meas using float64 for precision.

        :param sub_matrix: Calibration submatrix P(measure|true), float64.
        :param probability_array: Measured probability vector P(measure), float64.
        :return: Raw corrected probability P(true) as float64 array, or None if inversion fails.
        """
        if probability_array.size == 0:
            return np.array([], dtype=np.float64)

        p_meas_sub = probability_array.reshape(-1, 1)
        try:
            m_sub_inv = np.linalg.inv(sub_matrix)
        except np.linalg.LinAlgError:
            warn(f"Singular matrix encountered during inversion:\n{sub_matrix}")
            return None
        p_true_sub = m_sub_inv @ p_meas_sub
        return p_true_sub.flatten()

    @staticmethod
    def clip_and_normalize(prob_array: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Clips float64 probabilities to [0, 1] and renormalizes to sum to 1.

        :param prob_array: Float64 probability array P(true).
        :return: Clipped and normalized float64 probability array.
        """
        if prob_array.size == 0:
            return np.array([], dtype=np.float64)

        # Check for significant negative values before clipping
        if np.any(prob_array < -1e-7):
            warn(f"Corrected probabilities contained negative values before clipping: {prob_array}")

        clipped_probs = np.clip(prob_array, 0.0, 1.0)
        sum_clipped = np.sum(clipped_probs)

        if sum_clipped > 1e-9:
            normalized_probs = clipped_probs / sum_clipped
            # Ensure sum is exactly 1? Usually not necessary but can enforce:
            # normalized_probs /= np.sum(normalized_probs)
            return normalized_probs
        else:
            # If sum is zero after clipping
            warn("Corrected probabilities clipped/normalized to zero sum.")
            return clipped_probs  # Return the array of zeros
    # endregion


TStateClassifierContainer = TypeVar('TStateClassifierContainer', bound='IStateClassifierContainer')


class IStateClassifierContainer(ABC):
    """
    Interface class, describing get methods for access state information.
    """

    # region Interface Properties
    @property
    @abstractmethod
    def expected_parity(self) -> ParityType:
        """:return: Expected parity property."""
        raise InterfaceMethodException

    @property
    @abstractmethod
    def stabilizer_reset(self) -> bool:
        """:return: Boolean whether parity resets each round."""
        raise InterfaceMethodException

    @property
    def odd_weight_and_refocusing(self) -> bool:
        """
        :return: Boolean whether parity is based on odd-number of element and elements are (flipped) refocused each round.
        Mainly relevant for (weight-1) edge-stabilizers in 1D stability experiment.
        """
        raise InterfaceMethodException
    # endregion

    # region Interface Methods
    @abstractmethod
    def get_binary_classification(self) -> NDArray[np.int_]:
        """:return: Binary classification."""
        raise InterfaceMethodException

    @abstractmethod
    def get_ternary_classification(self) -> NDArray[np.int_]:
        """:return: Ternary classification."""
        raise InterfaceMethodException

    @abstractmethod
    def get_eigenvalue_classification(self) -> NDArray[np.int_]:
        """:return: Eigenvalue (0 -> +1, 1 -> -1) classification."""
        raise InterfaceMethodException

    @abstractmethod
    def get_parity_classification(self) -> NDArray[np.int_]:
        """:return: Parity classification based on eigenvalue classification."""
        raise InterfaceMethodException

    @abstractmethod
    def get_defect_classification(self) -> NDArray[np.int_]:
        """:return: Defect classification based on parity classification."""
        raise InterfaceMethodException

    @classmethod
    def reshape(cls, container: TStateClassifierContainer, index_slices: NDArray[np.int_]) -> TStateClassifierContainer:
        """:return: Reshaped version of state-classifiers based on iterator of index-slices."""
        raise InterfaceMethodException
    # endregion

    # region Static Interface Methods
    @staticmethod
    def calculate_parity(m: np.ndarray) -> np.ndarray:
        """
        Calculate the derivative of a tensor of +1 and -1 values using the formula
        p[n+1] = m[n+1] * m[n], with p[0] = m[0]. The operation is performed along the
        last dimension if its size is not 1, otherwise along the second-to-last dimension.

        :param m: Input tensor with arbitrary shape.
        :return: First derivative of m -> p.
        """
        result: np.ndarray = IStateClassifierContainer.calculate_derivative(m=m)
        # Determine the dimension along which to perform the operation
        axis: int = -1

        index = [slice(None)] * result.ndim  # Create a slicer for all dimensions
        index[axis] = 0  # Update the slicer to target the first element along the chosen axis
        result[tuple(index)] = m[tuple(index)]  # Update parity at first element along the chosen axis

        return result

    @staticmethod
    def calculate_defect(m: np.ndarray, initial_condition: int = +1, odd_weight_and_refocusing: bool = False) -> np.ndarray:
        """
        Calculate the derivative of a tensor of +1 and -1 values using the formula
        p[n+1] = m[n+1] * m[n], with p[0] = (initial condition) * m[0]. The operation is performed along the
        last dimension if its size is not 1, otherwise along the second-to-last dimension.

        :param m: Input tensor with arbitrary shape.
        :param initial_condition: Initial condition after taking derivative. (For p[0])
        :param odd_weight_and_refocusing: If stabilizer defect originates from an odd-weight parity AND data qubits get refocused every round, the definition of defect is inverted.
        :return: First derivative of m -> p.
        """
        result = IStateClassifierContainer.calculate_derivative(m=m, initial_condition=initial_condition)
        if odd_weight_and_refocusing:
            result *= -1
        return result

    @staticmethod
    def calculate_derivative(m: np.ndarray, initial_condition: int = +1) -> np.ndarray:
        """
        Calculate the derivative of a tensor of +1 and -1 values using the formula
        p[n+1] = m[n+1] * m[n], with p[0] = (initial condition) * m[0]. The operation is performed along the
        last dimension if its size is not 1, otherwise along the second-to-last dimension.

        :param m: Input tensor with arbitrary shape.
        :param initial_condition: Initial condition after taking derivative. (For p[0])
        :return: First derivative of m -> p.
        """
        # Determine the dimension along which to perform the operation
        axis: int = -1

        # Shift the array by one position along the chosen axis, rolling the last element to the start
        m_shifted = np.roll(m, shift=1, axis=axis)

        # Set the first element along the chosen axis to +1
        index = [slice(None)] * m.ndim  # Create a slicer for all dimensions
        index[axis] = 0  # Update the slicer to target the first element along the chosen axis
        m_shifted[tuple(index)] = initial_condition

        # Element-wise multiplication of the original and shifted tensors
        p = m * m_shifted

        return p

    @staticmethod
    def binary_to_eigenvalue(m: np.ndarray) -> np.ndarray:
        """:return: Translated array from binary to eigenvalue subspace. (0 -> +1, 1 -> -1)."""
        return (1 - (m * 2)).astype(int)

    @staticmethod
    def eigenvalue_to_binary(m: np.ndarray) -> np.ndarray:
        """:return: Translated array from eigenvalue to binary subspace. (+1 -> 0, -1 -> 1)."""
        return ((1 - m) // 2).astype(int)
    # endregion


@dataclass(frozen=True)
class StateClassifierContainer(IStateClassifierContainer):
    """Data class, containing classified states based on already classified states."""
    state_classification: NDArray[np.int_]
    _expected_parity: ParityType = field(default=ParityType.EVEN)
    _stabilizer_reset: bool = field(default=False)
    _odd_weight_and_refocusing: bool = field(default=False)

    # region Interface Properties
    @property
    def expected_parity(self) -> ParityType:
        """:return: Expected parity property."""
        return self._expected_parity

    @property
    def stabilizer_reset(self) -> bool:
        """:return: Boolean whether parity resets each round."""
        return self._stabilizer_reset

    @property
    def odd_weight_and_refocusing(self) -> bool:
        """
        :return: Boolean whether parity is based on odd-number of element and elements are (flipped) refocused each round.
        Mainly relevant for (weight-1) edge-stabilizers in 1D stability experiment.
        """
        return self._odd_weight_and_refocusing
    # endregion

    # region Class Methods
    def get_binary_classification(self) -> NDArray[np.int_]:
        """:return: Binary classification based on acquisition shots and decision boundaries."""
        return self.state_classification

    def get_ternary_classification(self) -> NDArray[np.int_]:
        """:return: Ternary classification based on acquisition shots and decision boundaries."""
        return self.state_classification

    def get_eigenvalue_classification(self) -> NDArray[np.int_]:
        """:return: Eigenvalue (0 -> +1, 1 -> -1) classification based on acquisition shots and decision boundaries."""
        return IStateClassifierContainer.binary_to_eigenvalue(self.get_binary_classification())

    def get_parity_classification(self) -> NDArray[np.int_]:
        """:return: Parity classification based on eigenvalue classification."""
        if self._stabilizer_reset:
            return self.get_eigenvalue_classification()

        return IStateClassifierContainer.calculate_parity(
            m=self.get_eigenvalue_classification(),
        )

    def get_defect_classification(self) -> NDArray[np.int_]:
        """:return: Defect classification based on parity classification."""
        return IStateClassifierContainer.calculate_defect(
            m=self.get_parity_classification(),
            initial_condition=self.expected_parity.value,
            odd_weight_and_refocusing=self.odd_weight_and_refocusing,
        )

    def get_defect_rate(self) -> float:
        """:return: Defect rate based on defect classification."""
        # Determine the dimension along which to perform the operation
        defect_array: np.ndarray = IStateClassifierContainer.eigenvalue_to_binary(self.get_defect_classification())
        axis = -1 if defect_array.shape[-1] != 1 else -2
        return float(np.mean(defect_array, axis=axis))

    @classmethod
    def reshape(cls, container: TStateClassifierContainer, index_slices: NDArray[np.int_]) -> TStateClassifierContainer:
        """:return: Reshaped version of state-classifiers based on iterator of index-slices."""
        return StateClassifierContainer(
            state_classification=np.array([container.state_classification[index_slice] for index_slice in index_slices]),
            _expected_parity=container.expected_parity,
            _stabilizer_reset=container.stabilizer_reset,
        )
    # endregion


@dataclass(frozen=True)
class ShotsClassifierContainer(IStateClassifierContainer):
    """Data class, containing classified states based on (complex) acquisition and decision boundaries."""
    shots: NDArray[np.complex64]
    decision_boundaries: DecisionBoundaries
    _expected_parity: ParityType = field(default=ParityType.EVEN)
    _stabilizer_reset: bool = field(default=False)
    _odd_weight_and_refocusing: bool = field(default=False)

    # region Interface Properties
    @property
    def expected_parity(self) -> ParityType:
        """:return: Expected parity property."""
        return self._expected_parity

    @property
    def stabilizer_reset(self) -> bool:
        """:return: Boolean whether parity resets each round."""
        return self._stabilizer_reset

    @property
    def odd_weight_and_refocusing(self) -> bool:
        """
        :return: Boolean whether parity is based on odd-number of element and elements are (flipped) refocused each round.
        Mainly relevant for (weight-1) edge-stabilizers in 1D stability experiment.
        """
        return self._odd_weight_and_refocusing
    # endregion

    # region Class Properties
    @property
    def state_classifier(self) -> StateClassifierContainer:
        """:return: Pure state classifier based on self."""
        # Guard clause, if shots is empty, raise exception
        if self.shots.size == 0:
            raise ZeroClassifierShotsException(f"Array size of shots used to perform state-classification is {self.shots.size}. Perhaps all shots are filtered by post-selection method?")

        return StateClassifierContainer(
            state_classification=self._process_tensor(self.shots, self.decision_boundaries.get_binary_predictions),
            _expected_parity=self.expected_parity,
            _stabilizer_reset=self.stabilizer_reset,
            _odd_weight_and_refocusing=self.odd_weight_and_refocusing,
        )

    @property
    def state_classifier_ternary(self) -> StateClassifierContainer:
        """:return: Pure state classifier based on self."""
        return StateClassifierContainer(
            state_classification=self._process_tensor(self.shots, self.decision_boundaries.get_predictions),
            _expected_parity=self.expected_parity,
            _stabilizer_reset=self.stabilizer_reset,
            _odd_weight_and_refocusing=self.odd_weight_and_refocusing,
        )
    # endregion

    # region Class Methods
    def get_binary_classification(self) -> NDArray[np.int_]:
        """:return: Binary classification based on acquisition shots and decision boundaries."""
        return self.state_classifier.get_binary_classification()

    def get_ternary_classification(self) -> NDArray[np.int_]:
        """:return: Ternary classification based on acquisition shots and decision boundaries."""
        return self.state_classifier_ternary.get_ternary_classification()

    def get_eigenvalue_classification(self) -> NDArray[np.int_]:
        """:return: Eigenvalue (0 -> +1, 1 -> -1) classification based on acquisition shots and decision boundaries."""
        return self.state_classifier.get_eigenvalue_classification()

    def get_parity_classification(self) -> NDArray[np.int_]:
        """:return: Parity classification based on eigenvalue classification."""
        return self.state_classifier.get_parity_classification()

    def get_defect_classification(self) -> NDArray[np.int_]:
        """:return: Defect classification based on parity classification."""
        return self.state_classifier.get_defect_classification()

    def get_defect_rate(self) -> float:
        """:return: Defect rate based on defect classification."""
        return self.state_classifier.get_defect_rate()

    @classmethod
    def reshape(cls, container: TStateClassifierContainer, index_slices: NDArray[np.int_]) -> TStateClassifierContainer:
        """:return: Reshaped version of state-classifiers based on iterator of index-slices."""
        return ShotsClassifierContainer(
            shots=np.array([container.shots[index_slice] for index_slice in index_slices]),
            decision_boundaries=container.decision_boundaries,
            _expected_parity=container.expected_parity,
            _stabilizer_reset=container.stabilizer_reset,
            _odd_weight_and_refocusing=container.odd_weight_and_refocusing,
        )
    # endregion

    # region Static Class Methods
    @staticmethod
    def _process_tensor(tensor: np.ndarray, func: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Flattens a tensor of arbitrary shape, processes it with a given function,
        and then reshapes it back to the original shape.

        :param tensor: Tensor (array) of arbitrary shape.
        :param func: Function that processes a 1D array.
        :return: Tensor reshaped to its original shape after processing.
        """
        original_shape = tensor.shape
        flattened_tensor: np.ndarray = tensor.flatten()
        processed_tensor: np.ndarray = func(flattened_tensor)
        reshaped_tensor: np.ndarray = processed_tensor.reshape(original_shape)
        return reshaped_tensor
    # endregion
