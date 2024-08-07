# -------------------------------------------
# Module containing interface and implementation of state discrimination and classification.
# -------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, unique
import itertools
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy.typing import NDArray
from typing import List, Dict, Optional, Callable, TypeVar
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
    shots: NDArray[np.complex128]

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

    def get_binary_predictions(self, shots: NDArray[np.complex128]) -> NDArray[np.int_]:
        """
        NOTE: Forces classification of element in group 1 or 2, disregarding other groups.
        NOTE: Returns integer prediction value, can be mapped to state-enum using self.prediction_index_to_state_lookup.
        :return: Array-like of State key predictions based on shots discrimination.
        """
        shot_reshaped: NDArray[np.float64] = StateAcquisitionContainer.complex_to_real_imag(shots)
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

    def get_predictions(self, shots: NDArray[np.complex128]) -> NDArray[np.int_]:
        """
        NOTE: Returns integer prediction value, can be mapped to state-enum using self.prediction_index_to_state_lookup.
        :return: Array-like of State key predictions based on shots discrimination.
        """
        shot_reshaped: NDArray[np.float64] = StateAcquisitionContainer.complex_to_real_imag(shots)
        state_indices: NDArray[np.int_] = self._discriminator.predict(shot_reshaped)  # 1 indexed
        return state_indices

    def get_prediction(self, shot: np.complex128) -> StateKey:
        """:return: State key prediction based on shot discrimination."""
        state_indices: NDArray[np.int_] = self.get_predictions(shots=np.asarray([shot]))
        int_to_enum = self.prediction_index_to_state_lookup
        return int_to_enum[state_indices[0]]

    def get_fidelity(self, shots: NDArray[np.complex128], assigned_state: StateKey) -> float:
        """:return: Assignment fidelity defined as the probability of shots being part of assigned state."""
        shots_reshaped: NDArray[np.float64] = StateAcquisitionContainer.complex_to_real_imag(shots)
        state_indices: NDArray[np.int_] = self._discriminator.predict(shots_reshaped)  # 1 indexed
        return float(np.mean(state_indices == self._state_lookup[assigned_state]))

    def post_select_on(self, shots_to_filter: NDArray[np.complex128], conditional_shots: NDArray[np.complex128], conditional_state: StateKey) -> NDArray[np.complex128]:
        """:return: Filtered shots based on conditional shots (of same length) and conditional state."""
        # Guard clause, if conditional shots are empty, return shots without filtering
        if len(conditional_shots) == 0:
            return shots_to_filter

        conditional_shots_reshaped: NDArray[np.float64] = StateAcquisitionContainer.complex_to_real_imag(
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
            raise ZeroDivisionError(f"During instersect calculation of {coef1} and {coef2}.")
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


@dataclass(frozen=True)
class StateAcquisitionContainer:
    """Data class, containing raw acquisition shots for state 0, 1 and 2."""
    state_acquisition_lookup: Dict[StateKey, StateAcquisition]
    decision_boundaries: DecisionBoundaries = field(init=False)

    # region Class Properties
    @property
    def estimate_threshold(self) -> float:
        """:return: Estimated 0-1 threshold based on x-axes projection."""
        return self.get_threshold_estimate(
            self.state_acquisition_lookup[StateKey.STATE_0].shots,
            self.state_acquisition_lookup[StateKey.STATE_1].shots,
        )

    @property
    def concatenated_shots(self) -> NDArray[np.complex128]:
        """:return: Array-like of (complex-valued) concatenated acquisition shots."""
        shots: List[NDArray[np.complex128]] = [value.shots for value in self.state_acquisition_lookup.values()]
        return np.concatenate(shots)

    @property
    def concatenated_labels(self) -> NDArray[np.int_]:
        """:return: Array-like of (arbitrary) integer labels, corresponding to self.concatenated_shots order."""
        labels = [[state.value] * len(acquisition.shots) for state, acquisition in
                  self.state_acquisition_lookup.items()]
        return np.concatenate(labels)

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
    def get_threshold_estimate(shots_0: NDArray[np.complex128], shots_1: NDArray[np.complex128]) -> float:
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
class AssignmentFidelityMatrix:
    """Data class, containing assignment fidelity matrix and array-like of state-key."""
    state_keys: List[StateKey]
    matrix: NDArray[np.float64]

    # region Class Methods
    @classmethod
    def from_acquisition_container(cls, acquisition_container: StateAcquisitionContainer) -> 'AssignmentFidelityMatrix':
        """:return: Class method constructor based on decision boundaries."""
        # Data allocation
        decision_boundaries: DecisionBoundaries = acquisition_container.decision_boundaries
        states: List[StateKey] = list(acquisition_container.state_acquisition_lookup.keys())
        fidelity_matrix: np.ndarray = np.zeros(shape=(len(states), len(states)))
        for i, state_acquisision in enumerate(acquisition_container.state_acquisition_lookup.values()):
            for j, state in enumerate(states):
                fidelity_matrix[i][j] = decision_boundaries.get_fidelity(
                    shots=state_acquisision.shots,
                    assigned_state=state,
                )

        return AssignmentFidelityMatrix(
            state_keys=states,
            matrix=fidelity_matrix,
        )
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
    def calculate_defect(m: np.ndarray, initial_condition: int = +1) -> np.ndarray:
        """
        Calculate the derivative of a tensor of +1 and -1 values using the formula
        p[n+1] = m[n+1] * m[n], with p[0] = (initial condition) * m[0]. The operation is performed along the
        last dimension if its size is not 1, otherwise along the second-to-last dimension.

        :param m: Input tensor with arbitrary shape.
        :param initial_condition: Initial condition after taking derivative. (For p[0])
        :return: First derivative of m -> p.
        """
        return IStateClassifierContainer.calculate_derivative(m=m, initial_condition=initial_condition)

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
    state_classification: NDArray[int]
    _expected_parity: ParityType = field(default=ParityType.EVEN)

    # region Interface Properties
    @property
    def expected_parity(self) -> ParityType:
        """:return: Expected parity property."""
        return self._expected_parity
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
        return IStateClassifierContainer.calculate_parity(
            m=self.get_eigenvalue_classification(),
        )

    def get_defect_classification(self) -> NDArray[np.int_]:
        """:return: Defect classification based on parity classification."""
        return IStateClassifierContainer.calculate_defect(
            m=self.get_parity_classification(),
            initial_condition=self.expected_parity.value,
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
        )
    # endregion


@dataclass(frozen=True)
class ShotsClassifierContainer(IStateClassifierContainer):
    """Data class, containing classified states based on (complex) acquisition and decision boundaries."""
    shots: NDArray[np.complex128]
    decision_boundaries: DecisionBoundaries
    _expected_parity: ParityType = field(default=ParityType.EVEN)

    # region Interface Properties
    @property
    def expected_parity(self) -> ParityType:
        """:return: Expected parity property."""
        return self._expected_parity
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
        )

    @property
    def state_classifier_ternary(self) -> StateClassifierContainer:
        """:return: Pure state classifier based on self."""
        return StateClassifierContainer(
            state_classification=self._process_tensor(self.shots, self.decision_boundaries.get_predictions),
            _expected_parity=self.expected_parity,
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
