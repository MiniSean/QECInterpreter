# -------------------------------------------
# Module for visualizing state classification.
# -------------------------------------------
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Tuple
from enum import Enum, unique, auto
from qce_interp.utilities.geometric_definitions import Vec2D, Polygon, euclidean_distance
from qce_interp.interface_definitions.intrf_state_classification import (
    StateAcquisitionContainer,
    StateBoundaryKey,
    DecisionBoundaries,
    StateKey,
)
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    LabelFormat,
    AxesFormat,
    SubplotKeywordEnum,
    IAxesFormat,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import ListedColormap, PowerNorm, Colormap
from matplotlib import colormaps
from matplotlib import patches


STATE_COLORMAP: Dict[StateKey, Colormap] = {
    StateKey.STATE_0: colormaps.get_cmap("Blues"),
    StateKey.STATE_1: colormaps.get_cmap("Reds"),
    StateKey.STATE_2: colormaps.get_cmap("Greens"),
}
STATE_LABELMAP: Dict[StateKey, str] = {
    StateKey.STATE_0: 'ground',
    StateKey.STATE_1: 'excited',
    StateKey.STATE_2: '$2^\mathrm{nd}$ excited',
}


@unique
class TraversalDirection(Enum):
    CLOCKWISE = auto()
    COUNTER_CLOCKWISE = auto()


class IQAxesFormat(IAxesFormat):
    """
    Specifies general axis formatting functions.
    """

    # region Interface Methods
    def apply_to_axes(self, axes: plt.Axes) -> plt.Axes:
        """
        Applies axes formatting settings to axis.
        :param axes: Axes to be formatted.
        :return: Updated Axes.
        """
        axes.grid(True, alpha=0.5, linestyle='dashed')  # Adds dashed gridlines
        axes.set_aspect('equal', adjustable='box')
        axes.set_axisbelow(True)  # Puts grid on background

        nr_ticks: int = 5
        # Set the locator for both x and y axes to ensure 5 ticks
        axes.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nr_ticks))
        axes.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nr_ticks))
        # Set minor ticks to off
        axes.xaxis.set_minor_locator(ticker.NullLocator())
        axes.yaxis.set_minor_locator(ticker.NullLocator())
        return axes
    # endregion

    # region Static Class Methods
    @staticmethod
    def default() -> 'AxesFormat':
        """:return: Default AxesFormat instance."""
        return AxesFormat()
    # endregion


def determine_axes_limits(state_classifier: StateAcquisitionContainer) -> Tuple[float, float, float, float]:
    """
    Compares maximum shot values with discrete selection of axes limits.
    Chooses appropriate axes limit for given data.
    :param state_classifier: State acquisition container, containing single-shot data.
    :return: Tuple of 4 limits (min_x, max_x, min_y, max_y).
    """
    all_shots: NDArray[np.complex_] = state_classifier.concatenated_shots
    maximum_limit: float = max(
        abs(min(all_shots.real)),
        abs(max(all_shots.real)),
        abs(min(all_shots.imag)),
        abs(max(all_shots.imag)),
    )
    # Default
    axes_limit: float = 1.0
    possible_axes_limits: List[float] = [0.1, 0.25, 0.5, 1.0, 2.0]
    for possible_axes_limit in possible_axes_limits:
        margin_percentage: float = 0.9  # Maximum (axes) limit should be less than 90% of total axes limit
        if maximum_limit < margin_percentage * possible_axes_limit:
            axes_limit = possible_axes_limit
            break
    return -axes_limit, +axes_limit, -axes_limit, +axes_limit


def plot_state_shots(state_classifier: StateAcquisitionContainer, **kwargs) -> IFigureAxesPair:
    """
    Plots state shots for a given state classifier.

    :param state_classifier: Container with state classification data.
    :param kwargs: Additional keyword arguments for plot customization.
    :return: Tuple containing the figure and axes of the plot.
    """
    # Data allocation
    mincnt = 1
    extent: Tuple[float, float, float, float] = determine_axes_limits(state_classifier=state_classifier)
    power_gamma: float = 0.45

    # Figures and Axes
    fig, ax = construct_subplot(**kwargs)

    alpha_colormaps: List[ListedColormap] = []
    for colormap in STATE_COLORMAP.values():
        sub_colormap: np.ndarray = colormap(np.arange(colormap.N))
        sub_colormap[:, -1] = np.linspace(0, 1, colormap.N)
        listed_colormap: ListedColormap = ListedColormap(sub_colormap)
        alpha_colormaps.append(listed_colormap)

    for state, state_acquisition in state_classifier.state_acquisition_lookup.items():
        ax.hexbin(
            x=state_acquisition.shots.real,
            y=state_acquisition.shots.imag,
            cmap=alpha_colormaps[state_acquisition.state.value],
            mincnt=mincnt,
            extent=extent,
            norm=PowerNorm(gamma=power_gamma),
            zorder=-state.value,
        )
        # For legend only
        ax.plot(
            np.nan,
            np.nan,
            linestyle='none',
            marker='o',
            color=STATE_COLORMAP[state](0.5),
            label=STATE_LABELMAP[state],
        )
    return fig, ax


def get_neighboring_boundary_keys(state: StateKey, boundary_keys: List[StateBoundaryKey]) -> List[StateBoundaryKey]:
    """
    Retrieves keys of boundaries neighboring a specified state.

    :param state: The state for which neighboring boundaries are sought.
    :param boundary_keys: List of all possible state boundary keys.
    :return: List of neighboring state boundary keys.
    """
    return [boundary_key for boundary_key in boundary_keys if state in boundary_key]


def find_axes_intersection(start_point: Vec2D, other_point: Vec2D, ax: plt.Axes) -> Vec2D:
    """
    Finds the intersection point of a line with plot axes boundaries.

    :param start_point: Starting point of the line.
    :param other_point: Other point on the line.
    :param ax: The axes on which the line is drawn.
    :return: Intersection point with axes boundaries.
    """
    x1, y1 = start_point.to_tuple()
    x2, y2 = other_point.to_tuple()
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()

    potential_intersections = []

    # Handle the case where the line is vertical
    if x1 == x2:
        if min_y <= y1 <= max_y or min_y <= y2 <= max_y:
            potential_intersections.append(Vec2D(x=x1, y=max_y if y2 > y1 else min_y))

    else:
        # Calculate slope and y-intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Check intersection with y-axis limits
        for y_limit in [min_y, max_y]:
            x_at_y = (y_limit - b) / m
            if min_x <= x_at_y <= max_x:
                potential_intersections.append(Vec2D(x=x_at_y, y=y_limit))

        # Check intersection with x-axis limits
        for x_limit in [min_x, max_x]:
            y_at_x = m * x_limit + b
            if min_y <= y_at_x <= max_y:
                potential_intersections.append(Vec2D(x=x_limit, y=y_at_x))

    # Find the closest intersection to other_point
    closest_intersection = None
    min_distance_to_other = float('inf')
    for intersection in potential_intersections:
        distance_to_other = euclidean_distance(intersection, other_point)
        if distance_to_other < min_distance_to_other:
            closest_intersection = intersection
            min_distance_to_other = distance_to_other

    return closest_intersection if closest_intersection else other_point


def get_axes_intersection_lookup(decision_boundaries: DecisionBoundaries, ax: plt.Axes) -> Dict[StateBoundaryKey, Vec2D]:
    """
    Creates a lookup for intersection points of decision boundaries with axes.

    :param decision_boundaries: Decision boundaries of states.
    :param ax: The axes on which the boundaries are drawn.
    :return: Dictionary mapping boundary keys to intersection points.
    """
    # Data allocation
    result: Dict[StateBoundaryKey, Vec2D] = {}
    center: Vec2D = decision_boundaries.mean
    boundary_keys: List[StateBoundaryKey] = list(decision_boundaries.boundary_lookup.keys())

    for boundary_key in boundary_keys:
        boundary_point: Vec2D = decision_boundaries.get_boundary(key=boundary_key)
        intersection_point: Vec2D = find_axes_intersection(start_point=center, other_point=boundary_point, ax=ax)
        result[boundary_key] = intersection_point

    return result


def get_axes_vertices(ax: plt.Axes) -> List[Vec2D]:
    """
    Retrieves the corner vertices of the axes boundaries.

    :param ax: The axes whose boundaries are considered.
    :return: List of Vec2D objects representing the corner vertices.
    """
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    return [
        Vec2D(min_x, min_y),
        Vec2D(max_x, min_y),
        Vec2D(max_x, max_y),
        Vec2D(min_x, max_y),
    ]


def filter_vertices_within_smaller_angle(center: Vec2D, intersection1: Vec2D, intersection2: Vec2D, vertices: List[Vec2D]) -> List[Vec2D]:
    """
    Filters vertices lying within the smaller angle formed by vectors from the center to two intersection points.

    :param center: Center Vec2D point.
    :param intersection1: First intersection Vec2D point.
    :param intersection2: Second intersection Vec2D point.
    :param vertices: List of Vec2D vertices to be filtered.
    :return: Filtered list of Vec2D vertices.
    """

    def is_clockwise(v1: np.ndarray, v2: np.ndarray) -> bool:
        """:return: Whether moving from v1 to v2 is a clockwise rotation or not."""
        return np.cross(v1, v2) < 0

    # Convert Vec2D to numpy arrays
    vec1: np.ndarray = intersection1.to_vector() - center.to_vector()
    vec2: np.ndarray = intersection2.to_vector() - center.to_vector()

    # Determine the direction of the smaller angle (clockwise or counter-clockwise)
    cross_product = np.cross(vec1, vec2)
    smaller_angle_is_ccw = cross_product > 0

    # Filter the vertices within the smaller angle
    filtered_vertices = []
    for vertex in vertices:
        if vertex == center or vertex == intersection1 or vertex == intersection2:
            continue

        vertex_vector: np.ndarray = vertex.to_vector() - center.to_vector()
        cw_to_vec1 = is_clockwise(vertex_vector, vec1)
        cw_to_vec2 = is_clockwise(vertex_vector, vec2)

        # For counter-clockwise smaller angle, check if vertex is clockwise to vec1 and counter-clockwise to vec2
        # For clockwise smaller angle, check if vertex is counter-clockwise to vec1 and clockwise to vec2
        within_ccw_smaller_angle: bool = (smaller_angle_is_ccw and cw_to_vec1 and not cw_to_vec2)
        within_cw_smaller_angle: bool = (not smaller_angle_is_ccw and not cw_to_vec1 and cw_to_vec2)
        if within_ccw_smaller_angle or within_cw_smaller_angle:
            filtered_vertices.append(vertex)

    return filtered_vertices


def plot_decision_boundary(decision_boundaries: DecisionBoundaries, **kwargs) -> IFigureAxesPair:
    """
    Plots decision boundaries for state classification.

    :param decision_boundaries: Decision boundaries of states.
    :param kwargs: Additional keyword arguments for plot customization.
    :return: Tuple containing the figure and axes of the plot.
    """
    # Data allocation
    center: Vec2D = decision_boundaries.mean
    boundary_keys: List[StateBoundaryKey] = list(decision_boundaries.boundary_lookup.keys())

    # Figures and Axes
    fig, ax = construct_subplot(**kwargs)
    boundary_intersections: Dict[StateBoundaryKey, Vec2D] = get_axes_intersection_lookup(decision_boundaries=decision_boundaries, ax=ax)

    # Store the current limits
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()
    for boundary_key in boundary_keys:
        intersection_point: Vec2D = boundary_intersections[boundary_key]
        ax.plot(
            [center.x, intersection_point.x],
            [center.y, intersection_point.y],
            linestyle='--',
            color='k',
            linewidth=1,
        )
        # Restore the original limits
        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)

    return fig, ax


def plot_decision_region(state_classifier: StateAcquisitionContainer, **kwargs) -> IFigureAxesPair:
    """
    Plots decision regions for state classification.

    :param state_classifier: Container with state classification data.
    :param kwargs: Additional keyword arguments for plot customization.
    :return: Tuple containing the figure and axes of the plot.
    """
    # Data allocation
    decision_boundaries: DecisionBoundaries = state_classifier.decision_boundaries
    center: Vec2D = decision_boundaries.mean
    boundary_keys: List[StateBoundaryKey] = list(decision_boundaries.boundary_lookup.keys())

    # Figures and Axes
    fig, ax = construct_subplot(**kwargs)
    boundary_intersections: Dict[StateBoundaryKey, Vec2D] = get_axes_intersection_lookup(decision_boundaries=decision_boundaries, ax=ax)

    # Store the current limits
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    rectangle_vertices: List[Vec2D] = get_axes_vertices(ax=ax)
    for state in state_classifier.state_acquisition_lookup.keys():
        color = STATE_COLORMAP[state](1.0)
        neighbor_boundary_keys: List[StateBoundaryKey] = get_neighboring_boundary_keys(
            state=state,
            boundary_keys=boundary_keys,
        )
        intersection1, intersection2 = boundary_intersections[neighbor_boundary_keys[0]], boundary_intersections[
            neighbor_boundary_keys[1]]
        vertices: List[Vec2D] = filter_vertices_within_smaller_angle(
            center=center,
            intersection1=intersection1,
            intersection2=intersection2,
            vertices=rectangle_vertices,
        )
        polygon: Polygon = Polygon(vertices=[intersection2, center, intersection1] + vertices)

        vertices: np.ndarray = np.asarray([vertex.to_tuple() for vertex in polygon.get_convex_vertices()])
        _patch = patches.Polygon(vertices, color=color, alpha=0.2, lw=0, zorder=-10)
        ax.add_patch(_patch)
        # break

    # Restore the original limits
    ax.set_xlim(original_xlim)
    ax.set_ylim(original_ylim)
    return fig, ax


def plot_state_classification(state_classifier: StateAcquisitionContainer, **kwargs) -> IFigureAxesPair:
    """
    Creates a plot visualizing state classification and decision boundaries.

    :param state_classifier: Container with state classification data.
    :param kwargs: Additional keyword arguments for plot customization.
    :return: Tuple containing the figure and axes of the plot.
    """
    decision_boundaries: DecisionBoundaries = state_classifier.decision_boundaries
    kwargs[SubplotKeywordEnum.AXES_FORMAT.value] = IQAxesFormat()
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = LabelFormat(
        x_label='Integrated voltage I',
        y_label='Integrated voltage Q',
    )
    # Figure and Axes
    fig, ax = construct_subplot(**kwargs)
    kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
    plot_state_shots(state_classifier=state_classifier, **kwargs)
    plot_decision_boundary(decision_boundaries=decision_boundaries, **kwargs)
    fig, ax = plot_decision_region(state_classifier=state_classifier, **kwargs)
    ax.legend(frameon=False)
    return fig, ax
