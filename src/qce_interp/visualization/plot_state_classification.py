# -------------------------------------------
# Module for visualizing state classification.
# -------------------------------------------
import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Tuple
from enum import Enum, unique, auto
from qce_interp.utilities.geometric_definitions import Vec2D, Polygon, euclidean_distance
from qce_interp.interface_definitions.intrf_state_classification import (
    IStateAcquisitionContainer,
    StateAcquisitionContainer,
    StateBoundaryKey,
    DirectedStateBoundaryKey,
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
        # Set scientific notation
        axes.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axes.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axes.ticklabel_format(style='scientific', axis='both', scilimits=(-1, 1))
        return axes
    # endregion

    # region Static Class Methods
    @staticmethod
    def default() -> 'AxesFormat':
        """:return: Default AxesFormat instance."""
        return AxesFormat()
    # endregion


def determine_axes_limits(state_classifier: IStateAcquisitionContainer) -> Tuple[float, float, float, float]:
    """
    Compares maximum shot values with discrete selection of axes limits.
    Chooses appropriate axes limit for given data.
    :param state_classifier: State acquisition container, containing single-shot data.
    :return: Tuple of 4 limits (min_x, max_x, min_y, max_y).
    """
    all_shots: NDArray[np.complex64] = state_classifier.concatenated_shots
    maximum_limit: float = max(
        abs(min(all_shots.real)),
        abs(max(all_shots.real)),
        abs(min(all_shots.imag)),
        abs(max(all_shots.imag)),
    )
    # Indicates order of magnitude
    base_limit = 10 ** np.ceil(np.log10(maximum_limit))

    # Default
    axes_limit: float = base_limit * 1.0
    possible_axes_limits: List[float] = [0.15, 0.25, 0.5, 0.75, 1.0]
    for possible_axes_limit in possible_axes_limits:
        margin_percentage: float = 0.9  # Maximum (axes) limit should be less than 90% of total axes limit
        if maximum_limit < base_limit * margin_percentage * possible_axes_limit:
            axes_limit = base_limit * possible_axes_limit
            break
    return -axes_limit, +axes_limit, -axes_limit, +axes_limit


def plot_state_shots(state_classifier: IStateAcquisitionContainer, **kwargs) -> IFigureAxesPair:
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

    for state in state_classifier.contained_states:
        state_acquisition = state_classifier.get_state_acquisition(state=state)
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

    # Find the closest intersection to other_point (with respect to center)
    closest_intersection = None
    for intersection in potential_intersections:
        distance_to_other = euclidean_distance(intersection, other_point)
        distance_to_center = euclidean_distance(intersection, start_point)
        if distance_to_other < distance_to_center:
            closest_intersection = intersection

    return closest_intersection if closest_intersection else other_point


def get_axes_intersection_lookup(decision_boundaries: DecisionBoundaries, ax: plt.Axes) -> Dict[DirectedStateBoundaryKey, Vec2D]:
    """
    Creates a lookup for intersection points of decision boundaries with axes.

    :param decision_boundaries: Decision boundaries of states.
    :param ax: The axes on which the boundaries are drawn.
    :return: Dictionary mapping boundary keys to intersection points.
    """
    # Data allocation
    result: Dict[DirectedStateBoundaryKey, Vec2D] = {}
    center: Vec2D = decision_boundaries.mean
    boundary_keys: List[StateBoundaryKey] = list(decision_boundaries.boundary_lookup.keys())

    continues_boundary: bool = len(boundary_keys) == 1
    if continues_boundary:
        boundary_key: StateBoundaryKey = boundary_keys[0]
        # Main intersection
        boundary_point: Vec2D = decision_boundaries.get_boundary(key=boundary_key)
        intersection_point: Vec2D = find_axes_intersection(start_point=center, other_point=boundary_point, ax=ax)
        directed_boundary_key: DirectedStateBoundaryKey = DirectedStateBoundaryKey(state_a=boundary_key.state_a, state_b=boundary_key.state_b)
        result[directed_boundary_key] = intersection_point
        # Opposite (rotated) intersection
        rotated_boundary_point = rotation_point_180_degrees(boundary_point, center)
        opposite_intersection_point: Vec2D = find_axes_intersection(start_point=center, other_point=rotated_boundary_point, ax=ax)
        opposite_directed_boundary_key: DirectedStateBoundaryKey = DirectedStateBoundaryKey(state_a=boundary_key.state_b, state_b=boundary_key.state_a)
        result[opposite_directed_boundary_key] = opposite_intersection_point
    else:
        for boundary_key in boundary_keys:
            # Main intersection
            boundary_point: Vec2D = decision_boundaries.get_boundary(key=boundary_key)
            intersection_point: Vec2D = find_axes_intersection(start_point=center, other_point=boundary_point, ax=ax)
            directed_boundary_key: DirectedStateBoundaryKey = DirectedStateBoundaryKey(state_a=boundary_key.state_a, state_b=boundary_key.state_b)
            opposite_directed_boundary_key: DirectedStateBoundaryKey = DirectedStateBoundaryKey(state_a=boundary_key.state_b, state_b=boundary_key.state_a)
            # Insert same intersection point for both directions (mimics non-directional lookup)
            result[directed_boundary_key] = intersection_point
            result[opposite_directed_boundary_key] = intersection_point

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
    # Guard clause, if cross product is very small (aka vec1 and vec2 are almost parallel), set True.
    if abs(cross_product) < 1e-16:
        smaller_angle_is_ccw = True

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


def clip_line_with_bounds(line_point1: Vec2D, line_point2: Vec2D, min_x: float, max_x: float, min_y: float, max_y: float) -> Tuple[Vec2D, Vec2D]:
    """
    Clips the coordinates of a line to fit within a bounding box while maintaining the line's direction.

    :param line_point1: First point as Vec2D.
    :param line_point2: Second point as Vec2D.
    :param min_x: Minimum x boundary.
    :param max_x: Maximum x boundary.
    :param min_y: Minimum y boundary.
    :param max_y: Maximum y boundary.
    :return: Clipped coordinates of the line as (line_point1, line_point2).
    """

    def clip(val, min_val, max_val):
        return max(min(val, max_val), min_val)

    # Clip the points to the min and max boundaries
    x1_clipped = clip(line_point1.x, min_x, max_x)
    y1_clipped = clip(line_point1.y, min_y, max_y)
    x2_clipped = clip(line_point2.x, min_x, max_x)
    y2_clipped = clip(line_point2.y, min_y, max_y)

    # Adjust the points to maintain the line if any were clipped
    if line_point1.x != x1_clipped or line_point1.y != y1_clipped or line_point2.x != x2_clipped or line_point2.y != y2_clipped:
        # Calculate slope
        dx = line_point2.x - line_point1.x
        dy = line_point2.y - line_point1.y
        if dx != 0:
            slope = dy / dx
            intercept = line_point1.y - slope * line_point1.x
        else:
            slope = None  # Vertical line
            intercept = line_point1.x  # x-intercept for vertical lines

        # Clip the x-values first and recalculate y if possible
        if slope is not None:
            if x1_clipped != line_point1.x:
                y1_clipped = slope * x1_clipped + intercept
                y1_clipped = clip(y1_clipped, min_y, max_y)
            if x2_clipped != line_point2.x:
                y2_clipped = slope * x2_clipped + intercept
                y2_clipped = clip(y2_clipped, min_y, max_y)
        else:  # For vertical lines, clip y-values directly
            y1_clipped = clip(line_point1.y, min_y, max_y)
            y2_clipped = clip(line_point2.y, min_y, max_y)

    return Vec2D(x1_clipped, y1_clipped), Vec2D(x2_clipped, y2_clipped)


def clip_line(line_point1: Vec2D, line_point2: Vec2D, ax: plt.Axes) -> Tuple[Vec2D, Vec2D]:
    """
    Clips the coordinates of a line to fit within a bounding box while maintaining the line's direction.

    :param line_point1: First point as Vec2D.
    :param line_point2: Second point as Vec2D.
    :param ax: Matplotlib Axes object to get the boundaries from.
    :return: Clipped coordinates of the line as (line_point1, line_point2).
    """
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    return clip_line_with_bounds(line_point1, line_point2, min_x, max_x, min_y, max_y)


def rotation_point_180_degrees(point: Vec2D, center: Vec2D) -> Vec2D:
    """:return: Rotated (180 degree) point around center."""
    translated_point: Vec2D = point - center
    rotated_point: Vec2D = - 1 * translated_point
    result_point: Vec2D = rotated_point + center
    return result_point


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
    boundary_intersections: Dict[DirectedStateBoundaryKey, Vec2D] = get_axes_intersection_lookup(decision_boundaries=decision_boundaries, ax=ax)

    # Store the current limits
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()
    intersection_points: List[Vec2D] = []
    two_state_classification: bool = len(boundary_keys) == 1
    if two_state_classification:
        for boundary_key in boundary_keys:
            intersection_points.extend([
                boundary_intersections[DirectedStateBoundaryKey(boundary_key.state_a, boundary_key.state_b)],
                boundary_intersections[DirectedStateBoundaryKey(boundary_key.state_b, boundary_key.state_a)]
            ])
    else:
        for boundary_key in boundary_keys:
            intersection_points.extend([
                boundary_intersections[DirectedStateBoundaryKey(boundary_key.state_a, boundary_key.state_b)],
            ])

    # Clip intersection points
    for i, intersection_point in enumerate(intersection_points):
        _, clipped_intersection_point = clip_line_with_bounds(
            line_point1=center,
            line_point2=intersection_point,
            min_x=original_xlim[0],
            max_x=original_xlim[1],
            min_y=original_ylim[0],
            max_y=original_ylim[1],
        )
        intersection_points[i] = clipped_intersection_point  # Update intersection points

    for intersection_point in intersection_points:
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


def plot_decision_region(state_classifier: IStateAcquisitionContainer, **kwargs) -> IFigureAxesPair:
    """
    Plots decision regions for state classification.

    :param state_classifier: Container with state classification data.
    :param kwargs: Additional keyword arguments for plot customization.
    :return: Tuple containing the figure and axes of the plot.
    """
    # Data allocation
    decision_boundaries: DecisionBoundaries = state_classifier.classification_boundaries
    center: Vec2D = decision_boundaries.mean
    boundary_keys: List[StateBoundaryKey] = list(decision_boundaries.boundary_lookup.keys())

    # Figures and Axes
    fig, ax = construct_subplot(**kwargs)
    boundary_intersections: Dict[DirectedStateBoundaryKey, Vec2D] = get_axes_intersection_lookup(
        decision_boundaries=decision_boundaries,
        ax=ax,
    )
    two_state_classification: bool = len(boundary_keys) == 1

    # Store the current limits
    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    rectangle_vertices: List[Vec2D] = get_axes_vertices(ax=ax)
    for state in state_classifier.contained_states:
        color = STATE_COLORMAP[state](1.0)

        neighbor_boundary_keys: List[StateBoundaryKey] = get_neighboring_boundary_keys(
            state=state,
            boundary_keys=boundary_keys,
        )
        if two_state_classification:
            boundary_key: StateBoundaryKey = neighbor_boundary_keys[0]
            opposite_state = boundary_key.state_a if state == boundary_key.state_b else boundary_key.state_b
            boundary1 = DirectedStateBoundaryKey(state_a=state, state_b=opposite_state)
            boundary2 = DirectedStateBoundaryKey(state_a=opposite_state, state_b=state)
        else:
            boundary1 = DirectedStateBoundaryKey(state_a=neighbor_boundary_keys[0].state_a, state_b=neighbor_boundary_keys[0].state_b)
            boundary2 = DirectedStateBoundaryKey(state_a=neighbor_boundary_keys[1].state_a, state_b=neighbor_boundary_keys[1].state_b)
        intersection1: Vec2D = boundary_intersections[boundary1]
        intersection2: Vec2D = boundary_intersections[boundary2]

        intersection1, intersection2 = clip_line_with_bounds(
            line_point1=intersection1,
            line_point2=intersection2,
            min_x=original_xlim[0],
            max_x=original_xlim[1],
            min_y=original_ylim[0],
            max_y=original_ylim[1],
        )
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


def plot_state_classification(state_classifier: IStateAcquisitionContainer, use_binary_classification: bool = False, **kwargs) -> IFigureAxesPair:
    """
    Creates a plot visualizing state classification and decision boundaries.

    :param state_classifier: Container with state classification data.
    :param kwargs: Additional keyword arguments for plot customization.
    :return: Tuple containing the figure and axes of the plot.
    """
    if use_binary_classification:
        state_classifier = StateAcquisitionContainer.from_state_acquisitions(
            acquisitions=[
                state_classifier.get_state_acquisition(state=StateKey.STATE_0),
                state_classifier.get_state_acquisition(state=StateKey.STATE_1),
            ]
        )

    decision_boundaries: DecisionBoundaries = state_classifier.classification_boundaries
    kwargs[SubplotKeywordEnum.AXES_FORMAT.value] = IQAxesFormat()
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = LabelFormat(
        x_label='Integrated voltage I [V]',
        y_label='Integrated voltage Q [V]',
    )
    # Figure and Axes
    fig, ax = construct_subplot(**kwargs)
    kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
    plot_state_shots(state_classifier=state_classifier, **kwargs)
    plot_decision_boundary(decision_boundaries=decision_boundaries, **kwargs)
    fig, ax = plot_decision_region(state_classifier=state_classifier, **kwargs)
    ax.legend(frameon=False)
    return fig, ax
