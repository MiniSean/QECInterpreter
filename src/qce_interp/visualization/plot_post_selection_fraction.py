# -------------------------------------------
# Module for visualizing post-selection fraction.
# -------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
import scipy.optimize as optimize
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    SubplotKeywordEnum,
    LabelFormat,
)


def get_fit_plot_arguments(x_array: np.ndarray, y_array: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], dict]:
    """
    Perform curve fitting on given data arrays and prepare plot arguments.

    :param x_array: The array of x values.
    :type x_array: np.ndarray
    :param y_array: The array of y values corresponding to x_array.
    :type y_array: np.ndarray
    :return: A tuple containing the x values for plotting and a dictionary with plotting arguments,
             including line style, marker, color, and label with fitted parameters.
    :rtype: Tuple[Tuple[np.ndarray, np.ndarray], dict] (Tuple[*args, **kwargs])
    """
    # Define the linear function with a fixed intercept of 1.0 in log scale
    def log_linear_func(x, slope, log_intercept):
        # log_intercept = 0  # log(1.0) = 0
        return slope * x + log_intercept

    # Exclude points where y_array has 0 values
    mask_zero = (y_array != 0)
    x_array_filtered: np.ndarray = x_array[mask_zero]
    y_array_filtered: np.ndarray = y_array[mask_zero]

    # Exclude points where long(y_array) has NaN or INF values
    y_log_array: np.ndarray = np.log(y_array_filtered)
    mask_finite = np.isfinite(y_log_array)
    x_array_filtered = x_array_filtered[mask_finite]
    y_log_array_filtered: np.ndarray = y_log_array[mask_finite]

    # Perform the curve fit
    response = optimize.curve_fit(log_linear_func, x_array_filtered, y_log_array_filtered)
    popt = response[0]
    pcov = response[1]
    slope = popt[0]
    log_intercept = popt[1]
    slope_std_err = np.sqrt(np.diag(pcov))[0]

    # Prepare x values for plotting and calculate fitted function values
    plot_x_values = x_array
    plot_y_values = np.exp(log_linear_func(x_array, slope, log_intercept))

    # Prepare plotting arguments
    plot_args = dict(
        linestyle='--',
        marker='none',
        color='k',
        label=rf"$\alpha$ = {slope:.2%}$\pm${slope_std_err:.2%} per round",
    )

    return (plot_x_values, plot_y_values), plot_args


def plot_post_selection_fraction(error_identifier: IErrorDetectionIdentifier, qec_rounds: List[int], fit_fraction_rate: bool = False, **kwargs) -> IFigureAxesPair:
    """
    :param error_identifier: Instance that identifiers errors.
    :param qec_rounds: Array-like of qec cycles that should be included in the post-selection plot.
    :param fit_fraction_rate: Boolean for fitting linear regression fit to
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    qec_rounds: np.ndarray = np.asarray(qec_rounds)
    fraction = np.zeros_like(qec_rounds, dtype=np.float_)
    for i, qec_round in enumerate(qec_rounds):
        post_selection_mask: NDArray[np.bool_] = error_identifier.get_post_selection_mask(cycle_stabilizer_count=qec_round)
        fraction[i] = np.sum(post_selection_mask) / len(post_selection_mask)
    # Sub kwargs
    plot_kwargs = {}
    for key in ['label', 'color']:
        if key in kwargs:
            plot_kwargs[key] = kwargs[key]

    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = kwargs.get(
        SubplotKeywordEnum.LABEL_FORMAT.value,
        LabelFormat(
            x_label='QEC round',
            y_label='Retained fraction',
        ),
    )
    fig, ax = construct_subplot(**kwargs)
    ax.plot(
        qec_rounds,
        fraction,
        '.-',
        **plot_kwargs,
    )

    contains_nan_values: bool = np.isnan(fraction).any()
    if fit_fraction_rate:  # and not contains_nan_values:
        args, kwargs = get_fit_plot_arguments(x_array=qec_rounds, y_array=fraction)
        ax.plot(
            *args,
            **kwargs,
        )

    return fig, ax


def plot_post_selection_fraction_composite(error_identifier: IErrorDetectionIdentifier, qec_rounds: List[int], **kwargs) -> IFigureAxesPair:
    """
    Combines multiple preset post-selection fractions with labels.
    :param error_identifier: Instance that identifiers errors.
    :param qec_rounds: Array-like of qec cycles that should be included in the post-selection plot.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """

    # Data allocation
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = kwargs.get(
        SubplotKeywordEnum.LABEL_FORMAT.value,
        LabelFormat(
            x_label='QEC round',
            y_label='Retained fraction',
        ),
    )
    fig, ax = construct_subplot(**kwargs)
    kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)

    fig, ax = plot_post_selection_fraction(
        error_identifier=error_identifier.copy_with_post_selection(
            use_heralded_post_selection=True,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=False,
        ),
        qec_rounds=qec_rounds,
        label='Heralded fraction',
        **kwargs,
    )
    plot_post_selection_fraction(
        error_identifier=error_identifier.copy_with_post_selection(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=True,
            use_stabilizer_leakage_post_selection=False,
        ),
        qec_rounds=qec_rounds,
        label='Leakage (Data) fraction',
        **kwargs,
    )
    plot_post_selection_fraction(
        error_identifier=error_identifier.copy_with_post_selection(
            use_heralded_post_selection=False,
            use_projected_leakage_post_selection=False,
            use_stabilizer_leakage_post_selection=True,
        ),
        qec_rounds=qec_rounds,
        fit_fraction_rate=True,
        label='Leakage (Ancilla) fraction',
        **kwargs,
    )

    # Default
    minimum_limit = get_minimum_plotted_value(ax)
    possible_axes_limits: List[float] = [1e-2]
    axes_limit: float = possible_axes_limits[-1]
    for possible_axes_limit in possible_axes_limits:
        margin_percentage: float = 0.9  # Maximum (axes) limit should be less than 90% of total axes limit
        if minimum_limit > margin_percentage * possible_axes_limit:
            axes_limit = margin_percentage * possible_axes_limit
            break

    ax.grid(which='minor', axis='y', alpha=0.5, linestyle='dashed')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(axes_limit, 1.1)
    ax.set_yscale('log')

    return fig, ax


def get_minimum_plotted_value(ax: plt.Axes) -> float:
    result = float('inf')
    for line in ax.get_lines():
        y_data = line.get_ydata()
        result = min(result, np.min(y_data))
    return result
