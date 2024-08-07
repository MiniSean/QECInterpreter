# -------------------------------------------
# Module for visualizing logical fidelity and error rate.
# -------------------------------------------
import itertools
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from qce_circuit.language import InitialStateContainer
from qce_interp.utilities.custom_exceptions import ZeroClassifierShotsException
from qce_interp.interface_definitions.intrf_syndrome_decoder import IDecoder
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    LabelFormat,
    SubplotKeywordEnum,
)


# Define an orange to red to purple color cycler
orange_red_purple_shades = [
    '#ffa500',  # orange
    '#ff8c00',  # dark orange
    '#ff4500',  # orange red
    '#ff0000',  # red
    '#dc143c',  # crimson
    '#800080',  # purple
    '#8a2be2',  # blue violet (to transition towards more purplish shades)
    '#9370db',  # medium purple
]


def fit_function(x: np.ndarray, error: float, x_0: float) -> np.ndarray:
    """
    Calculate the fit function value for given inputs.

    :param x: The independent variable values for which to calculate the function's output.
    :type x: np.ndarray
    :param error: The error parameter of the fit function.
    :type error: float
    :param x_0: The x_0 parameter of the fit function, representing a shift along the x-axis.
    :type x_0: float
    :return: The calculated values of the fit function for each input x.
    :rtype: np.ndarray
    """
    return 0.5 * (1 + (1 - 2 * error) ** (x - x_0))


def get_fit_plot_arguments(x_array: np.ndarray, y_array: np.ndarray, exclude_first_n: int = 0) -> Tuple[tuple, dict]:
    """
    Perform curve fitting on given data arrays and prepare plot arguments.

    :param x_array: The array of x values.
    :type x_array: np.ndarray
    :param y_array: The array of y values corresponding to x_array.
    :type y_array: np.ndarray
    :param exclude_first_n: Number of initial elements to exclude from fitting, defaults to 0.
    :type exclude_first_n: int
    :return: A tuple containing the x values for plotting and a dictionary with plotting arguments,
             including line style, marker, color, and label with fitted parameters.
    :rtype: Tuple[*args, **kwargs]
    """
    # Bounds for the parameters (assuming error is between 0 and 0.5 and x_0 is within some range)
    bounds = ([0, -np.inf], [0.5, np.inf])

    # Exclude the first N points
    x_array_filtered: np.ndarray = x_array[exclude_first_n:]
    y_array_filtered: np.ndarray = y_array[exclude_first_n:]

    # Exclude points where y_array has NaN values
    mask = ~np.isnan(y_array_filtered)
    x_array_filtered = x_array_filtered[mask]
    y_array_filtered = y_array_filtered[mask]

    # Check if there are enough points left after filtering
    min_points: int = 3
    if len(x_array_filtered) < min_points or len(y_array_filtered) < min_points:
        return (np.asarray([]), np.asarray([])), dict()

    # Perform curve fitting
    response = curve_fit(fit_function, x_array_filtered, y_array_filtered, bounds=bounds)
    popt = response[0]
    pcov = response[1]
    fitted_error, fitted_x0 = popt
    perr = np.sqrt(np.diag(pcov))

    # Prepare x values for plotting and calculate fitted function values
    plot_x_values = x_array[exclude_first_n:]
    plot_y_values = fit_function(plot_x_values, *popt)

    # Prepare plotting arguments
    plot_args = dict(
        linestyle='--',
        marker='none',
        color='k',
        label=rf'$\epsilon_L$ = {fitted_error:.2%}$\pm${perr[0]:.2%}',
    )

    return (plot_x_values, plot_y_values), plot_args


def plot_fidelity(decoder: IDecoder, included_rounds: List[int], target_state: InitialStateContainer, label: Optional[str] = None, fit_error_rate: bool = False, **kwargs) -> IFigureAxesPair:
    """
    :param decoder: Decoder used to evaluate fidelity at each QEC-round.
    :param included_rounds: Array-like of included QEC-rounds. Each round will be evaluated.
    :param target_state: InitialStateContainer instance representing target state.
    :param fit_error_rate: (Optional) Boolean whether or not to fit the logical error rate to fidelity values.
    :param label: (Optional) Label passed to plot constructor.
    :param kwargs: Key-word arguments passed to subplot constructor.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    x_array: np.ndarray = np.asarray(included_rounds)
    y_array: np.ndarray = np.full_like(x_array, np.nan, dtype=np.float64)
    for i, x in tqdm(enumerate(x_array), desc=f"Processing {decoder.__class__.__name__} Decoder", total=len(x_array)):
        try:
            value: float = decoder.get_fidelity(x, target_state=target_state.as_array)
        except ZeroClassifierShotsException:
            value = np.nan
        y_array[i] = value

    color: str = kwargs.pop('color', orange_red_purple_shades[0])
    label_format: LabelFormat = kwargs.get(SubplotKeywordEnum.LABEL_FORMAT.value, LabelFormat(
        x_label='QEC-Rounds',
        y_label='Logical fidelity'
    ))

    # Plotting
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    fig, ax = construct_subplot(**kwargs)

    ax.plot(
        x_array,
        y_array,
        linestyle='-',
        marker='.',
        color=color,
        label=label,
    )
    contains_nan_values: bool = np.isnan(y_array).any()
    if fit_error_rate and not contains_nan_values:
        code_distance: int = len(target_state.as_array)
        exclude_first_n: int = code_distance
        if code_distance < 7:
            exclude_first_n = 2 * code_distance

        args, kwargs = get_fit_plot_arguments(x_array=x_array, y_array=y_array, exclude_first_n=exclude_first_n)
        ax.plot(
            *args,
            **kwargs,
        )

    ax.set_xlim([-0.1, ax.get_xlim()[1]])
    ax.set_ylim([0.45, 1.02])
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    return fig, ax


def plot_compare_fidelity(decoders: List[IDecoder], included_rounds: List[int], target_state: InitialStateContainer, **kwargs) -> IFigureAxesPair:
    """
    Plots multiple decoders fidelity in one subplot.
    :param decoders: Decoder used to evaluate fidelity at each QEC-round.
    :param included_rounds: Array-like of included QEC-rounds. Each round will be evaluated.
    :param target_state: InitialStateContainer instance representing target state.
    :param kwargs: Key-word arguments passed to subplot constructor.
    :return: Tuple of Figure and Axes pair.
    """
    color_cycle = itertools.cycle(orange_red_purple_shades)

    fig, ax = construct_subplot(**kwargs)
    for decoder in decoders:
        kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
        kwargs['color'] = next(color_cycle)
        fig, ax = plot_fidelity(
            decoder=decoder,
            included_rounds=included_rounds,
            target_state=target_state,
            label=f"{decoder.__class__.__name__}",
            fit_error_rate=True,
            **kwargs,
        )
    return fig, ax
