# -------------------------------------------
# Module for visualizing logical fidelity and error rate.
# -------------------------------------------
import itertools
from typing import List, Optional, Tuple
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from qce_circuit.language import InitialStateContainer
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
color_cycle = itertools.cycle(orange_red_purple_shades)


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


def get_fit_plot_arguments(x_array: np.ndarray, y_array: np.ndarray, exclude_first_n: int = 0) -> Tuple[np.ndarray, dict]:
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
    :rtype: Tuple[np.ndarray, dict]
    """
    # Bounds for the parameters (assuming error is between 0 and 0.5 and x_0 is within some range)
    bounds = ([0, -np.inf], [0.5, np.inf])

    # Perform curve fitting
    popt, _ = curve_fit(fit_function, x_array[exclude_first_n:], y_array[exclude_first_n:], bounds=bounds)
    fitted_error, fitted_x0 = popt

    # Prepare x values for plotting and calculate fitted function values
    plot_x_values = x_array[exclude_first_n:]
    plot_y_values = fit_function(plot_x_values, *popt)

    # Prepare plotting arguments
    plot_args = dict(
        linestyle='--',
        marker='none',
        color='k',
        # label=rf'$\epsilon_L$ = {fitted_error:.2%}, $x_0$ = {fitted_x0:.2f}',
        label=rf'$\epsilon_L$ = {fitted_error:.2%}',
    )

    return (plot_x_values, plot_y_values), plot_args


def plot_fidelity(decoder: IDecoder, included_rounds: List[int], target_state: InitialStateContainer, label: Optional[str] = None, fit_error_rate: bool = False, **kwargs) -> IFigureAxesPair:
    """
    :param decoder: Decoder used to evaluate fidelity at each QEC-round.
    :param included_rounds: Array-like of included QEC-rounds. Each round will be evaluated.
    :param target_state: InitialStateContainer instance representing target state.
    :param label: (Optional) Label passed to plot constructor.
    :param kwargs: Key-word arguments passed to subplot constructor.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    x_array: np.ndarray = np.asarray(included_rounds)
    y_array: np.ndarray = np.array([
        decoder.get_fidelity(x, target_state=target_state.as_array)
        for x in tqdm(x_array, desc=f"Processing {decoder.__class__.__name__} Decoder")
    ])
    # Plotting
    label_format: LabelFormat = LabelFormat(
        x_label='QEC-Rounds',
        y_label='Logical fidelity'
    )
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    fig, ax = construct_subplot(**kwargs)

    ax.plot(
        x_array,
        y_array,
        linestyle='-',
        marker='.',
        color=next(color_cycle),
        label=label,
    )
    if fit_error_rate:
        args, kwargs = get_fit_plot_arguments(x_array=x_array, y_array=y_array, exclude_first_n=2 * len(target_state.as_array))
        ax.plot(
            *args,
            **kwargs,
        )

    ax.set_xlim([-0.1, ax.get_xlim()[1]])
    ax.set_ylim([0.45, 1.02])
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    return fig, ax
