# -------------------------------------------
# Module for visualizing logical fidelity and error rate.
# -------------------------------------------
from dataclasses import dataclass, field
import itertools
from typing import List, Optional, Tuple, Type, Dict
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from qce_circuit.language import InitialStateContainer
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID
from qce_interp.utilities.custom_exceptions import ZeroClassifierShotsException
from qce_interp.interface_definitions.intrf_syndrome_decoder import IDecoder
from qce_interp.decoder_examples.mwpm_decoders import (
    MWPMDecoder,
    MWPMDecoderFast,
)
from qce_interp.decoder_examples.majority_voting import MajorityVotingDecoder
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


@dataclass(frozen=True)
class DecoderToLabel:
    """
    Data class, containing decoder class or instance to label.
    """
    default_label: str = "Unlabeled"
    decoder_type_to_label: Dict[Type[IDecoder], str] = field(default_factory=dict)
    decoder_instance_to_label: Dict[IDecoder, str] = field(default_factory=dict)

    # region Class Methods
    def __post_init__(self):
        default_decoder_type_to_label: Dict[Type[IDecoder], str] = {
            MWPMDecoder: "MWPM",
            MWPMDecoderFast: "MWPM (optimized)",
            MajorityVotingDecoder: "MV",
        }
        default_decoder_type_to_label.update(self.decoder_type_to_label)
        object.__setattr__(self, 'decoder_type_to_label', default_decoder_type_to_label)

    def to_label(self, decoder: IDecoder) -> str:
        """:return: label based on decoder instance, else based on decoder type, else default label."""
        if decoder in self.decoder_instance_to_label:
            return self.decoder_instance_to_label[decoder]
        decoder_type = type(decoder)
        if decoder_type in self.decoder_type_to_label:
            return self.decoder_type_to_label[decoder_type]
        return f"{decoder.__class__.__name__}"
    # endregion


@dataclass(frozen=True)
class DecoderToColor:
    """
    Data class, containing decoder class or instance to color.
    """
    default_color: str = "grey"
    decoder_type_to_color: Dict[Type[IDecoder], str] = field(default_factory=dict)
    decoder_instance_to_color: Dict[IDecoder, str] = field(default_factory=dict)

    # region Class Methods
    def __post_init__(self):
        default_decoder_type_to_color: Dict[Type[IDecoder], str] = {
            MWPMDecoder: orange_red_purple_shades[0],
            MWPMDecoderFast: orange_red_purple_shades[0],
            MajorityVotingDecoder: "grey",
        }
        default_decoder_type_to_color.update(self.decoder_type_to_color)
        object.__setattr__(self, 'decoder_type_to_color', default_decoder_type_to_color)

    def to_color(self, decoder: IDecoder) -> str:
        """:return: label based on decoder instance, else based on decoder type, else default label."""
        if decoder in self.decoder_instance_to_color:
            return self.decoder_instance_to_color[decoder]
        decoder_type = type(decoder)
        if decoder_type in self.decoder_type_to_color:
            return self.decoder_type_to_color[decoder_type]
        return self.default_color
    # endregion


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


def plot_fidelity(decoder: IDecoder, included_rounds: List[int], target_state: InitialStateContainer, target_state_order: Optional[List[IQubitID]] = None, label: Optional[str] = None, fit_error_rate: bool = False, **kwargs) -> IFigureAxesPair:
    """
    :param decoder: Decoder used to evaluate fidelity at each QEC-round.
    :param included_rounds: Array-like of included QEC-rounds. Each round will be evaluated.
    :param target_state: InitialStateContainer instance representing target state.
    :param target_state_order: (Optional) list of ordered qubit-ID's, used to construct the target state.
    :param fit_error_rate: (Optional) Boolean whether or not to fit the logical error rate to fidelity values.
    :param label: (Optional) Label passed to plot constructor.
    :param kwargs: Key-word arguments passed to subplot constructor.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    x_array: np.ndarray = np.asarray(included_rounds)
    y_array: np.ndarray = np.full_like(x_array, np.nan, dtype=np.float32)
    y_err_array: np.ndarray = np.full_like(x_array, np.nan, dtype=np.float32)
    for i, x in tqdm(enumerate(x_array), desc=f"Processing {decoder.__class__.__name__} Decoder", total=len(x_array)):
        try:
            value: float = decoder.get_fidelity(x, target_state=target_state.as_ordered_array(qubit_order=target_state_order))
            value_err: float = decoder.get_fidelity_uncertainty(x, target_state=target_state.as_ordered_array(qubit_order=target_state_order))
        except ZeroClassifierShotsException:
            value = np.nan
            value_err = np.nan
        y_array[i] = value
        y_err_array[i] = value_err

    color: str = kwargs.pop('color', orange_red_purple_shades[0])
    label_format: LabelFormat = kwargs.get(SubplotKeywordEnum.LABEL_FORMAT.value, LabelFormat(
        x_label='QEC-Rounds',
        y_label='Logical fidelity'
    ))

    # Plotting
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    fig, ax = construct_subplot(**kwargs)

    ax.errorbar(
        x_array,
        y_array,
        yerr=y_err_array,
        linestyle='-',
        marker='.',
        color=color,
        label=label,
        capsize=3,
    )
    contains_nan_values: bool = np.isnan(y_array).any()
    if fit_error_rate and not contains_nan_values:
        code_distance: int = len(target_state.as_ordered_array(qubit_order=target_state_order))
        exclude_first_n: int = code_distance
        if code_distance < 5:
            exclude_first_n = 2 * code_distance
        try:
            args, kwargs = get_fit_plot_arguments(x_array=x_array, y_array=y_array, exclude_first_n=exclude_first_n)
            ax.errorbar(
                *args,
                yerr=0.0,
                **kwargs,
            )
        except RuntimeError:
            pass

    ax.set_xlim([-0.1, ax.get_xlim()[1]])
    ax.set_ylim([0.45, 1.02])
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    return fig, ax


def plot_compare_fidelity(decoders: List[IDecoder], included_rounds: List[int], target_state: InitialStateContainer, target_state_order: Optional[List[IQubitID]] = None, decoder_labels: DecoderToLabel = DecoderToLabel(), decoder_colors: DecoderToColor = DecoderToColor(), **kwargs) -> IFigureAxesPair:
    """
    Plots multiple decoders fidelity in one subplot.
    :param decoders: Decoder used to evaluate fidelity at each QEC-round.
    :param included_rounds: Array-like of included QEC-rounds. Each round will be evaluated.
    :param target_state: InitialStateContainer instance representing target state.
    :param target_state_order: (Optional) list of ordered qubit-ID's, used to construct the target state.
    :param decoder_labels: (Optional) Translation from decoder to label.
    :param kwargs: Key-word arguments passed to subplot constructor.
    :return: Tuple of Figure and Axes pair.
    """
    color_cycle = itertools.cycle(orange_red_purple_shades)

    fig, ax = construct_subplot(**kwargs)
    for decoder in decoders:
        kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
        kwargs['color'] = decoder_colors.to_color(decoder)  # next(color_cycle)
        fig, ax = plot_fidelity(
            decoder=decoder,
            included_rounds=included_rounds,
            target_state=target_state,
            target_state_order=target_state_order,
            label=decoder_labels.to_label(decoder=decoder),
            fit_error_rate=True,
            **kwargs,
        )
    return fig, ax
