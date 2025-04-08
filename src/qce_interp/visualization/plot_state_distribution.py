# -------------------------------------------
# Module for visualizing state after QEC cycles.
# -------------------------------------------
import numpy as np
from numpy.typing import NDArray
from typing import Callable, List, Any
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import ISyndromeDecoder
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    EmptyAxesFormat,
    SubplotKeywordEnum,
    LabelFormat,
)
from matplotlib.ticker import MultipleLocator


def generate_gray_code(n: int) -> NDArray[np.int_]:
    """
    Generate a 2^n by n array of Gray codes.

    :param n: The number of bits in each Gray code.
    :return: A 2^n by n array of Gray codes.
    """
    num_codes = 1 << n
    gray_codes = np.array([(i >> 1) ^ i for i in range(num_codes)], dtype=int)
    gray_code_array = ((gray_codes[:, None] & (1 << np.arange(n))) > 0).astype(int)
    gray_code_array = np.fliplr(gray_code_array)

    return gray_code_array


def calculate_state_fractions(large_array: NDArray[np.int_], gray_code_array: NDArray[np.int_]) -> NDArray[np.float32]:
    """
    Calculate the fraction of each Gray code state in the large M by N array.

    :param large_array: The M by N array to compare against Gray codes.
    :param gray_code_array: The 2^N by N array of Gray codes.
    :return: A 2^N by 1 array of fractions.
    """
    fractions = np.zeros(len(gray_code_array))
    for i, gray_code in enumerate(gray_code_array):
        matches = np.all(large_array == gray_code, axis=1).sum()
        fractions[i] = matches / len(large_array)

    return fractions


def get_state_fraction_array(
        classification_provider: Any,
        rounds: List[int],
        gray_code_n: int,
        classification_method: Callable[[Any, int], NDArray[np.int_]]
) -> NDArray[np.float32]:
    """
    Calculate the state fraction array for given rounds using a specified classification method.

    :param classification_provider: An instance that provides classification..
    :param rounds: A list of rounds for which to calculate state fractions.
    :param gray_code_n: The Gray code length.
    :param classification_method: A function that takes a LookupTableDecoder and an int (round number) and returns a binary classification array.
    :return: A 2D array representing state fractions for each round.
    """
    gray_code_array = generate_gray_code(gray_code_n)
    state_fraction_array = np.zeros((len(rounds), len(gray_code_array)))

    for i, round in enumerate(rounds):
        binary_classification = classification_method(classification_provider, round)
        n, _, d = binary_classification.shape
        binary_classification = binary_classification.reshape((n, d))
        state_fractions = calculate_state_fractions(binary_classification, gray_code_array)
        state_fraction_array[i, :] = state_fractions

    return state_fraction_array


def _plot_state_evolution(state_fraction_array: np.ndarray, gray_code_array: np.ndarray, **kwargs) -> IFigureAxesPair:
    """:return: State distribution per QEC-round."""

    kwargs[SubplotKeywordEnum.AXES_FORMAT.value] = EmptyAxesFormat()
    fig, ax = construct_subplot(**kwargs)

    # Flip the array in the vertical direction and plot matrix
    num_rows, num_cols = state_fraction_array.shape
    offset_extent = [-0.5, num_cols - 0.5, -0.5, num_rows - 0.5]
    im1 = ax.matshow(state_fraction_array, cmap='Blues', vmin=0, vmax=None, origin='lower', extent=offset_extent)

    # Set major and minor locators for the x-axis
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    # Set the major tick labels with binary strings and rotate them vertically
    binary_strings = [' '.join(map(str, state)) for state in gray_code_array]
    # Set major tick positions and labels
    ax.set_xticks(range(len(binary_strings)))
    ax.set_xticklabels(binary_strings, rotation='vertical')

    # Set x-ticks to the bottom
    ax.xaxis.set_ticks_position('bottom')

    # Set minor locator for the y-axis
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    # Enable grid for minor ticks
    ax.grid(which='both', axis='both', linestyle=':', color='gray', alpha=0.5)
    return fig, ax


def plot_state_evolution(syndrome_decoder: ISyndromeDecoder, target_state: np.ndarray, included_rounds: List[int], **kwargs) -> IFigureAxesPair:
    """:return: State distribution per QEC-round. Note: currently hardcodes highlighted states."""
    state_size: int = len(target_state)
    gray_code_array = generate_gray_code(n=state_size)
    state_fraction_array: np.ndarray = get_state_fraction_array(
        classification_provider=syndrome_decoder,
        rounds=included_rounds,
        gray_code_n=gray_code_array.shape[1],
        classification_method=lambda decoder, count: decoder.get_binary_projected_corrected(count),
    )

    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = LabelFormat(x_label='', y_label='Rounds')
    fig, ax = _plot_state_evolution(
        state_fraction_array=state_fraction_array,
        gray_code_array=gray_code_array,
        **kwargs,
    )
    # Highlight specific labels
    highlight_labels = ['0 1 0 1 0', '1 0 1 0 1']
    for label in ax.get_xticklabels():
        if label.get_text() in highlight_labels:
            label.set_fontweight('bold')
    ax.set_title('Computational parity and refocusing corrected State')
    return fig, ax
