# -------------------------------------------
# Module for visualizing state after QEC cycles.
# -------------------------------------------
import numpy as np
from typing import List, Dict
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    EmptyAxesFormat,
    SubplotKeywordEnum,
    LabelFormat,
)


def calculate_correlation_element(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """Source: https://arxiv.org/pdf/1712.02360.pdf"""
    numerator: float = (np.mean(x_i * x_j) - np.mean(x_i) * np.mean(x_j))
    denominator = 1 - 2 * np.mean((x_i + x_j) % 2)
    return 0.5 - np.sqrt(1 / 4 - numerator / denominator)


def calculate_correlation_matrix(identifier: IErrorDetectionIdentifier, rounds: List[int]) -> np.ndarray:
    """:return: Correlation matrix calculated based on binary defect lookup for each qubit at a fixed (max) number of rounds."""
    round_max: int = max(rounds)
    binary_classification_lookup: Dict[IQubitID, np.ndarray] = identifier.get_defect_stabilizer_lookup(
        cycle_stabilizer_count=round_max
    )
    qubit_ids: List[IQubitID] = list(binary_classification_lookup.keys())
    count_qubit_ids: int = len(qubit_ids)
    result: np.ndarray = np.zeros((count_qubit_ids * round_max, count_qubit_ids * round_max))

    # Order of ancilla qubits in Pij matrix
    for qi, qubit_id_i in enumerate(qubit_ids):
        for i in range(round_max):
            for qj, qubit_id_j in enumerate(qubit_ids):
                for j in range(round_max):
                    index_i: int = i + round_max * qi
                    index_j: int = j + round_max * qj

                    if (i == j) and (qi == qj):
                        result[index_i, index_j] = np.nan
                    elif i + round_max * qi > j + round_max * qj:
                        result[index_i, index_j] = np.nan
                    elif i == 0 or j == 0:
                        result[index_i, index_j] = np.nan
                    else:
                        xi: np.ndarray = binary_classification_lookup[qubit_id_i][:, i]
                        xj: np.ndarray = binary_classification_lookup[qubit_id_j][:, j]
                        result[index_i, index_j] = calculate_correlation_element(x_i=xi, x_j=xj)
    return result


# TODO: This method is from old analysis, requires reformatting. Unreadable.
def _plot_pij_matrix(pij_matrix: np.ndarray, ordered_qubit_ids: List[IQubitID], **kwargs) -> IFigureAxesPair:
    number_of_qubits: int = len(ordered_qubit_ids)

    fig, ax = construct_subplot(**kwargs)
    # Plot matrix
    im1 = ax.matshow(np.abs(pij_matrix).T, cmap='Blues', vmin=0, vmax=None)
    im2 = ax.matshow(np.abs(pij_matrix), cmap='Reds', vmin=0, vmax=0.05)
    # Set ticks

    R = int(pij_matrix.shape[0] / number_of_qubits)
    ticks_array = np.arange(0, number_of_qubits*R, R)-.5

    ax.set_xticks(ticks_array)
    ax.set_yticks(ticks_array)
    from matplotlib.ticker import MultipleLocator
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Write qubit labels
    ordered_qubit_names: List[str] = [qubit_id.id for qubit_id in ordered_qubit_ids]
    for i, q in enumerate(ordered_qubit_names):
        ax.text(i*R+(R-1)/2, -3.5, q, va='center', ha='center', size=12)
        ax.text(-3.5, i*R+(R-1)/2, q, va='center', ha='center', size=12)
    for i in range(R):
        for j in range(number_of_qubits):
            ax.text(-1.5, i+R*j, i, va='center', ha='center', size=5.5)
            ax.text(i+R*j, -1.5, i, va='center', ha='center', size=5.5)
    # Plot tick lines
    for i in range(number_of_qubits - 1):
        ax.axhline((i+1)*R-.5, color='gainsboro', alpha=1)
        ax.axvline((i+1)*R-.5, color='gainsboro', alpha=1)
    for i in range(number_of_qubits*R-1):
        ax.axhline((i+1)-.5, color='gainsboro', alpha=1, lw=.75)
        ax.axvline((i+1)-.5, color='gainsboro', alpha=1, lw=.75)
    # Plot colorbar
    cb1 = fig.colorbar(im1, aspect=40)
    cb2 = fig.colorbar(im2, aspect=40)
    pos = cb1.ax.get_position()
    cb1.ax.set_position([pos.x0-.1+.02, pos.y0+.26/2, pos.width, pos.height-0.26])
    cb1.ax.yaxis.set_ticks_position('left')
    pos = cb2.ax.get_position()
    cb2.ax.set_position([pos.x0+0.05+.02, pos.y0+.26/2, pos.width, pos.height-0.26])
    cb2.set_label('$\\mathrm{{P_{{i,j}}}}$ matrix coefficients')
    return fig, ax


# TODO: Relies on specific order in error_identifier.involved_stabilizer_qubit_ids. Should be replaced by Device layout definition.
def plot_pij_matrix(error_identifier: IErrorDetectionIdentifier, included_rounds: List[int], **kwargs) -> IFigureAxesPair:
    matrix: np.ndarray = calculate_correlation_matrix(error_identifier, included_rounds)
    ordered_qubit_ids = error_identifier.involved_stabilizer_qubit_ids

    kwargs[SubplotKeywordEnum.AXES_FORMAT.value] = EmptyAxesFormat()
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = LabelFormat(x_label='', y_label='')
    fig, ax = _plot_pij_matrix(
        pij_matrix=matrix,
        ordered_qubit_ids=ordered_qubit_ids,
        **kwargs,
    )
    return fig, ax
