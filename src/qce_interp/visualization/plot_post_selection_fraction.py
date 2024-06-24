# -------------------------------------------
# Module for visualizing post-selection fraction.
# -------------------------------------------
import numpy as np
from numpy.typing import NDArray
from typing import List
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    SubplotKeywordEnum,
    LabelFormat,
)


def plot_post_selection_fraction(error_identifier: IErrorDetectionIdentifier, qec_rounds: List[int], **kwargs) -> IFigureAxesPair:
    """
    :param error_identifier: Instance that identifiers errors.
    :param qec_rounds: Array-like of qec cycles that should be included in the post-selection plot.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """

    # Data allocation
    qec_rounds: np.ndarray = np.asarray(qec_rounds)
    ps_heralded_fraction = np.zeros_like(qec_rounds, dtype=np.float_)
    ps_data_fraction = np.zeros_like(qec_rounds, dtype=np.float_)
    ps_ancilla_fraction = np.zeros_like(qec_rounds, dtype=np.float_)

    heralded_error_identifier = error_identifier.copy_with_post_selection(
        use_heralded_post_selection=True,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=False,
    )
    heralded_data_error_identifier = error_identifier.copy_with_post_selection(
        use_heralded_post_selection=True,
        use_projected_leakage_post_selection=True,
        use_stabilizer_leakage_post_selection=False,
    )
    heralded_data_ancilla_error_identifier = error_identifier.copy_with_post_selection(
        use_heralded_post_selection=False,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=True,
    )

    for i, qec_round in enumerate(qec_rounds):
        ps_mask_heralded: NDArray[np.bool_] = heralded_error_identifier.get_post_selection_mask(cycle_stabilizer_count=qec_round)
        ps_heralded_fraction[i] = (1 - np.sum(ps_mask_heralded) / len(ps_mask_heralded))

        ps_mask_data: NDArray[np.bool_] = heralded_data_error_identifier.get_post_selection_mask(cycle_stabilizer_count=qec_round)
        ps_data_fraction[i] = (1 - np.sum(ps_mask_data) / len(ps_mask_data))

        ps_mask_ancilla: NDArray[np.bool_] = heralded_data_ancilla_error_identifier.get_post_selection_mask(cycle_stabilizer_count=qec_round)
        ps_ancilla_fraction[i] = (1 - np.sum(ps_mask_ancilla) / len(ps_mask_ancilla))

    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = kwargs.get(
        SubplotKeywordEnum.LABEL_FORMAT.value,
        LabelFormat(
            x_label='QEC round',
            y_label='Post-selection rejection fraction.',
        ),
    )
    fig, ax = construct_subplot(**kwargs)

    ax.plot(
        qec_rounds,
        ps_heralded_fraction,
        '.-',
        label='Heralded fraction',
    )
    ax.plot(
        qec_rounds,
        ps_data_fraction,
        '.-',
        label='Leakage (Data) fraction',
    )
    ax.plot(
        qec_rounds,
        ps_ancilla_fraction,
        '.-',
        label='Leakage (Ancilla) fraction',
    )
    ax.plot(
        qec_rounds,
        ps_heralded_fraction + ps_ancilla_fraction,
        '--',
        color='k',
        label='Leakage (Ancilla) + Heralded fraction',
    )

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0.0, 1.0)

    return fig, ax
