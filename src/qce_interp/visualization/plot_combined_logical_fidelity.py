# -------------------------------------------
# Module for visualizing logical fidelity and error rate.
# With additional details to defect rates and post selection.
# -------------------------------------------
from typing import List, Callable
from qce_interp.interface_definitions.intrf_error_identifier import IErrorDetectionIdentifier
from qce_interp.interface_definitions.intrf_syndrome_decoder import IDecoder
from qce_interp.decoder_examples.mwpm_decoders import MWPMDecoder
from qce_circuit.library.repetition_code.circuit_components import IRepetitionCodeDescription
from qce_circuit.language.intrf_declarative_circuit import InitialStateContainer
from qce_interp.visualization.plot_post_selection_fraction import plot_retained_fraction_composite
from qce_interp.visualization.plot_defect_rate import plot_all_defect_rate
from qce_interp.visualization.plot_logical_fidelity import (
    plot_fidelity,
    plot_compare_fidelity,
)
from qce_interp.visualization.plotting_functionality import (
    IFigureAxesPair,
    SubplotKeywordEnum,
    LabelFormat,
    AxesFormat,
)
import matplotlib.pyplot as plt


def plot_combined_overview(decoder: IDecoder, error_identifier: IErrorDetectionIdentifier, decoder_constructor: Callable[[IErrorDetectionIdentifier], IDecoder], qec_rounds: List[int], target_state: InitialStateContainer, **kwargs) -> IFigureAxesPair:
    """
    :param decoder: Decoder used to evaluate fidelity at each QEC-round.
    :param error_identifier: Instance that identifiers errors.
    :param decoder_constructor: Constructor function taking IErrorDetectionIdentifier as argument
        to construct the desired decoder used to evaluate under post-selection.
    :param qec_rounds: Array-like of qec cycles that should be included in the post-selection plot.
    :param target_state: InitialStateContainer instance representing target state.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    decoder_post_selected = decoder_constructor(error_identifier.copy_with_post_selection(
        use_heralded_post_selection=True,
        use_projected_leakage_post_selection=True,
        use_stabilizer_leakage_post_selection=True,
    ))
    decoder_post_selected_final = decoder_constructor(error_identifier.copy_with_post_selection(
        use_heralded_post_selection=True,
        use_projected_leakage_post_selection=True,
        use_stabilizer_leakage_post_selection=False,
    ))

    # Define grid details
    gridspec_constructor_kwargs: dict = dict(height_ratios=[1, 1, 1], width_ratios=[1])
    # Define the mosaic layout
    ax0_label: str = 'logical_error'
    ax1_label: str = 'defect_rate'
    ax2_label: str = 'post_selection'
    kwargs[SubplotKeywordEnum.FIGURE_SIZE.value] = kwargs.get(
        SubplotKeywordEnum.FIGURE_SIZE.value,
        (5, 7),
    )
    fig, axs_dict = plt.subplot_mosaic(
        [
            [ax0_label],
            [ax1_label],
            [ax2_label],
        ],
        gridspec_kw=gridspec_constructor_kwargs,
        constrained_layout=False,  # Set to False to allow manual adjustment of spacing
        sharex=True,
        **kwargs,
    )
    ax0 = axs_dict[ax0_label]
    ax1 = axs_dict[ax1_label]
    ax2 = axs_dict[ax2_label]

    # Set shared x-axis
    # ax1.get_shared_x_axes().join(ax0, ax1, ax2)

    # Hide x-labels for ax0 and ax1
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)

    # Adjust spacing between subplots to 0
    fig.subplots_adjust(hspace=0.05)

    kwargs_logical = {
        SubplotKeywordEnum.HOST_AXES.value: (fig, ax0),
        SubplotKeywordEnum.AXES_FORMAT.value: kwargs.get(
            SubplotKeywordEnum.AXES_FORMAT.value,
            AxesFormat(),
        ),
        SubplotKeywordEnum.LABEL_FORMAT.value: LabelFormat(
            x_label='',
            y_label='Logical fidelity'
        ),
    }
    plot_compare_fidelity(
        decoders=[
            decoder,
        ],
        included_rounds=qec_rounds,
        target_state=target_state,
        **kwargs_logical,
    )
    plot_fidelity(
        decoder=decoder_post_selected_final,
        included_rounds=qec_rounds,
        target_state=target_state,
        color='cyan',
        label='PS final-meas leakage',
        fit_error_rate=True,
        **kwargs_logical
    )
    plot_fidelity(
        decoder=decoder_post_selected,
        included_rounds=qec_rounds,
        target_state=target_state,
        color='magenta',
        label='PS final/parity-meas leakage',
        fit_error_rate=True,
        **kwargs_logical
    )

    kwargs_defect = {
        SubplotKeywordEnum.HOST_AXES.value: (fig, ax1),
        SubplotKeywordEnum.AXES_FORMAT.value: kwargs.get(
            SubplotKeywordEnum.AXES_FORMAT.value,
            AxesFormat(),
        ),
        SubplotKeywordEnum.LABEL_FORMAT.value: LabelFormat(
            x_label='',
            y_label=r'Defect rate $\langle d_i \rangle$',
        ),
    }
    plot_all_defect_rate(
        error_identifier=error_identifier,
        included_rounds=qec_rounds[-1],
        **kwargs_defect,
    )
    kwargs_postselect = {
        SubplotKeywordEnum.HOST_AXES.value: (fig, ax2),
        SubplotKeywordEnum.AXES_FORMAT.value: kwargs.get(
            SubplotKeywordEnum.AXES_FORMAT.value,
            AxesFormat(),
        )
    }
    plot_retained_fraction_composite(
        error_identifier=error_identifier,
        qec_rounds=qec_rounds,
        **kwargs_postselect,
    )
    return fig, ax0
