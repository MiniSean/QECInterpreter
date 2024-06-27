# -------------------------------------------
# Module for visualizing defect rates over all QEC cycles.
# -------------------------------------------
import itertools
import xarray as xr
import numpy as np
import matplotlib.transforms as transforms
from typing import List
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_error_identifier import (
    IErrorDetectionIdentifier,
    DataArrayLabels,
    LabeledErrorDetectionIdentifier,
)
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    LabelFormat,
    SubplotKeywordEnum,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Define a blue color cycler
blue_shades = [
    '#1f77b4',  # matplotlib default blue
    '#add8e6',  # light blue
    '#0000ff',  # blue
    '#00008b',  # dark blue
    '#4169e1',  # royal blue
    '#00bfff',  # deep sky blue
    '#87ceeb'   # sky blue
]

# Define a green color cycler
green_shades = [
    '#008000',  # green
    '#00FF00',  # lime
    '#006400',  # dark green
    '#32CD32',  # lime green
    '#98FB98',  # pale green
    '#00FA9A',  # medium spring green
    '#7CFC00'   # lawn green
]


def plot_post_selection_fraction(error_identifier: IErrorDetectionIdentifier, qubit_id: IQubitID, qec_cycles: int, **kwargs) -> IFigureAxesPair:
    # Data allocation
    labeled_error_identifier: LabeledErrorDetectionIdentifier = LabeledErrorDetectionIdentifier(error_identifier)
    labeled_error_identifier_post_selected: LabeledErrorDetectionIdentifier = labeled_error_identifier.copy_with_post_selection(
        use_heralded_post_selection=labeled_error_identifier.include_heralded_post_selection,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=True,
    )
    data_array: xr.DataArray = labeled_error_identifier.get_labeled_defect_stabilizer_lookup(cycle_stabilizer_count=qec_cycles)[qubit_id]
    data_array_post_selected: xr.DataArray = labeled_error_identifier_post_selected.get_labeled_defect_stabilizer_lookup(cycle_stabilizer_count=qec_cycles)[qubit_id]
    # Calculate sample size across 'measurement_repetition'
    sample_size: int = data_array.sizes[DataArrayLabels.MEASUREMENT.value]
    sample_size_post_selected: int = data_array_post_selected.sizes[DataArrayLabels.MEASUREMENT.value]
    try:
        post_selection_ratio: float = sample_size_post_selected / sample_size
    except ZeroDivisionError:
        post_selection_ratio = np.nan

    # Figure and Axes
    fig, ax = construct_subplot(**kwargs)

    # Draw (post) selection ratio
    transform = transforms.blended_transform_factory(ax.transAxes, ax.transAxes)
    ax.text(
        0.98,
        0.90,
        f'Leakage Post-Sel. Ratio: {post_selection_ratio:.2%} ({sample_size_post_selected}/{sample_size})',
        ha='right', va='bottom', transform=transform,
    )
    return fig, ax


def is_post_selection_valid(labeled_error_identifier: LabeledErrorDetectionIdentifier, qubit_id: IQubitID, qec_cycles: int) -> bool:
    # Data allocation
    labeled_error_identifier_post_selected: LabeledErrorDetectionIdentifier = labeled_error_identifier.copy_with_post_selection(
        use_heralded_post_selection=labeled_error_identifier.include_heralded_post_selection,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=True,
    )
    data_array: xr.DataArray = labeled_error_identifier.get_labeled_defect_stabilizer_lookup(
        cycle_stabilizer_count=qec_cycles
    )[qubit_id]
    try:
        data_array_post_selected: xr.DataArray = labeled_error_identifier_post_selected.get_labeled_defect_stabilizer_lookup(
            cycle_stabilizer_count=qec_cycles
        )[qubit_id]
        # Calculate sample size across 'measurement_repetition'
        sample_size: int = data_array.sizes[DataArrayLabels.MEASUREMENT.value]
        sample_size_post_selected: int = data_array_post_selected.sizes[DataArrayLabels.MEASUREMENT.value]
        try:
            post_selection_ratio: float = sample_size_post_selected / sample_size
        except ZeroDivisionError:
            post_selection_ratio = np.nan

        if np.isnan(post_selection_ratio) or post_selection_ratio < 0.02:
            valid_post_selection = False
        else:
            valid_post_selection = True
    except Exception:
        return False

    return valid_post_selection


def plot_defect_rate(error_identifier: IErrorDetectionIdentifier, qubit_id: IQubitID, qec_cycles: int, **kwargs) -> IFigureAxesPair:
    """
    :param error_identifier: Instance that identifiers errors.
    :param qubit_id: Qubit identifier for which the defects are plotted.
    :param qec_cycles: Integer number of qec cycles that should be included in the defect plot.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    labeled_error_identifier: LabeledErrorDetectionIdentifier = LabeledErrorDetectionIdentifier(error_identifier)
    labeled_error_identifier_post_selected: LabeledErrorDetectionIdentifier = labeled_error_identifier.copy_with_post_selection(
        use_heralded_post_selection=labeled_error_identifier.include_heralded_post_selection,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=True,
    )
    data_array: xr.DataArray = labeled_error_identifier.get_labeled_defect_stabilizer_lookup(cycle_stabilizer_count=qec_cycles)[qubit_id]
    valid_post_selection: bool = is_post_selection_valid(
        labeled_error_identifier=labeled_error_identifier,
        qubit_id=qubit_id,
        qec_cycles=qec_cycles,
    )
    # Calculate the mean across 'measurement_repetition'
    averages = data_array.mean(dim=DataArrayLabels.MEASUREMENT.value)
    label: str = qubit_id.id
    color: str = kwargs.pop('color', blue_shades[0])

    # Plotting
    label_format: LabelFormat = kwargs.get(
        SubplotKeywordEnum.LABEL_FORMAT.value,
        LabelFormat(
            x_label='QEC round',
            y_label=r'Defect rate $\langle d_i \rangle$',
        ),
    )
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    if valid_post_selection:
        # If no host-axes are available, apply post-selection fraction plotting
        flag_post_selection_fraction: str = "FLAG_plotted_post_selection_fraction"
        if not kwargs.get(flag_post_selection_fraction, False):
            kwargs[SubplotKeywordEnum.HOST_AXES.value] = plot_post_selection_fraction(
                error_identifier=error_identifier,
                qubit_id=qubit_id,
                qec_cycles=qec_cycles,
                **kwargs,
            )
            kwargs[flag_post_selection_fraction] = True

    fig, ax = construct_subplot(**kwargs)
    averages.plot.line('.-', color=color, ax=ax, label=label)
    if valid_post_selection:
        data_array_post_selected: xr.DataArray = \
        labeled_error_identifier_post_selected.get_labeled_defect_stabilizer_lookup(
            cycle_stabilizer_count=qec_cycles
        )[qubit_id]
        averages_post_selected = data_array_post_selected.mean(dim=DataArrayLabels.MEASUREMENT.value)
        averages_post_selected.plot.line('--', color=color, ax=ax)
    ax = label_format.apply_to_axes(axes=ax)

    ax.set_ylim([0.0, 0.5])
    ax.set_xlim([-0.5, data_array.coords[DataArrayLabels.STABILIZER_REPETITION.value].max() + 1])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return fig, ax


def plot_all_defect_rate(error_identifier: IErrorDetectionIdentifier, included_rounds: int, **kwargs) -> IFigureAxesPair:
    """
    :param error_identifier: Instance that identifiers errors.
    :param included_rounds: Integer number of qec cycles that should be included in the defect plot.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    fig, ax = construct_subplot(**kwargs)
    color_cycle = itertools.cycle(blue_shades)
    for qubit_id in error_identifier.involved_stabilizer_qubit_ids:
        kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
        kwargs['color'] = next(color_cycle)
        fig, ax = plot_defect_rate(
            error_identifier=error_identifier,
            qubit_id=qubit_id,
            qec_cycles=included_rounds,
            **kwargs,
        )
    return fig, ax


def plot_population_fraction(error_identifier: IErrorDetectionIdentifier, qubit_id: IQubitID, included_rounds: List[int], fraction_state: int = 2, **kwargs) -> IFigureAxesPair:
    """
    :param error_identifier: Instance that identifiers errors.
    :param qubit_id: Qubit-ID from which to plot population fraction.
    :param included_rounds: Integer number of qec cycles that should be included in the plot.
    :param fraction_state: State index (e.g. 0: Ground, 1: Excited, 2: Leakage) used to filter population.
        Defaults to leakage state.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    x_array: np.ndarray = np.asarray(included_rounds)
    fraction_ratio_array: np.ndarray = np.empty(shape=len(included_rounds))
    labeled_error_identifier: LabeledErrorDetectionIdentifier = LabeledErrorDetectionIdentifier(error_identifier)
    label: str = qubit_id.id
    color: str = kwargs.pop('color', blue_shades[0])

    # Populate leakage ratio
    for i, qec_round in enumerate(included_rounds):
        ternary_projected_array: xr.DataArray = labeled_error_identifier.get_labeled_ternary_projected_classification(cycle_stabilizer_count=qec_round)
        ternary_array: np.ndarray = ternary_projected_array.sel({DataArrayLabels.DATA_QUBIT_ID.value: qubit_id.id}).values
        fraction_count: int = int(np.sum(ternary_array == fraction_state))
        fraction_ratio: float = fraction_count / len(ternary_array)
        fraction_ratio_array[i] = fraction_ratio

    # Figure and Axes
    label_format: LabelFormat = kwargs.get(
        SubplotKeywordEnum.LABEL_FORMAT.value,
        LabelFormat(
            x_label='QEC round',
            y_label=r'Leakage [%]',
        ),
    )
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    fig, ax = construct_subplot(**kwargs)
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax.plot(
        x_array,
        fraction_ratio_array,
        '.-',
        color=color,
        label=label,
    )

    ax.set_ylim([0.0, 0.5])
    ax.set_xlim([-0.5, max(included_rounds) + 1])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return fig, ax


def plot_all_population_fraction(error_identifier: IErrorDetectionIdentifier, included_rounds: List[int], **kwargs) -> IFigureAxesPair:
    """
    Plots (leakage) population fraction of all data-qubits.
    :param error_identifier: Instance that identifiers errors.
    :param included_rounds: Integer number of qec cycles that should be included in the plot.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """
    # Data allocation
    fig, ax = construct_subplot(**kwargs)
    color_cycle = itertools.cycle(green_shades)
    for qubit_id in error_identifier.involved_data_qubit_ids:
        kwargs[SubplotKeywordEnum.HOST_AXES.value] = (fig, ax)
        kwargs['color'] = next(color_cycle)
        fig, ax = plot_population_fraction(
            error_identifier=error_identifier,
            qubit_id=qubit_id,
            qec_cycles=included_rounds,
            included_rounds=included_rounds,
            **kwargs,
        )

    return fig, ax


def plot_all_defect_and_leakage(error_identifier: IErrorDetectionIdentifier, included_rounds: List[int], **kwargs) -> IFigureAxesPair:
    """
    Plots (leakage) population fraction of all data-qubits.
    :param error_identifier: Instance that identifiers errors.
    :param included_rounds: Integer number of qec cycles that should be included in the plot.
    :param kwargs: Key-word arguments that are passed to plt.subplots() method.
    :return: Tuple of Figure and Axes pair.
    """

    kwargs = {}
    kwargs[SubplotKeywordEnum.FIGURE_SIZE.value] = (6.4, 4.8)
    # Define grid details
    gridspec_constructor_kwargs: dict = dict(height_ratios=[3, 1], width_ratios=[1], hspace=0.0, wspace=0.0)
    # Define the mosaic layout
    ax0_label: str = 'defects'
    ax1_label: str = 'leakage_population'
    fig, axs_dict = plt.subplot_mosaic(
        [
            [ax0_label],
            [ax1_label],
        ],
        gridspec_kw=gridspec_constructor_kwargs,
        **kwargs,
    )
    ax0 = axs_dict[ax0_label]
    ax1 = axs_dict[ax1_label]

    kwargs_defect = {
        SubplotKeywordEnum.HOST_AXES.value: (fig, ax0),
        SubplotKeywordEnum.LABEL_FORMAT.value: LabelFormat(
            x_label='',
            y_label=r'Defect rate $\langle d_i \rangle$',
        ),
    }
    kwargs_leakage = {
        SubplotKeywordEnum.HOST_AXES.value: (fig, ax1),
    }

    plot_all_defect_rate(
        error_identifier=error_identifier,
        included_rounds=max(included_rounds),
        **kwargs_defect,
    )
    plot_all_population_fraction(
        error_identifier=error_identifier,
        included_rounds=included_rounds,
        **kwargs_leakage,
    )

    # Remove x-axis and y-axis tick labels
    ax0.set_xticklabels([])
    return fig, ax0
