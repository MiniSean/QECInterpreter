# -------------------------------------------
# Module for visualizing defect rates over all QEC cycles.
# -------------------------------------------
import itertools
import xarray as xr
from qce_interp.interface_definitions.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_error_identifier import (
    IErrorDetectionIdentifier,
    DataArrayLabels,
    ErrorDetectionIdentifier,
    LabeledErrorDetectionIdentifier,
)
from qce_interp.visualization.plotting_functionality import (
    construct_subplot,
    IFigureAxesPair,
    LabelFormat,
    AxesFormat,
    SubplotKeywordEnum,
)

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
    data_array: xr.DataArray = labeled_error_identifier.get_labeled_defect_stabilizer_lookup(cycle_stabilizer_count=qec_cycles)[qubit_id]
    # Calculate the mean across 'measurement_repetition'
    averages = data_array.mean(dim=DataArrayLabels.MEASUREMENT.value)
    label: str = qubit_id.id
    color: str = kwargs.pop('color', blue_shades[0])

    # Plotting
    label_format: LabelFormat = LabelFormat(
        x_label='QEC round',
        y_label=r'Defect rate $\langle d_i \rangle$',
    )
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    fig, ax = construct_subplot(**kwargs)
    averages.plot.line('.-', color=color, ax=ax, label=label)
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
            error_identifier,
            qubit_id,
            qec_cycles=included_rounds,
            **kwargs,
        )
    return fig, ax