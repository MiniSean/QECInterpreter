# -------------------------------------------
# Module for visualizing defect rates over all QEC cycles.
# -------------------------------------------
import itertools
import xarray as xr
import matplotlib.pyplot as plt
from qce_interp.utilities.connectivity_surface_code import IQubitID
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
color_cycle = itertools.cycle(blue_shades)


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

    # Plotting
    label_format: LabelFormat = LabelFormat(
        x_label='QEC round',
        y_label=r'Defect rate $\langle d_i \rangle$',
    )
    kwargs[SubplotKeywordEnum.LABEL_FORMAT.value] = label_format
    fig, ax = construct_subplot(**kwargs)
    averages.plot.line('.-', color=next(color_cycle), ax=ax, label=label)
    ax = label_format.apply_to_axes(axes=ax)

    ax.set_ylim([0.0, 0.5])
    ax.set_xlim([-0.5, data_array.coords[DataArrayLabels.STABILIZER_REPETITION.value].max() + 1])
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    return fig, ax
