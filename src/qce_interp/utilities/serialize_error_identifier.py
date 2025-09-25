# -------------------------------------------
# Functions for serializing error-detection identifier
# -------------------------------------------
from typing import List, Tuple, TypeVar, Union
import xarray as xr
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from qce_circuit.language.intrf_declarative_circuit import (
    InitialStateContainer,
    InitialStateEnum,
)
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID
from qce_circuit.connectivity.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_interp.utilities.custom_exceptions import ZeroClassifierShotsException
from qce_interp.interface_definitions.intrf_error_identifier import (
    ErrorDetectionIdentifier,
    ILabeledErrorDetectionIdentifier,
    LabeledErrorDetectionIdentifier,
    DataArrayLabels,
)
from qce_interp.decoder_examples.mwpm_decoders import MWPMDecoderFast
from qce_interp.decoder_examples.majority_voting import MajorityVotingDecoder
from qce_interp.utilities.initial_state_manager import InitialStateManager


T = TypeVar("T")


def construct_processed_dataset(error_identifier: ErrorDetectionIdentifier, initial_state: InitialStateContainer, qec_rounds: List[int], code_layout: ISurfaceCodeLayer) -> xr.Dataset:

    processed_dataset = xr.Dataset()
    decoder_set: List[Tuple[MWPMDecoderFast, MajorityVotingDecoder, InitialStateContainer]] = construct_sub_error_identifiers(
        error_identifier=error_identifier,
        initial_state=initial_state,
        code_layout=code_layout,
    )
    # Add defect rates
    processed_dataset = update_defect_rates(
        dataset=processed_dataset,
        labeled_error_identifier=LabeledErrorDetectionIdentifier(error_identifier),
        qec_round=qec_rounds[-1],
    )
    # Add logical fidelities
    processed_dataset = update_logical_fidelity(
        dataset=processed_dataset,
        decoder_set=decoder_set,
        qec_rounds=qec_rounds,
    )

    return processed_dataset


def get_odd_subarrays(full_array: List[T], skip: int = 1) -> List[List[T]]:
    """
    Generate all sub-arrays of all possible odd lengths (>= 3) from a given 1D array,
    with an option to skip every second odd length.

    :param full_array: The input 1D array with arbitrary element types.
    :param skip: Determines step size for odd lengths (1 = every odd, 2 = every second odd).
    :return: List of sub-arrays.
    """
    n = len(full_array)
    lengths = [length for length in range(n, 2, -2 * skip)]  # Step controls skipping behavior
    sub_arrays = []

    for length in lengths:
        for i in range(0, n - length + 1, skip):  # Skip also affects starting index
            sub_arrays.append(full_array[i:i + length])

    return sub_arrays


def construct_sub_error_identifiers(error_identifier: ErrorDetectionIdentifier, initial_state: InitialStateContainer, code_layout: ISurfaceCodeLayer) -> List[Tuple[MWPMDecoderFast, MajorityVotingDecoder, InitialStateContainer]]:
    ordered_involved_qubit_ids: List[IQubitID] = InitialStateManager.construct_qubit_chain(
        code_layout=code_layout,
        involved_data_qubit_ids=error_identifier.involved_qubit_ids,
    )

    initial_state_arrays = get_odd_subarrays(full_array=initial_state.as_array, skip=1)
    involved_qubit_arrays = get_odd_subarrays(full_array=ordered_involved_qubit_ids, skip=2)

    result: List[Tuple[MWPMDecoderFast, MajorityVotingDecoder, InitialStateContainer]] = []
    for _initial_state, _involved_qubits in zip(initial_state_arrays, involved_qubit_arrays):
        initial_state_container: InitialStateContainer = InitialStateContainer.from_ordered_list([
            InitialStateEnum.ZERO if state == 0 else InitialStateEnum.ONE
            for state in _initial_state
        ])

        _error_identifier: ErrorDetectionIdentifier = error_identifier.copy_with_involved_qubit_ids(
            involved_qubit_ids=_involved_qubits,
        )
        decoder_mwpm = MWPMDecoderFast(
            error_identifier=_error_identifier,
            qec_rounds=_error_identifier.qec_rounds,
            initial_state_container=initial_state_container,
            max_optimization_shots=2000,
            optimize=False,
            optimized_round=_error_identifier.qec_rounds[-1]
        )
        decoder_mv = MajorityVotingDecoder(
            error_identifier=_error_identifier,
        )
        result.append((decoder_mwpm, decoder_mv, initial_state_container))
    return result


def update_defect_rates(dataset: xr.Dataset, labeled_error_identifier: ILabeledErrorDetectionIdentifier, qec_round: int) -> xr.Dataset:
    labeled_error_identifier_post_selected: ILabeledErrorDetectionIdentifier = labeled_error_identifier.copy_with_post_selection(
        use_heralded_post_selection=labeled_error_identifier.include_heralded_post_selection,
        use_projected_leakage_post_selection=False,
        use_stabilizer_leakage_post_selection=True,
    )

    for qubit_id in labeled_error_identifier.involved_stabilizer_qubit_ids:
        data_array: xr.DataArray = labeled_error_identifier.get_labeled_defect_stabilizer_lookup(
            cycle_stabilizer_count=qec_round,
        )[qubit_id]
        # Calculate the mean across 'measurement_repetition'
        dataset[f"defect_rates_{qubit_id.id}"] = data_array.mean(dim=DataArrayLabels.MEASUREMENT.value)

        try:
            data_array_post_selected: xr.DataArray = labeled_error_identifier_post_selected.get_labeled_defect_stabilizer_lookup(
                cycle_stabilizer_count=qec_round,
            )[qubit_id]
            dataset[f"defect_rates_post_selected_{qubit_id.id}"] = data_array_post_selected.mean(dim=DataArrayLabels.MEASUREMENT.value)
        except ZeroClassifierShotsException as e:
            pass
    return dataset


def update_logical_fidelity(dataset: xr.Dataset, decoder_set: List[Tuple[MWPMDecoderFast, MajorityVotingDecoder, InitialStateContainer]], qec_rounds: Union[NDArray[np.int_], List[int]]) -> xr.Dataset:
    x_array: np.ndarray = np.asarray(qec_rounds)

    for decoder_index, (decoder_mwpm, decoder_mv, initial_state) in enumerate(decoder_set):
        # MWPM Decoder
        mwpm_y_array: np.ndarray = np.full_like(x_array, np.nan, dtype=np.float64)
        for i, x in tqdm(enumerate(x_array), desc=f"Processing {decoder_mwpm.__class__.__name__} Decoder (d {len(initial_state.as_array)})", total=len(x_array)):
            try:
                value: float = decoder_mwpm.get_fidelity(x, target_state=initial_state.as_array)
            except ZeroClassifierShotsException:
                value = np.nan
            mwpm_y_array[i] = value
        dataset[f"logical_fidelity_mwpm_d{len(initial_state.as_array)}_{decoder_index}"] = xr.DataArray(
            mwpm_y_array,
            coords={"qec_cycles": x_array},
            dims=["qec_cycles"],
            name="logical_fidelity",
        )
        # MV Decoder
        mv_y_array: np.ndarray = np.full_like(x_array, np.nan, dtype=np.float64)
        for i, x in tqdm(enumerate(x_array), desc=f"Processing {decoder_mv.__class__.__name__} Decoder (d {len(initial_state.as_array)})", total=len(x_array)):
            try:
                value: float = decoder_mv.get_fidelity(x, target_state=initial_state.as_array)
            except ZeroClassifierShotsException:
                value = np.nan
            mv_y_array[i] = value
        dataset[f"logical_fidelity_mv_d{len(initial_state.as_array)}_{decoder_index}"] = xr.DataArray(
            mv_y_array,
            coords={"qec_cycles": x_array},
            dims=["qec_cycles"],
            name="logical_fidelity",
        )

    return dataset
