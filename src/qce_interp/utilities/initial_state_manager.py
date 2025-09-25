# -------------------------------------------
# Module containing functionality for determining initial state from code-layout and included qubit-IDs.
# -------------------------------------------
import collections
from typing import Dict, Any, Union, List, Optional, Set, Tuple
import numpy as np
from qce_circuit.connectivity.intrf_connectivity_surface_code import ISurfaceCodeLayer
from qce_circuit.connectivity.intrf_channel_identifier import IQubitID
from qce_interp.interface_definitions.intrf_error_identifier import ErrorDetectionIdentifier
from qce_interp.utilities.custom_context_manager import WhileLoopSafety


class InitialStateManager:
    """
    Behaviour class, contains functionality for determining (odd) initial state distribution.
    Deterministic outcome based on order of input data-qubit IDs.
    """

    # region Static Class Methods
    @staticmethod
    def _find_initial_state(
            qubit_mapping: Dict[Any, Union[np.ndarray, List[int]]]
    ) -> Optional[Dict[int, int]]:
        """
        Determines an initial state for data qubits to ensure even state distribution
        for each measurement qubit.

        This function solves a constraint satisfaction problem where data qubits must be
        assigned a state (0 or 1) such that for each measurement qubit, the set of
        data qubits it references has an equal number of 0s and 1s.

        It uses a backtracking algorithm with constraint propagation to find a valid
        solution. It can handle complex dependencies, including shared data qubits.

        :param qubit_mapping: A dictionary where keys are measurement qubit identifiers
                              and values are lists or numpy arrays of the data qubit
                              indices they measure.
        :return: A dictionary mapping each data qubit index to its determined initial
                 state (0 or 1), or None if no solution is found.
        """

        # Parse input and build data structures for the solver
        all_data_qubits: Set[int] = set()
        for data_qubits in qubit_mapping.values():
            for qubit_idx in data_qubits:
                all_data_qubits.add(qubit_idx)

        sorted_qubits: List[int] = sorted(list(all_data_qubits))
        initial_states: Dict[int, Optional[int]] = {q: None for q in sorted_qubits}

        constraints: List[Dict[str, Union[Tuple[int, ...], int]]] = []
        qubit_to_constraint_indices: Dict[int, List[int]] = collections.defaultdict(list)

        for meas_qubit, data_qubits in qubit_mapping.items():
            num_data_qubits = len(data_qubits)
            if num_data_qubits % 2 != 0:
                raise ValueError(
                    f"Measurement qubit '{meas_qubit}' references an odd number of data "
                    f"qubits ({num_data_qubits}), making an even distribution of "
                    "0s and 1s impossible."
                )

            constraint = {
                'qubits': tuple(sorted(data_qubits)),
                'target_sum': num_data_qubits // 2
            }
            constraints.append(constraint)
            constraint_idx = len(constraints) - 1
            for q_idx in data_qubits:
                qubit_to_constraint_indices[q_idx].append(constraint_idx)

        def solve(current_states: Dict[int, Optional[int]]) -> Optional[Dict[int, int]]:
            """
            Inner recursive function to perform the backtracking search.
            """
            # Find the next unassigned data qubit
            try:
                next_qubit = next(q for q in sorted_qubits if current_states[q] is None)
            except StopIteration:
                # Base case: all qubits have been successfully assigned a state
                return current_states

            # Try assigning both possible states (0 and 1)
            for state_to_try in [0, 1]:
                # Create a copy of the current states to allow for backtracking
                new_states = current_states.copy()
                new_states[next_qubit] = state_to_try

                # Propagate constraints
                is_consistent, propagated_states = propagate(new_states, next_qubit)

                if is_consistent:
                    # If propagation is successful, continue solving recursively
                    solution = solve(propagated_states)
                    if solution:
                        return solution

            # If neither 0 nor 1 leads to a solution, backtrack
            return None

        def propagate(
                states: Dict[int, Optional[int]],
                initial_qubit: int
        ) -> Tuple[bool, Dict[int, Optional[int]]]:
            """
            Propagates the consequences of a qubit's state assignment.
            """
            queue = collections.deque([initial_qubit])

            while queue:
                qubit = queue.popleft()
                for const_idx in qubit_to_constraint_indices[qubit]:
                    constraint = constraints[const_idx]
                    target_sum = constraint['target_sum']
                    involved_qubits = constraint['qubits']

                    known_states_sum = 0
                    unknown_qubits: List[int] = []

                    for q_idx in involved_qubits:
                        if states[q_idx] is not None:
                            known_states_sum += states[q_idx]
                        else:
                            unknown_qubits.append(q_idx)

                    if not unknown_qubits:  # All qubit states in this constraint are known
                        if known_states_sum != target_sum:
                            return False, states  # Conflict detected
                    elif len(unknown_qubits) == 1:
                        # Can determine the state of the single unknown qubit
                        unknown_q = unknown_qubits[0]
                        required_state = target_sum - known_states_sum

                        if required_state not in [0, 1]:
                            # Required state is not binary, so this path is invalid
                            return False, states

                        # If the state was already set by another constraint, check for conflict
                        if states[unknown_q] is not None and states[unknown_q] != required_state:
                            return False, states

                        if states[unknown_q] is None:
                            states[unknown_q] = required_state
                            queue.append(unknown_q)

            return True, states

        # Start the recursive search
        return solve(initial_states)

    @staticmethod
    def construct_odd_initial_state(code_layout: ISurfaceCodeLayer, involved_data_qubit_ids: Optional[List[IQubitID]] = None) -> Dict[IQubitID, int]:
        # Data allocation
        result: Dict[IQubitID, int] = {}

        if not involved_data_qubit_ids:
            involved_data_qubit_ids = code_layout.data_qubit_ids

        parity_index_lookup: Dict[IQubitID, np.ndarray] = ErrorDetectionIdentifier.get_parity_index_lookup(
            parity_layout=code_layout,
            involved_data_qubit_ids=involved_data_qubit_ids,
            involved_ancilla_qubit_ids=code_layout.ancilla_qubit_ids,
        )
        for qubit_index, state_id in InitialStateManager._find_initial_state(qubit_mapping=parity_index_lookup).items():
            result[involved_data_qubit_ids[qubit_index]] = state_id

        return result

    @staticmethod
    def construct_qubit_chain(code_layout: ISurfaceCodeLayer, involved_data_qubit_ids: List[IQubitID]) -> List[IQubitID]:
        """
        Constructs a 1D chain of alternating data and ancilla qubits from a given
        set of involved data qubits.

        This method models the qubit layout as a graph and assumes that the provided
        data qubits and their connecting ancillas form a simple, unbranched 1D chain.
        It traverses the graph structure to reconstruct the chain sequence.

        :param code_layout: The surface code layout object, containing information about all data and ancilla qubits.
        :param involved_data_qubit_ids: An unordered list of (data) qubit IDs that are known to form the chain.
        :return: A list of qubit IDs representing the alternating 1D chain,
                 e.g., [data1, ancilla1, data2, ancilla2, ...]. Returns an empty
                 list if no valid chain can be formed.
        """
        if not involved_data_qubit_ids:
            return []

        # 1. Build a complete connectivity map (ID-based) for the entire layout.
        all_data_qubits_in_layout = code_layout.data_qubit_ids
        full_index_lookup = ErrorDetectionIdentifier.get_parity_index_lookup(
            parity_layout=code_layout,
            involved_data_qubit_ids=all_data_qubits_in_layout,
            involved_ancilla_qubit_ids=code_layout.ancilla_qubit_ids,
        )
        ancilla_to_data_ids_map: Dict[IQubitID, List[IQubitID]] = {
            ancilla_id: [all_data_qubits_in_layout[idx] for idx in data_indices]
            for ancilla_id, data_indices in full_index_lookup.items()
        }

        # 2. Build a filtered adjacency list for the subgraph of the chain.
        involved_data_set = set(involved_data_qubit_ids)
        adj = collections.defaultdict(list)
        for ancilla_id, connected_data_ids in ancilla_to_data_ids_map.items():
            # As per the requirement, we only consider ancillas connecting two qubits.
            if len(connected_data_ids) != 2:
                continue

            d1, d2 = connected_data_ids
            # An ancilla is part of the chain if it connects two data qubits
            # that are both in our set of interest.
            if d1 in involved_data_set and d2 in involved_data_set:
                adj[d1].append(ancilla_id)
                adj[ancilla_id].append(d1)
                adj[d2].append(ancilla_id)
                adj[ancilla_id].append(d2)

        if not adj:
            return [involved_data_qubit_ids[0]] if involved_data_qubit_ids else []

        # 3. Find an endpoint of the chain to start the traversal.
        # An endpoint in a 1D chain is a node with only one connection in the subgraph.
        start_node = None
        for data_qubit_id in involved_data_qubit_ids:
            if data_qubit_id in adj and len(adj[data_qubit_id]) == 1:
                start_node = data_qubit_id
                break

        # If no endpoint is found (e.g., a cycle), start with any node.
        if start_node is None:
            start_node = involved_data_qubit_ids[0]

        # 4. Walk along the chain from the start node to construct the ordered list.
        chain = []
        visited = set()
        current_node = start_node
        with WhileLoopSafety(max_iterations=len(code_layout.qubit_ids)) as loop:
            # Execute while loop in safety environment
            while (current_node is not None and current_node not in visited) and loop.safety_condition():
                chain.append(current_node)
                visited.add(current_node)

                # Find the next unvisited neighbor to continue the chain.
                next_node = None
                for neighbor in adj[current_node]:
                    if neighbor not in visited:
                        next_node = neighbor
                        break
                current_node = next_node

        return chain
    # endregion
