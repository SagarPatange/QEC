"""Experimental Tableau quantum manager scaffold.

This module intentionally lives in QEC for iterative development.
It inherits SeQUeNCe's QuantumManager API so we can keep the same
architecture and swap formalisms with minimal core changes.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from sequence.components.circuit import Circuit as SequenceCircuit
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import TABLEAU_FORMALISM

from sequence.kernel.quantum_state import TableauState

import stim
from stim import Tableau, TableauSimulator, Circuit


@QuantumManager.register(TABLEAU_FORMALISM)
class QuantumManagerTableau(QuantumManager):
    """Quantum manager scaffold for tableau-state simulation.

    This class follows SeQUeNCe's `QuantumManager` contract so the formalism
    can be swapped with minimal architectural changes.

    Notes:
        Current implementation supports state creation and assignment. Circuit
        execution is intentionally left as a staged TODO.
    """

    ONE_QUBIT_GATE_TIME_PS = 20_000
    TWO_QUBIT_GATE_TIME_PS = 250_000
    MEASUREMENT_TIME_PS = 500_000
    RESET_TIME_PS = 1_200_000

    def __init__(self, truncation: int = 1, seed: Optional[int] = None, **kwargs):
        """Initialize a tableau manager instance.

        Args:
            truncation (int): Hilbert-space truncation placeholder, retained to
                match the parent API and other formalisms.
            seed (Optional[int]): Base seed used for deterministic child-seed
                generation. If `None`, seeding is disabled.
            gate_fid (float): Single-qubit gate fidelity in [0, 1].
            two_qubit_gate_fid (float): Two-qubit gate fidelity in [0, 1].
            measurement_fid (float): Measurement fidelity in [0, 1].
            **kwargs: Extra keyword arguments accepted for compatibility.

        Returns:
            None.

        Notes:
            The manager uses a counter-based seed derivation strategy for
            reproducible, human-traceable per-state seeds.
        """
        super().__init__(truncation=truncation)
        # Base seed controls deterministic per-state/per-operation seed derivation.
        self.base_seed = seed
        # Monotonic counter used with `base_seed` to produce unique child seeds.
        self._seed_counter = 0
        # Default fidelities are noiseless (1.0) unless explicitly overridden.
        self.gate_fid = float(kwargs.get("gate_fid", kwargs.get("single_qubit_gate_fid", 1.0)))
        self.two_qubit_gate_fid = float(kwargs.get("two_qubit_gate_fid", 1.0))
        self.measurement_fid = float(kwargs.get("measurement_fid", 1.0))
        self.gate_error_channel = str(kwargs.get("gate_error_channel", "depolarize")).lower()  # Gate-noise mode: "depolarize" (uniform) or "pauli" (weighted).
        self.pauli_1q_weights = tuple(float(w) for w in kwargs.get("pauli_1q_weights", (1.0, 1.0, 1.0)))  # Relative PAULI_CHANNEL_1 weights in X, Y, Z order.
        self.pauli_2q_weights = tuple(float(w) for w in kwargs.get("pauli_2q_weights", (1.0,) * 15))  # Relative PAULI_CHANNEL_2 weights in Stim's 15-term order.

    def new(self, state: Optional[Union[TableauState, Tableau, TableauSimulator, Circuit]] = None) -> int:
        """Create and register a new tableau-backed state key.

        Args:
            state (Optional[Union[TableauState, Tableau, TableauSimulator, Circuit]]):
                Optional initializer:
                - None: default seeded single-qubit simulator state.
                - TableauState: copied and rebound to the new key.
                - TableauSimulator: copied and rebound to the new key.
                - Tableau / Circuit: converted to simulator state.
                - Also accepts additional formats (e.g., gate-name strings,
                  stabilizer lists, numpy vectors/matrices, and mapping specs).

        Returns:
            int: Newly allocated state key.

        Raises:
            TypeError: If `state` is not a supported initializer type.
        """
        key = self._least_available
        self._least_available += 1
        if state is None:
            self.states[key] = TableauState.zero_state(key=key, seed=self._next_seed())
        elif isinstance(state, TableauState):
            new_state = state.copy()
            new_state.keys = [key]
            self.states[key] = new_state
        elif isinstance(state, TableauSimulator):
            sim_copy = state.copy() if hasattr(state, "copy") else state
            self.states[key] = TableauState(state=sim_copy, keys=[key])
        else:
            tableau = self._tableau_from_initializer(state)
            self.states[key] = TableauState(state=self._sim_from_tableau(tableau, num_keys=1), keys=[key])
        return key

    def set(self, keys: list[int], amplitudes: Any) -> None:
        """Assign a shared tableau state object to one or more keys.

        Args:
            keys (list[int]): State keys that should reference the same state.
            amplitudes (Any): State payload to assign.
                - `TableauState`: copied before assignment.
                - Any other object: wrapped in a new `TableauState`.

        Returns:
            None.

        Examples:
            `qm.set([k0], tableau_obj)`
            `qm.set([k0, k1], tableau_state)`

        Notes:
            As in other SeQUeNCe managers, all provided keys are bound to the
            same underlying state object to represent entanglement/grouping.
        """
        super().set(keys, amplitudes)
        if isinstance(amplitudes, TableauState):
            state = amplitudes.copy()
            state.keys = list(keys)
        elif isinstance(amplitudes, TableauSimulator):
            state = TableauState(state=amplitudes, keys=list(keys))
        else:
            tableau = self._tableau_from_initializer(amplitudes)
            state = TableauState(state=self._sim_from_tableau(tableau, num_keys=len(keys)), keys=list(keys))
        for key in keys:
            self.states[key] = state

    def run_circuit(self, circuit, keys: list[int], meas_samp=None) -> dict[int, int]:
        """Execute a SeQUeNCe circuit on tableau-backed states.

        Args:
            circuit: SeQUeNCe Circuit object to execute.
            keys (list[int]): Ordered keys mapped to circuit qubit indices.
            meas_samp: Measurement sample value used by run preparation.

        Returns:
            dict[int, int]: Measurement outcomes keyed by measured state keys.
        """
        # Prepare validated inputs, merged/shared topology, and key index mapping.
        meas_samp, state_obj, key_to_local = self._prepare_circuit(circuit, keys, meas_samp)
        if state_obj is None:
            return {}

        simulator = state_obj.state

        # Apply ideal gates, then configured gate-noise channel.
        for gate_name, indices, arg in circuit.gates:
            mapped_targets = [key_to_local[keys[i]] for i in indices]
            self._apply_sequence_gate(simulator, gate_name, mapped_targets, arg)
            self._apply_gate_error(simulator, gate_name, mapped_targets)

        # No measurements: topology was already prepared, so nothing else to rewrite.
        if len(circuit.measured_qubits) == 0:
            return {}

        # Determine measured and remaining keys in deterministic order.
        measured_keys = [keys[i] for i in circuit.measured_qubits]
        measured_key_set = set(measured_keys)
        remaining_keys = [key for key in state_obj.keys if key not in measured_key_set]

        # Readout-fidelity noise uses deterministic RNG seeded from meas_samp.
        rng_seed = int(float(meas_samp) * (2 ** 31 - 1))
        rng = np.random.default_rng(rng_seed)

        # Measure each requested key, report (possibly flipped) bit, and split measured states.
        results: dict[int, int] = {}
        for measured_key in measured_keys:
            physical_bit = int(simulator.measure(key_to_local[measured_key]))

            reported_bit = physical_bit
            if self.measurement_fid < 1.0 and rng.random() > self.measurement_fid:
                reported_bit ^= 1
            results[measured_key] = reported_bit
            collapsed = TableauSimulator(seed=self._next_seed())
            if physical_bit == 1:
                collapsed.x(0)
            self.states[measured_key] = TableauState(state=collapsed, keys=[measured_key])

        # Physically drop measured qubits so simulator indices stay compact for remaining keys.
        if measured_keys:
            simulator, remaining_keys = self._drop_keys_from_tableau_simulator(simulator, state_obj.keys, measured_keys)

        # Keep unmeasured keys grouped on the shared post-measurement simulator state.
        if remaining_keys:
            remaining_state = TableauState(state=simulator, keys=remaining_keys)
            for key in remaining_keys:
                self.states[key] = remaining_state

        return results

    def get_circuit_duration(self, circuit: SequenceCircuit) -> int:
        """Return the estimated execution time of a circuit in picoseconds.

        Args:
            circuit: SeQUeNCe circuit to estimate.

        Returns:
            int: Estimated circuit duration in picoseconds.
        """
        one_qubit_gates = {
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "t",
            "phase",
            "root_iZ",
            "minus_root_iZ",
            "root_iY",
            "minus_root_iY",
        }
        two_qubit_gates = {"cx", "cz", "swap"}

        duration_ps = 0
        for gate_name, _, _ in circuit.gates:
            if gate_name in one_qubit_gates:
                duration_ps += self.ONE_QUBIT_GATE_TIME_PS
            elif gate_name in two_qubit_gates:
                duration_ps += self.TWO_QUBIT_GATE_TIME_PS
            elif gate_name == "ccx":
                duration_ps += 2 * self.TWO_QUBIT_GATE_TIME_PS
            else:
                raise RuntimeError(f"Unsupported gate for duration estimate: {gate_name}")

        duration_ps += len(circuit.measured_qubits) * self.MEASUREMENT_TIME_PS
        return int(duration_ps)

    def get_reset_duration(self, num_qubits: int) -> int:
        """Return the estimated reset time in picoseconds.

        Args:
            num_qubits: Number of qubits being reset.

        Returns:
            int: Estimated reset duration in picoseconds.
        """
        if num_qubits < 0:
            raise RuntimeError(f"num_qubits must be >= 0, got {num_qubits}")
        return int(num_qubits * self.RESET_TIME_PS)
    
    def set_to_zero(self, key: int) -> None:
        """Reset a single qubit to the |0⟩ computational basis state.

        Args:
            key (int): State key of the qubit to reset.

        Returns:
            None.
        """
        sim = TableauSimulator(seed=self._next_seed())
        self.states[key] = TableauState(state=sim, keys=[key])

    def set_to_one(self, key: int) -> None:
        """Reset a single qubit to the |1⟩ computational basis state.

        Args:
            key (int): State key of the qubit to reset.

        Returns:
            None.
        """
        sim = TableauSimulator(seed=self._next_seed())
        sim.x(0)
        self.states[key] = TableauState(state=sim, keys=[key])

    def remove(self, key: int) -> None:
        """Remove a key and refresh the debug key layout map."""
        super().remove(key)
        
    def _next_seed(self) -> Optional[int]:
        """Return next seed value or None if unseeded."""
        # `None` means deterministic seeding is disabled for this manager.
        if self.base_seed is None:
            return None
        # Derive a reproducible child seed and advance the counter.
        seed = int(self.base_seed + self._seed_counter)
        self._seed_counter += 1
        return seed

    def _sim_from_tableau(self, tableau: Tableau, num_keys: int) -> TableauSimulator:
        """Convert a tableau into a seeded simulator state.

        Args:
            tableau (Tableau): Tableau to convert.
            num_keys (int): Expected number of qubits.

        Returns:
            TableauSimulator: Simulator initialized from the supplied tableau.

        Raises:
            ValueError: If tableau qubit count does not match `num_keys`.
        """
        # Guard against mismatched manager key grouping vs tableau width.
        if len(tableau) != num_keys:
            raise ValueError(f"Initializer tableau has {len(tableau)} qubits but {num_keys} keys were supplied.")
        # Fresh simulator gets a derived seed; this affects only future RNG behavior.
        simulator = TableauSimulator(seed=self._next_seed())
        # Stim simulator stores inverse tableau internally; load that representation.
        simulator.set_inverse_tableau(tableau.inverse())
        return simulator

    def _tableau_from_initializer(self, initializer: Union[Circuit, dict[str, object], np.ndarray, list[Union[int, float, complex]], tuple[Union[int, float, complex], ...]]) -> Tableau:
        """Convert supported initializer formats into a Stim tableau.

        Args:
            initializer (Union[Circuit, dict[str, object], np.ndarray, list[Union[int, float, complex]], tuple[Union[int, float, complex], ...]]):
                Supported forms:
                - `Circuit` for `Tableau.from_circuit`.
                - 1D numeric `np.ndarray`, `list`, or `tuple` for `Tableau.from_state_vector`.
                - `dict` of keyword args for `Tableau.from_numpy`.

        Returns:
            Tableau: Tableau constructed from the initializer.
        """
        # Circuit initializer: compile the circuit into an equivalent tableau.
        if isinstance(initializer, Circuit):
            return Tableau.from_circuit(initializer)

        # Dict initializer: forward explicit kwargs to Stim's numpy-based constructor.
        if isinstance(initializer, dict):
            try:
                return Tableau.from_numpy(**initializer)
            except TypeError as exc:
                raise TypeError("Unsupported dict initializer for Tableau.from_numpy.") from exc

        # Vector initializer: treat numeric 1D input as a state vector.
        if isinstance(initializer, (np.ndarray, list, tuple)):
            arr = np.asarray(initializer)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                return Tableau.from_state_vector(arr, endian="little")

        # Everything else is intentionally rejected to keep initializer semantics strict.
        raise TypeError("Unsupported tableau initializer. Supported forms: Circuit, "
                        "1D numeric state vector, or dict kwargs for Tableau.from_numpy.")

    def _apply_sequence_gate(self, simulator: TableauSimulator, gate_name: str, targets: list[int], arg=None) -> None:
        """Apply a SeQUeNCe gate name onto a Stim tableau simulator."""
        name = gate_name.lower()
        if name == "h":
            simulator.h(targets[0])
        elif name == "x":
            simulator.x(targets[0])
        elif name == "y":
            simulator.y(targets[0])
        elif name == "z":
            simulator.z(targets[0])
        elif name == "s":
            simulator.s(targets[0])
        elif name == "sdg":
            simulator.s_dag(targets[0])
        elif name == "cx":
            simulator.cx(targets[0], targets[1])
        elif name == "cz":
            simulator.cz(targets[0], targets[1])
        elif name == "swap":
            simulator.swap(targets[0], targets[1])
        else:
            raise NotImplementedError(f"Gate '{gate_name}' is not supported in QuantumManagerTableau.")

    def _apply_gate_error(self, simulator: TableauSimulator, gate_name: str, targets: list[int]) -> None:
        """Apply gate-error noise after ideal gate application using Stim channels.

        Args:
            simulator (TableauSimulator): Active simulator to mutate.
            gate_name (str): Name of gate that was just applied.
            targets (list[int]): Simulator-local target indices for the gate.

        Returns:
            None.
        """
        name = gate_name.lower()
        single_qubit_gates = {"h", "x", "y", "z", "s", "sdg"}
        two_qubit_gates = {"cx", "cz", "swap"}
        channel = self.gate_error_channel

        if name in single_qubit_gates:
            p_error = max(0.0, min(1.0, 1.5 * (1.0 - self.gate_fid)))
            if p_error <= 0.0:
                return
            noise = Circuit()
            if channel == "depolarize":
                noise.append("DEPOLARIZE1", [targets[0]], p_error)
            elif channel in {"pauli", "paulierror", "pauli_channel"}:
                if len(self.pauli_1q_weights) != 3:
                    raise ValueError("pauli_1q_weights must have 3 entries for X, Y, Z.")
                total = sum(self.pauli_1q_weights)
                probs = [p_error / 3.0, p_error / 3.0, p_error / 3.0] if total <= 0.0 else [p_error * (w / total) for w in self.pauli_1q_weights]
                noise.append("PAULI_CHANNEL_1", [targets[0]], probs)
            else:
                raise ValueError("gate_error_channel must be 'depolarize' or 'pauli'.")
            simulator.do(noise)
            return

        if name in two_qubit_gates:
            p_error = max(0.0, min(1.0, 1.25 * (1.0 - self.two_qubit_gate_fid)))
            if p_error <= 0.0:
                return
            noise = Circuit()
            if channel == "depolarize":
                noise.append("DEPOLARIZE2", [targets[0], targets[1]], p_error)
            elif channel in {"pauli", "paulierror", "pauli_channel"}:
                if len(self.pauli_2q_weights) != 15:
                    raise ValueError("pauli_2q_weights must have 15 entries in Stim PAULI_CHANNEL_2 order.")
                total = sum(self.pauli_2q_weights)
                probs = ([p_error / 15.0] * 15 if total <= 0.0 else [p_error * (w / total) for w in self.pauli_2q_weights])
                noise.append("PAULI_CHANNEL_2", [targets[0], targets[1]], probs)
            else:
                raise ValueError("gate_error_channel must be 'depolarize' or 'pauli'.")
            simulator.do(noise)
            return

    def _prepare_circuit(self, circuit, keys: list[int], meas_samp=None) -> tuple[float | None, TableauState | None, dict[int, int]]:
        """Validate/normalize run input and prepare state-topology context.

        Args:
            circuit: SeQUeNCe Circuit object to execute.
            keys (list[int]): Ordered keys mapped to circuit qubit indices.
            meas_samp: Optional measurement sample from caller.

        Returns:
            tuple[float | None, TableauState | None, dict[int, int]]:
                Normalized measurement sample, prepared shared state object (or
                `None` for empty input), and key-to-local index mapping.
        """
        # If caller omits measurement sample, use midpoint default.
        if len(circuit.measured_qubits) > 0 and meas_samp is None:
            meas_samp = 0.5

        # Local contract checks for run_circuit inputs.
        if len(keys) != circuit.size:
            raise ValueError(f"circuit.size ({circuit.size}) must equal len(keys) ({len(keys)}).")
        if len(circuit.measured_qubits) > 0 and meas_samp is None:
            raise ValueError("meas_samp must be provided when measuring qubits.")

        # Empty-key fast path.
        if not keys:
            return meas_samp, None, {}

        # Keys must be unique and present in manager state.
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate keys are not allowed in run_circuit: {keys}")
        missing_keys = [key for key in keys if key not in self.states]
        if missing_keys:
            raise ValueError(f"Unknown key(s) in run_circuit: {missing_keys}")

        # Circuit-content checks (mirrors Circuit/Ket assumptions but validates at execution boundary).
        supported_gates = {"h", "x", "y", "z", "s", "sdg", "cx", "cz", "swap"}

        for gate_name, indices, arg in circuit.gates:
            if gate_name not in supported_gates:
                raise ValueError(f"Unsupported gate '{gate_name}' for tableau manager.")
            if len(indices) == 0:
                raise ValueError(f"Gate '{gate_name}' must target at least one qubit.")
            if not all(0 <= i < circuit.size for i in indices):
                raise ValueError(f"Gate '{gate_name}' has out-of-range indices: {indices}")
            if len(set(indices)) != len(indices):
                raise ValueError(f"Gate '{gate_name}' has duplicate target indices: {indices}")

            if gate_name in {"h", "x", "y", "z", "s", "sdg"} and len(indices) != 1:
                raise ValueError(f"Gate '{gate_name}' expects 1 index, got {indices}")
            if gate_name in {"cx", "cz", "swap"} and len(indices) != 2:
                raise ValueError(f"Gate '{gate_name}' expects 2 indices, got {indices}")

        measured = circuit.measured_qubits
        if not all(0 <= i < circuit.size for i in measured):
            raise ValueError(f"Measured qubit index out of range: {measured}")
        if len(set(measured)) != len(measured):
            raise ValueError(f"Duplicate measured qubit indices are not allowed: {measured}")

        # Topology prep (Ket-style): collect unique state blocks in key traversal order.
        unique_states: list[TableauState] = []
        seen_state_ids: set[int] = set()
        for key in keys:
            qstate = self.states[key]
            if not isinstance(qstate, TableauState):
                raise ValueError(f"Expected TableauState for key {key}, got {type(qstate)}")
            state_id = id(qstate)
            if state_id not in seen_state_ids:
                seen_state_ids.add(state_id)
                unique_states.append(qstate)

        # Already grouped: reuse existing state block (supports subset keys naturally).
        if len(unique_states) == 1:
            state_obj = unique_states[0]
        else:
            # Multiple independent blocks: merge into one shared tableau state.
            merged_keys: list[int] = []
            merged_tableau = None
            for qstate in unique_states:
                merged_keys.extend(qstate.keys)
                block_tableau = qstate.current_tableau()
                merged_tableau = block_tableau if merged_tableau is None else (merged_tableau + block_tableau)

            if len(set(merged_keys)) != len(merged_keys):
                raise ValueError(f"Merged state contains duplicate keys: {merged_keys}")
            merged_sim = self._sim_from_tableau(merged_tableau, num_keys=len(merged_keys))
            state_obj = TableauState(state=merged_sim, keys=merged_keys)
            for key in merged_keys:
                self.states[key] = state_obj

        key_to_local = {key: i for i, key in enumerate(state_obj.keys)}
        return meas_samp, state_obj, key_to_local

    def _drop_keys_from_tableau_simulator(self, simulator: TableauSimulator, state_keys: list[int], drop_keys: list[int]) -> tuple[TableauSimulator, list[int]]:
        """Drop arbitrary keys by swapping them to tail and truncating qubits.

        Args:
            simulator (TableauSimulator): Simulator to mutate in place.
            state_keys (list[int]): Current key order mapped to simulator qubit indices.
            drop_keys (list[int]): Keys to remove from simulator state.

        Returns:
            tuple[TableauSimulator, list[int]]: Mutated simulator and remaining key order.
        """
        drop_set = set(drop_keys)
        keep_keys = [key for key in state_keys if key not in drop_set]
        tail_keys = [key for key in state_keys if key in drop_set]
        desired_order = keep_keys + tail_keys
        working_keys = list(state_keys)

        # Permute simulator qubits so kept keys come first and dropped keys are at the tail.
        for i, key in enumerate(desired_order):
            j = working_keys.index(key)
            if i != j:
                simulator.swap(i, j)
                working_keys[i], working_keys[j] = working_keys[j], working_keys[i]

        # Truncate tail qubits to physically shrink simulator state.
        if tail_keys:
            simulator.set_num_qubits(len(keep_keys))

        return simulator, keep_keys
