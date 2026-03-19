"""Experimental Tableau quantum manager scaffold.

This module intentionally lives in QEC for iterative development.
It inherits SeQUeNCe's QuantumManager API so we can keep the same
architecture and swap formalisms with minimal core changes.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Union

import numpy as np

from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import TABLEAU_FORMALISM

from QEC.quantum_state_tableau import TableauState

import stim


@QuantumManager.register(TABLEAU_FORMALISM)
class QuantumManagerTableau(QuantumManager):
    """Quantum manager scaffold for tableau-state simulation.

    This class follows SeQUeNCe's `QuantumManager` contract so the formalism
    can be swapped with minimal architectural changes.

    Notes:
        Current implementation supports state creation and assignment. Circuit
        execution is intentionally left as a staged TODO.
    """

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
        self.base_seed = seed
        self._seed_counter = 0
        self.gate_fid = float(kwargs.get("gate_fid", kwargs.get("single_qubit_gate_fid", 1.0)))
        self.two_qubit_gate_fid = float(kwargs.get("two_qubit_gate_fid", 1.0))
        self.measurement_fid = float(kwargs.get("measurement_fid", 1.0))
        self._key_layout: dict[int, tuple[int, int]] = {}

    def _next_seed(self) -> Optional[int]:
        """Return next seed value or None if unseeded."""
        if self.base_seed is None:
            return None
        seed = int(self.base_seed + self._seed_counter)
        self._seed_counter += 1
        return seed

    def _rebuild_key_layout(self) -> None:
        """Rebuild key layout as key -> (state_object_id, local_index)."""
        layout: dict[int, tuple[int, int]] = {}
        seen_state_ids = set()
        for state_obj in self.states.values():
            state_id = id(state_obj)
            if state_id in seen_state_ids:
                continue
            seen_state_ids.add(state_id)
            for local_index, key in enumerate(state_obj.keys):
                layout[key] = (state_id, local_index)
        self._key_layout = layout

    def _sim_from_tableau(self, tableau: "stim.Tableau", num_keys: int) -> "stim.TableauSimulator":
        """Convert a tableau into a seeded simulator state.

        Args:
            tableau (stim.Tableau): Tableau to convert.
            num_keys (int): Expected number of qubits.

        Returns:
            stim.TableauSimulator: Simulator initialized from the supplied tableau.

        Raises:
            ValueError: If tableau qubit count does not match `num_keys`.
        """
        if len(tableau) != num_keys:
            raise ValueError(
                f"Initializer tableau has {len(tableau)} qubits but {num_keys} keys were supplied."
            )
        simulator = stim.TableauSimulator(seed=self._next_seed())
        simulator.set_inverse_tableau(tableau.inverse())
        return simulator

    def _tableau_from_initializer(self, initializer: Any) -> "stim.Tableau":
        """Convert supported initializer formats into a `stim.Tableau`."""
        def is_pauli_sequence(values: Any) -> bool:
            return isinstance(values, (list, tuple)) and all(
                isinstance(v, (str, stim.PauliString)) for v in values
            )

        if isinstance(initializer, stim.Tableau):
            return initializer

        if isinstance(initializer, stim.Circuit):
            return stim.Tableau.from_circuit(initializer)

        if isinstance(initializer, str):
            return stim.Tableau.from_named_gate(initializer)

        if isinstance(initializer, Mapping):
            # Explicit keyed initializers.
            if "circuit" in initializer:
                return stim.Tableau.from_circuit(initializer["circuit"])
            if "name" in initializer:
                return stim.Tableau.from_named_gate(initializer["name"])
            if "stabilizers" in initializer:
                return stim.Tableau.from_stabilizers(initializer["stabilizers"])
            if "xs" in initializer and "zs" in initializer:
                return stim.Tableau.from_conjugated_generators(
                    xs=initializer["xs"],
                    zs=initializer["zs"],
                )
            if "state_vector" in initializer:
                return stim.Tableau.from_state_vector(
                    initializer["state_vector"],
                    endian=initializer.get("endian", "little"),
                )
            if "matrix" in initializer:
                return stim.Tableau.from_unitary_matrix(
                    initializer["matrix"],
                    endian=initializer.get("endian", "little"),
                )

            # Fall back to Stim's from_numpy signature if keys match that API.
            try:
                return stim.Tableau.from_numpy(**dict(initializer))
            except TypeError as exc:
                raise TypeError(
                    "Unsupported mapping initializer for tableau construction."
                ) from exc

        # from_conjugated_generators direct tuple form: (xs, zs)
        if (
            isinstance(initializer, tuple)
            and len(initializer) == 2
            and is_pauli_sequence(initializer[0])
            and is_pauli_sequence(initializer[1])
        ):
            xs, zs = initializer
            return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)

        # from_stabilizers direct list/tuple form.
        if is_pauli_sequence(initializer):
            return stim.Tableau.from_stabilizers(initializer)

        # Numpy-like matrix/vector forms:
        # - 1D -> from_state_vector
        # - 2D square -> from_unitary_matrix
        if isinstance(initializer, np.ndarray) or isinstance(initializer, (list, tuple)):
            arr = np.asarray(initializer)
            if arr.ndim == 1 and np.issubdtype(arr.dtype, np.number):
                return stim.Tableau.from_state_vector(arr, endian="little")
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and np.issubdtype(arr.dtype, np.number):
                return stim.Tableau.from_unitary_matrix(arr, endian="little")

        raise TypeError(
            "Unsupported tableau initializer. Supported forms include: "
            "stim.Tableau, stim.TableauSimulator, stim.Circuit, gate-name str, "
            "stabilizer list, (xs, zs) tuple, numpy/list state vector, "
            "numpy/list square matrix, and mapping-based initializers."
        )

    def new(self, state: Optional[Union["TableauState", "stim.Tableau", "stim.TableauSimulator", "stim.Circuit"]] = None) -> int:
        """Create and register a new tableau-backed state key.

        Args:
            state (Optional[Union[TableauState, stim.Tableau, stim.TableauSimulator, stim.Circuit]]):
                Optional initializer:
                - None: default seeded single-qubit simulator state.
                - TableauState: copied and rebound to the new key.
                - stim.TableauSimulator: copied and rebound to the new key.
                - stim.Tableau / stim.Circuit: converted to simulator state.
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
            self.states[key] = TableauState.from_seed(key=key, seed=self._next_seed())
        elif isinstance(state, TableauState):
            new_state = state.copy()
            new_state.keys = [key]
            self.states[key] = new_state
        elif isinstance(state, stim.TableauSimulator):
            sim_copy = state.copy() if hasattr(state, "copy") else state
            self.states[key] = TableauState(state=sim_copy, keys=[key])
        else:
            tableau = self._tableau_from_initializer(state)
            self.states[key] = TableauState(state=self._sim_from_tableau(tableau, num_keys=1), keys=[key])
        self._rebuild_key_layout()
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
        elif isinstance(amplitudes, stim.TableauSimulator):
            state = TableauState(state=amplitudes, keys=list(keys))
        else:
            tableau = self._tableau_from_initializer(amplitudes)
            state = TableauState(state=self._sim_from_tableau(tableau, num_keys=len(keys)), keys=list(keys))
        for key in keys:
            self.states[key] = state
        self._rebuild_key_layout()

    def remove(self, key: int) -> None:
        """Remove a key and refresh the debug key layout map."""
        super().remove(key)
        self._rebuild_key_layout()

    def get_readable_state(self, key: int) -> str:
        """Return a user-friendly string representation of a stored state."""
        qstate = self.get(key)
        if isinstance(qstate, TableauState):
            return qstate.readable_tableau()
        return str(qstate)

    def _apply_sequence_gate(self, simulator: "stim.TableauSimulator", gate_name: str, targets: list[int], arg=None) -> None:
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

    def _apply_gate_error(self, simulator: "stim.TableauSimulator", gate_name: str, targets: list[int], rng) -> None:
        """Apply gate-error noise after ideal gate application using fidelity parameters."""
        name = gate_name.lower()
        single_qubit_gates = {"h", "x", "y", "z", "s", "sdg"}
        two_qubit_gates = {"cx", "cz", "swap"}

        if name in single_qubit_gates:
            # Conversion assumes `gate_fid` is average gate fidelity F_avg and
            # errors are modeled as a uniform non-identity Pauli channel:
            #   F_avg = 1 - (d/(d+1)) * p_error
            # => p_error = ((d+1)/d) * (1 - F_avg), with d = 2 for 1-qubit gates.
            p_error = max(0.0, min(1.0, 1.5 * (1.0 - self.gate_fid)))
            if p_error > 0.0 and rng.random() < p_error:
                pauli = ("X", "Y", "Z")[int(rng.integers(0, 3))]
                if pauli == "X":
                    simulator.x(targets[0])
                elif pauli == "Y":
                    simulator.y(targets[0])
                else:
                    simulator.z(targets[0])
            return

        if name in two_qubit_gates:
            # Same formula as above, now with d = 4 for 2-qubit gates:
            #   p_error = ((d+1)/d) * (1 - F_avg) = (5/4) * (1 - F_avg).
            p_error = max(0.0, min(1.0, 1.25 * (1.0 - self.two_qubit_gate_fid)))
            if p_error > 0.0 and rng.random() < p_error:
                # Uniformly sample one of the 15 non-identity two-qubit Pauli errors.
                pauli_pairs = [
                    ("I", "X"), ("I", "Y"), ("I", "Z"),
                    ("X", "I"), ("X", "X"), ("X", "Y"), ("X", "Z"),
                    ("Y", "I"), ("Y", "X"), ("Y", "Y"), ("Y", "Z"),
                    ("Z", "I"), ("Z", "X"), ("Z", "Y"), ("Z", "Z"),
                ]
                p0, p1 = pauli_pairs[int(rng.integers(0, len(pauli_pairs)))]

                if p0 == "X":
                    simulator.x(targets[0])
                elif p0 == "Y":
                    simulator.y(targets[0])
                elif p0 == "Z":
                    simulator.z(targets[0])

                if p1 == "X":
                    simulator.x(targets[1])
                elif p1 == "Y":
                    simulator.y(targets[1])
                elif p1 == "Z":
                    simulator.z(targets[1])
            return

    def run_circuit(self, circuit, keys: list[int], meas_samp=None):
        """Execute a SeQUeNCe circuit on tableau-backed states.

        Args:
            circuit: Circuit object to execute (expected SeQUeNCe Circuit or
                compatible representation).
            keys (list[int]): Ordered keys mapped to circuit qubit indices.
            meas_samp: Optional measurement sample value used by base-manager
                validation conventions.

        Returns:
            dict[int, int]: Measurement outcomes keyed by measured qstate keys.

        Raises:
            ValueError: If keys do not currently share one grouped state object.
            NotImplementedError: If a circuit gate is not supported.

        Notes:
            Measured qubits are always split from remaining entangled keys
            (Ket-style behavior).
        """
        super().run_circuit(circuit, keys, meas_samp)

        if not keys:
            return {}

        # Current MVP requires a single shared TableauState for all input keys.
        # This mirrors "already grouped" semantics and avoids ambiguous merge logic.
        state_objects = [self.states[key] for key in keys]
        if not all(obj is state_objects[0] for obj in state_objects):
            raise ValueError(
                "All keys in run_circuit must currently share one TableauState object. "
                "Group/initialize them together with set(keys, ...) first."
            )

        state_obj = state_objects[0]
        simulator = state_obj.state
        gate_rng = np.random.default_rng(self._next_seed())
        self._rebuild_key_layout()

        # Map manager keys to simulator-local qubit indices for this shared state.
        state_id = id(state_obj)
        key_to_local: dict[int, int] = {}
        for key in state_obj.keys:
            entry = self._key_layout.get(key)
            if entry is None or entry[0] != state_id:
                raise RuntimeError(f"Stale key layout entry for key {key}.")
            key_to_local[key] = entry[1]

        # Apply unitary gates in circuit order.
        for gate_name, indices, arg in circuit.gates:
            # SeQUeNCe gate indices are local to `keys`; convert into state-local indices.
            mapped_targets = [key_to_local[keys[i]] for i in indices]
            self._apply_sequence_gate(simulator, gate_name, mapped_targets, arg)
            self._apply_gate_error(simulator, gate_name, mapped_targets, gate_rng)

        # Fast path: no measurements requested, only state evolution.
        if len(circuit.measured_qubits) == 0:
            for key in state_obj.keys:
                self.states[key] = state_obj
            self._rebuild_key_layout()
            return {}

        measured_keys = [keys[i] for i in circuit.measured_qubits]
        remaining_keys = [k for k in state_obj.keys if k not in measured_keys]

        # Use a deterministic RNG stream for readout-fidelity flips.
        # Parent validation guarantees `meas_samp` is provided when measuring.
        rng_seed = int(float(meas_samp) * (2 ** 31 - 1))
        rng = np.random.default_rng(rng_seed)

        results: dict[int, int] = {}
        physical_results: dict[int, int] = {}

        for measured_key in measured_keys:
            # This is the physical collapse result produced by the simulator.
            physical_bit = int(simulator.measure(key_to_local[measured_key]))
            physical_results[measured_key] = physical_bit

            # Readout infidelity applies to the reported classical bit only.
            reported_bit = physical_bit
            if self.measurement_fid < 1.0 and rng.random() > self.measurement_fid:
                reported_bit ^= 1
            results[measured_key] = reported_bit

        # Re-assign measured keys as independent collapsed states.
        for measured_key in measured_keys:
            collapsed = stim.TableauSimulator(seed=self._next_seed())
            if physical_results[measured_key] == 1:
                collapsed.x(0)
            self.states[measured_key] = TableauState(state=collapsed, keys=[measured_key])

        # Remaining keys share one post-measurement simulator state.
        if remaining_keys:
            remaining_state = TableauState(state=simulator, keys=remaining_keys)
            for key in remaining_keys:
                self.states[key] = remaining_state

        self._rebuild_key_layout()
        return results
