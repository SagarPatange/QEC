"""Tableau state definition for QEC experiments.

This class follows the same style as SeQUeNCe's core state classes:
store `state` and `keys` with minimal extra machinery.
"""

from __future__ import annotations

from typing import Optional

from sequence.kernel.quantum_state import State

import stim


class TableauState(State):
    """Tableau-backed quantum state used by a quantum manager."""

    def __init__(self, state: Optional["stim.TableauSimulator"], keys: list[int]):
        """Create a tableau state.

        Args:
            state (Optional[stim.TableauSimulator]): Simulator payload.
                If `None`, a default simulator in |0...0> is created.
            keys (list[int]): Keys associated with this state.
        """
        super().__init__()
        self.keys = list(keys)

        if state is None:
            self.state = stim.TableauSimulator()
        elif isinstance(state, stim.TableauSimulator):
            self.state = state
        else:
            raise TypeError(
                f"state must be stim.TableauSimulator or None, got {type(state)}"
            )

    @classmethod
    def from_seed(cls, key: int, seed: Optional[int] = None) -> "TableauState":
        """Create a one-key state with an optional simulator seed."""
        sim = stim.TableauSimulator(seed=seed) if seed is not None else stim.TableauSimulator()
        return cls(state=sim, keys=[key])

    def copy(self) -> "TableauState":
        """Create a copy of this state."""
        if self.state is None:
            return TableauState(state=None, keys=self.keys.copy())
        sim_copy = self.state.copy() if hasattr(self.state, "copy") else self.state
        return TableauState(state=sim_copy, keys=self.keys.copy())

    def current_inverse_tableau(self):
        """Return current inverse tableau from simulator state.

        This is mainly for advanced/internal use.
        """
        if self.state is None:
            raise ValueError("TableauState is uninitialized (state is None).")
        return self.state.current_inverse_tableau()

    def current_tableau(self):
        """Return the forward tableau for user-facing state inspection."""
        inverse_tableau = self.current_inverse_tableau()
        return inverse_tableau.inverse()

    def readable_tableau(self) -> str:
        """Return a readable tableau string for display/debugging."""
        return "\n".join([
            "Keys:",
            str(self.keys),
            "Tableau:",
            str(self.current_tableau()),
        ])

    def __str__(self):
        """String form defaults to a readable forward-tableau view."""
        return self.readable_tableau()

    def serialize(self) -> dict:
        raise NotImplementedError(
            "TableauState does not support base complex serialization."
        )

    def deserialize(self, json_data) -> None:
        raise NotImplementedError(
            "TableauState cannot be deserialized from base complex format."
        )
