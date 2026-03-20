"""Tableau state definition for QEC experiments.

This class follows the same style as SeQUeNCe's core state classes:
store `state` and `keys` with minimal extra machinery.
"""

from __future__ import annotations

from sequence.kernel.quantum_state import State

from stim import TableauSimulator, Tableau


class TableauState(State):
    """Tableau-backed quantum state used by a quantum manager."""

    def __init__(self, state: TableauSimulator, keys: list[int], seed: int = None):
        """Create a tableau state.

        Args:
            state TableauSimulaton: Simulator payload.
                If `None`, a default simulator in |0...0> is created.
            keys (list[int]): Keys associated with this state.
        """
        super().__init__()
        self.keys = list(keys)
        self.seed = seed
        if state is None:
            self.state = TableauSimulator(seed=seed)
        elif isinstance(state, TableauSimulator):
            self.state = state
        else:
            raise TypeError(f"state must be stim.TableauSimulator or None, got {type(state)}")
        
    def set_seed(self, seed: int):
        """Set the random seed for this state, affecting future simulator operations."""
        self.seed = seed
        if self.state is not None:
            self.state = self.state.copy(seed=seed)

    def get_seed(self) -> int:
        """Get the current random seed for this state."""
        return self.seed

    def current_inverse_tableau(self) -> Tableau:
        """Return current inverse tableau from simulator state.

        This is mainly for advanced/internal use.
        """
        if self.state is None:
            raise ValueError("TableauState is uninitialized (state is None).")
        return self.state.current_inverse_tableau()

    def current_tableau(self) -> Tableau:
        """Return the forward tableau for user-facing state inspection."""
        if self.state is None:
            raise ValueError("TableauState is uninitialized (state is None).")
        inverse_tableau = self.state.current_inverse_tableau()
        return inverse_tableau.inverse()

    def __str__(self) -> str:
        """String form defaults to a readable forward-tableau view."""
        return "\n".join(["Keys:", str(self.keys), "Tableau:", str(self.current_tableau()),])

    def serialize(self) -> dict:
        raise NotImplementedError("TableauState does not support base complex serialization.")

    def deserialize(self) -> None:
        raise NotImplementedError("TableauState cannot be deserialized from base complex format.")
