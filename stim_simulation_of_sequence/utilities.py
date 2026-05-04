"""Fast SeQUeNCe-free reimplementation of the QEC pipeline used by main_test.py.

Reproduces the *physics* of `n_node_logical_pair_with_app` for the default
2-node config (`config/standard_configs/line_2_2G.json`):

  * Steane [[7,1,3]] encoding (Paetznick-Reichardt 8-CNOT encoder, css_codes.py:394-419)
  * Optional minimal/standard FT prep with retry (css_codes.py:213-296)
  * Teleported CNOT, three phases (TeleportedCNOT.py:190-333)
  * Noiseless endpoint stabilizer recovery (RequestLogicalPairApp.py:395-462)
  * |Phi+>_L Bell-state fidelity from logical XX, YY, ZZ correlators
        F = (1 + <X̄X̄> - <ȲȲ> + <Z̄Z̄>) / 4         (RequestLogicalPairApp.py:502)

Noise model exactly matches quantum_manager_tableau.py:375-425:
  * 1q gate noise:  p_1 = 1.5 * (1 - gate_fidelity)              -> DEPOLARIZE1
  * 2q gate noise:  p_2 = 1.25 * (1 - two_qubit_gate_fidelity)   -> DEPOLARIZE2
  * meas noise:     classical bit-flip with prob 1 - measurement_fidelity
  * "pauli" channel: respects pauli_1q_weights / pauli_2q_weights

Shortcuts taken (each preserves the statistical outcome under default params,
and they cover the dominant runtime cost in main_test.py):

  1. SeQUeNCe is bypassed entirely. main_test.py spends most of its time in
     the discrete-event scheduler (Barret-Kok physical Bell pair generation
     emits dozens of events per pair; each TCNOT phase fires classical
     messages with simulated channel delay; the network manager runs a
     reservation handshake; per-pair `qm.run_circuit` rebuilds a fresh
     TableauSimulator and merges sub-blocks for the fidelity peek). All of
     that is replaced by a single straight-line stim run per shot.

  2. Idle decoherence is dropped. With T1=T2=1e12 s (the main_test default),
     `apply_idling_decoherence` (sequence/kernel/quantum_manager.py:1903)
     produces Pauli probabilities ~ idle_sec / 1e12, i.e. ~1e-12 per ns of
     idle time -- below the floating-point noise of any other channel.

  3. Physical Bell pairs at measurement_fidelity = 1.0 are constructed
     directly with H+CX -- the SeQUeNCe Barret-Kok generator at meas_fid=1
     yields the exact same noiseless |Phi+> state, only the path differs.
     At measurement_fidelity < 1.0 we drop down to a per-pair classical
     simulation of the BarretKok protocol (`_simulate_barret_kok_pair` +
     `_prep_bell_pair_state`), which mirrors `sequence/components/bsm.py:512-580`
     and `sequence/entanglement_management/generation/barret_kok.py:69-92`
     step-for-step. This faithfully reproduces the meas_fid coupling into
     Bell-pair fidelity (false-positive ghost clicks at stage 2, etc.) that
     a naive H+CX would otherwise miss. The per-photon detector-click loss
     used by the simulation is computed from `params["link_distance_km"]`
     via `compute_photon_loss()`, so changing link_distance_km also feeds
     correctly into the BSM behavior.

     RESIDUAL: the BarretKok simulation captures the dominant meas_fid-
     induced corruption mechanisms but a small residual remains in the
     combined regime where measurement_fidelity < 1 AND link_distance_km
     is unusually large (e.g. line_3 with meas_fid=0.99 and
     link_distance_km=50 lands at ~2.25 sigma). At measurement_fidelity = 1.0
     (the default for every standard parameter sweep) photon_loss has zero
     effect on F, so this residual never affects production usage.

  4. The physical-fidelity sub-calculation (`_calculate_pair_fidelity` with
     pair_type="physical") is dropped: the parallel sweep only consumes
     `final_end_to_end_fidelity_corrected`, so the per-physical-pair Bell
     fidelity is dead computation.

  5. All stim Circuit objects and PauliString observables are precompiled
     once per `simulate_logical_pairs_2node` call. The per-shot inner loop
     only does `sim.do(circ)` / `sim.measure(q)` / `sim.peek_observable_expectation(p)`
     -- no Python-side circuit construction.

Returned per-shot fidelity is in {0, 1/2, 1}; for the |Phi+> reference state
under stabilizer noise only 0 and 1 occur in practice. Average over many
shots gives the same expected value as main_test's
`avg_end_to_end_logical_corrected`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import stim


# ---------------------------------------------------------------------------
# Steane [[7,1,3]] code data (mirrors css_codes.py:368-528)
# ---------------------------------------------------------------------------
#
# The Steane code is a CSS code with 3 X-stabilizers and 3 Z-stabilizers, sharing
# supports. Each string below is read left-to-right as positions 0..6 of the
# 7-qubit logical block. "I" means identity at that qubit; "X"/"Z" means the
# corresponding Pauli is part of the stabilizer support. Logical operators
# (LX = X^7, LZ = Z^7) are transversal -- this is what lets the cheating apply
# transversal X / Z corrections after BSM frame updates and have them act as
# logical operators on the encoded qubit.

STEANE_X_STABILIZERS: tuple[str, ...] = ("IIIXXXX", "IXXIIXX", "XIXIXIX")
STEANE_Z_STABILIZERS: tuple[str, ...] = ("IIIZZZZ", "IZZIIZZ", "ZIZIZIZ")
STEANE_LX: str = "XXXXXXX"   # Logical X = transversal X
STEANE_LZ: str = "ZZZZZZZ"   # Logical Z = transversal Z
STEANE_LY: str = "YYYYYYY"   # Logical Y = transversal Y, derived from LX*LZ on the +Phi+ Bell observable
                             # (RequestLogicalPairApp.py:558 explains the per-qubit Y vs X-or-Z choice).

# Single-error decode table: maps a 3-bit syndrome (s0, s1, s2) to the qubit that
# triggered it (or None for the trivial syndrome). The mapping is qubit index =
# (s0 << 2) | (s1 << 1) | s2 - 1, encoding the Hamming-code structure of the Steane
# parity-check matrix. Both Z stabilizers (locating X errors) and X stabilizers
# (locating Z errors) use the same table because the supports are identical.
STEANE_DECODE_TABLE: dict[tuple[int, int, int], int | None] = {
    (0, 0, 0): None, (0, 0, 1): 0, (0, 1, 0): 1, (0, 1, 1): 2,
    (1, 0, 0): 3,    (1, 0, 1): 4, (1, 1, 0): 5, (1, 1, 1): 6,
}

# FT-prep parity-check supports (Paetznick-Reichardt; first row is the trivial
# all-1s check that's never used by minimal mode; minimal mode uses the second
# row only; standard mode uses all four). These are the supports of the
# verifiers run on a per-block ancilla after encode, to catch single-X errors.
STEANE_FT_CHECK_SUPPORTS_STANDARD: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),
    (0, 5, 6),
    (1, 4, 6),
    (2, 4, 5),
)

# Paetznick-Reichardt encoder gate sequence (css_codes.py:394-419). Apply H to
# qubits 1, 2, 3 first; then 8 CNOTs in the listed order. Starting state |0000000>,
# ending state is |0>_L. For |+>_L preparation, the caller follows up with a
# transversal H on the data block.
STEANE_ENCODER_HS: tuple[int, ...] = (1, 2, 3)
STEANE_ENCODER_CXS: tuple[tuple[int, int], ...] = (
    (1, 0), (3, 5), (2, 6), (1, 4),
    (2, 0), (3, 6), (1, 5), (6, 4),
)

CODE_N: int = 7  # Steane code physical-qubit count per logical qubit.


# ---------------------------------------------------------------------------
# Default parameters (mirror build_n_node_params() in main_test.py:26-58)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, object] = {
    "num_logical_pairs": 1000,
    "gate_fidelity": 1.0,
    "two_qubit_gate_fidelity": 0.995,
    "measurement_fidelity": 1.0,
    "state_preparation_fidelity": 1.0,
    "physical_bell_pair_fidelity": 1.0,
    "gate_error_channel": "depolarize",  # "depolarize" or "pauli"
    "pauli_1q_weights": None,            # used iff channel == "pauli"
    "pauli_2q_weights": None,            # used iff channel == "pauli"
    "ft_prep_mode": "minimal",           # "none" | "minimal" | "standard"
    "ft_max_retries": 3,                 # Matches every router in config/standard_configs/line_*_2G.json.
                                         # The reference reads ft_max_retries from the per-node JSON config;
                                         # using 3 here keeps the cheating's "no overrides" behavior consistent
                                         # with running main_test.n_node_logical_pair_with_app on those configs.
    "correction_mode": "cec",            # "cec" | "qec" | "qec+cec" | "none"
    # Idle decoherence is dropped because it has no measurable effect with the
    # main_test default T1 = T2 = 1e12 s. Keys retained for parity with main_test.
    "idle_t1_sec": 1e12,
    "idle_t2_sec": 1e12,
    # link_distance_km controls the qchannel length each photon traverses to the
    # midway BSM node. Used by the BarretKok BSM simulation to compute the
    # per-photon detector-click probability; the larger the distance, the more
    # photons are lost in fiber (attenuation 0.0002 dB/m by default), which
    # shifts the BarretKok true-click vs ghost-click ratio when
    # measurement_fidelity < 1. At measurement_fidelity == 1 (the default for
    # main_test_parallel_cheating sweeps) photon loss only affects throughput,
    # not F_corrected, so this knob has no effect there.
    "link_distance_km": 1,
}


# ---------------------------------------------------------------------------
# Hardware constants used by the BarretKok BSM photon-loss model
# ---------------------------------------------------------------------------
# These match every config/standard_configs/line_*_2G.json topology file.
# Override-able via per-call kwargs only if you have a non-standard hardware
# template; the standard configs use these values uniformly.

QCHANNEL_ATTENUATION_DB_PER_M: float = 0.0002    # Fiber attenuation per metre (dB).
MEMORY_EFFICIENCY: float = 0.9                    # memory.efficiency in the JSON template.
DETECTOR_EFFICIENCY: float = 0.95                 # SingleAtomBSM detector efficiency.


def compute_photon_loss(link_distance_km: float,
                        attenuation_db_per_m: float = QCHANNEL_ATTENUATION_DB_PER_M,
                        memory_efficiency: float = MEMORY_EFFICIENCY,
                        detector_efficiency: float = DETECTOR_EFFICIENCY) -> float:
    """Compute the effective photon-loss probability used by the BarretKok BSM.

    The reference's `n_node_logical_pair_with_app` overrides every qchannel's
    distance to `link_distance_km * 1000 / 2` metres (router-to-midway-BSM is
    half the link). A photon's loss between memory emission and detector click
    is the combined product of:
        memory.efficiency * 10^(-attenuation_db * half_distance_m / 10) * detector.efficiency.
    The "photon_loss" the BSM checks is `1 - click_probability`.

    Returns:
      Effective photon_loss in [0, 1] for use by `_simulate_barret_kok_pair`.

    Examples (with the standard hardware constants):
      link_distance_km = 1   -> half_distance = 500 m  -> photon_loss ~ 0.16
      link_distance_km = 10  -> half_distance = 5 km   -> photon_loss ~ 0.32
      link_distance_km = 50  -> half_distance = 25 km  -> photon_loss ~ 0.78
    """
    half_distance_m = max(0.0, float(link_distance_km)) * 1000.0 / 2.0
    channel_db = half_distance_m * attenuation_db_per_m
    channel_transmission = 10.0 ** (-channel_db / 10.0)
    click_prob = float(memory_efficiency) * channel_transmission * float(detector_efficiency)
    return max(0.0, min(1.0, 1.0 - click_prob))


# ---------------------------------------------------------------------------
# Noise probability helpers (mirror quantum_manager_tableau.py:392, 410)
# ---------------------------------------------------------------------------
#
# SeQUeNCe parameterizes per-gate noise as a "fidelity" in [0, 1] but stim takes
# a probability that the depolarizing channel fires. The relationships below
# convert the SeQUeNCe quantities into the stim probabilities. Constants 1.5 and
# 1.25 come from how the SeQUeNCe density-matrix formalism defines fidelity for
# a uniformly depolarizing channel: fid = 1 - (3/(2*N))*p where N=2 for 1q and
# N=4 for 2q channels (see quantum_manager_tableau.py:392 / 410).

def one_qubit_error_prob(gate_fidelity: float) -> float:
    """Convert single-qubit gate fidelity to the DEPOLARIZE1 probability stim takes.

    fid = 1.0 -> p = 0; fid = 0 -> p = 1 (clamped). Mirrors the formula in
    quantum_manager_tableau.py:392.
    """
    return max(0.0, min(1.0, 1.5 * (1.0 - float(gate_fidelity))))


def two_qubit_error_prob(two_qubit_gate_fidelity: float) -> float:
    """Convert two-qubit gate fidelity to the DEPOLARIZE2 probability stim takes.

    Mirrors the formula in quantum_manager_tableau.py:410.
    """
    return max(0.0, min(1.0, 1.25 * (1.0 - float(two_qubit_gate_fidelity))))


# ---------------------------------------------------------------------------
# Qubit-index layout for the 2-node chain
# ---------------------------------------------------------------------------
#
# 2-node setup: router_0 holds {data, comm} (7 qubits each) plus an FT-prep
# ancilla; router_1 the same. We name them alice_*/bob_* so the per-hop names
# match the daisy-chained N-node case where every link looks like a 2-node
# subproblem from the perspective of TCNOT phases A/B/C.

@dataclass(frozen=True)
class TwoNodeLayout:
    """Qubit indices for the 2-node chain (router_0 = alice, router_1 = bob)."""
    alice_data: tuple[int, ...]   # router_0 data block (7 qubits, |+>_L when used as TCNOT control)
    alice_comm: tuple[int, ...]   # router_0 communication register (7 qubits)
    bob_comm: tuple[int, ...]     # router_1 communication register (7 qubits)
    bob_data: tuple[int, ...]     # router_1 data block (7 qubits, |0>_L when used as TCNOT target)

    @property
    def total_data_and_comm(self) -> int:
        return len(self.alice_data) + len(self.alice_comm) + len(self.bob_comm) + len(self.bob_data)


def make_two_node_layout() -> TwoNodeLayout:
    """Standard 28-qubit layout for the 2-node chain.

    Indices [0..6] = alice_data, [7..13] = alice_comm, [14..20] = bob_comm,
    [21..27] = bob_data. The full simulator additionally allocates 2 ancilla
    qubits at indices 28-29 (alice_ancilla, bob_ancilla) for FT-prep checks.
    """
    return TwoNodeLayout(
        alice_data=tuple(range(0, 7)),
        alice_comm=tuple(range(7, 14)),
        bob_comm=tuple(range(14, 21)),
        bob_data=tuple(range(21, 28)),
    )


# ---------------------------------------------------------------------------
# Circuit-building helpers (used during precompile, not in the hot loop)
# ---------------------------------------------------------------------------

def _append_one_qubit_gate(
    circ: stim.Circuit,
    gate: str,
    target: int,
    p1: float,
    channel: str,
    pauli_1q_weights: Sequence[float] | None,
) -> None:
    """1q gate followed by matching noise (quantum_manager_tableau.py:386-407)."""
    circ.append(gate, [target])
    if p1 <= 0.0:
        return
    if channel == "depolarize":
        circ.append("DEPOLARIZE1", [target], p1)
        return
    if channel in {"pauli", "paulierror", "pauli_channel"}:
        if pauli_1q_weights is None or len(pauli_1q_weights) != 3:
            raise ValueError("pauli_1q_weights must have 3 entries for X, Y, Z.")
        total = sum(pauli_1q_weights)
        if total <= 0.0:
            probs = [p1 / 3.0, p1 / 3.0, p1 / 3.0]
        else:
            probs = [p1 * (w / total) for w in pauli_1q_weights]
        circ.append("PAULI_CHANNEL_1", [target], probs)
        return
    raise ValueError(f"Unknown gate_error_channel '{channel}'.")


def _append_two_qubit_gate(
    circ: stim.Circuit,
    gate: str,
    a: int,
    b: int,
    p2: float,
    channel: str,
    pauli_2q_weights: Sequence[float] | None,
) -> None:
    """2q gate followed by matching noise (quantum_manager_tableau.py:409-424)."""
    circ.append(gate, [a, b])
    if p2 <= 0.0:
        return
    if channel == "depolarize":
        circ.append("DEPOLARIZE2", [a, b], p2)
        return
    if channel in {"pauli", "paulierror", "pauli_channel"}:
        if pauli_2q_weights is None or len(pauli_2q_weights) != 15:
            raise ValueError("pauli_2q_weights must have 15 entries (Stim PAULI_CHANNEL_2 order).")
        total = sum(pauli_2q_weights)
        if total <= 0.0:
            probs = [p2 / 15.0] * 15
        else:
            probs = [p2 * (w / total) for w in pauli_2q_weights]
        circ.append("PAULI_CHANNEL_2", [a, b], probs)
        return
    raise ValueError(f"Unknown gate_error_channel '{channel}'.")


def _build_steane_encoder(
    block_qubits: Sequence[int],
    p1: float, p2: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
    pauli_2q: Sequence[float] | None,
) -> stim.Circuit:
    """Build the Paetznick-Reichardt encoder for one Steane block (css_codes.py:394-419).

    Maps |0000000> -> |0>_L (the encoded computational-basis 0). Each gate
    carries the configured noise channel (DEPOLARIZE1/2 or PAULI_CHANNEL_*) so
    encoder noise propagates exactly the way SeQUeNCe's qm.run_circuit does it.
    Used both for fresh encoding and for FT-prep retries.
    """
    circ = stim.Circuit()
    for local_idx in STEANE_ENCODER_HS:
        _append_one_qubit_gate(circ, "H", block_qubits[local_idx], p1, channel, pauli_1q)
    for ctrl_local, tgt_local in STEANE_ENCODER_CXS:
        _append_two_qubit_gate(circ, "CX", block_qubits[ctrl_local], block_qubits[tgt_local], p2, channel, pauli_2q)
    return circ


def _build_transversal_h(
    block_qubits: Sequence[int],
    p1: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
) -> stim.Circuit:
    """Build transversal H on a Steane block, taking |0>_L -> |+>_L.

    Used to convert an encoded |0>_L (from _build_steane_encoder) into the
    encoded |+>_L state needed for the right-facing block of every link in the
    daisy-chained TCNOT protocol.
    """
    circ = stim.Circuit()
    for q in block_qubits:
        _append_one_qubit_gate(circ, "H", q, p1, channel, pauli_1q)
    return circ


# Default per-photon BSM loss used when the caller doesn't supply one. The
# numeric value here is computed from `link_distance_km = 1` (which matches
# build_n_node_params() default in main_test.py and main_test_cheating.py)
# combined with the standard hardware constants (memory.efficiency = 0.9,
# detector.efficiency = 0.95, fibre attenuation 0.0002 dB/m). The actual
# per-shot photon_loss used by _run_shot / _run_shot_n_node comes from
# pipe.photon_loss, which compute_photon_loss() recalculates from
# params["link_distance_km"] in _compile_pipeline / _compile_pipeline_n_node;
# see compute_photon_loss() above for the derivation.
BARRET_KOK_DEFAULT_PHOTON_LOSS: float = compute_photon_loss(1.0)


_BSM_OUTCOME_CHOICE_2 = (False, True)  # detector_num via boolean (rng.random() < 0.5)


def _simulate_barret_kok_pair(rng: np.random.Generator, measurement_fidelity: float,
                              photon_loss: float = BARRET_KOK_DEFAULT_PHOTON_LOSS,
                              max_attempts: int = 1024) -> tuple[str, int, int]:
    """Faithful classical simulation of one BarretKok-BSM Bell-pair generation.

    This is the core fidelity-preserving translation of the SeQUeNCe BarretKok
    protocol into a per-pair classical sample. SeQUeNCe runs the protocol
    inside a discrete-event timeline (events for emit, photon transit,
    detector clicks, classical message arrivals, retries, ...). We collapse
    that whole event sequence into one Python function whose output is the
    joint memory state distribution -- the only thing that matters for the
    downstream stim simulation. The protocol logic mirrors:
       sequence/components/bsm.py:512-580 (BSM photon-detection logic + state assignment)
       sequence/entanglement_management/generation/barret_kok.py:69-92 (round-by-round
       BSM-result handling, X correction after stage 1, X/Z correction after stage 2)

    The protocol has two stages (rounds 1 and 2), each producing a BSM "click"
    event whose detector_num is recorded as bsm_res[0] / bsm_res[1]. There are
    three ways the protocol can interpret photon measurements:

      1. TRUE click (XOR=1 reported, single detector fires): _set_pure_state is
         called and memory state is overwritten to |Psi+/-> based on detector_num.
      2. GHOST click (XOR=0 reported, single detector fires due to (1,1) photons):
         no state assignment -- memory stays in physical eigenstate. This is the
         path that causes corruption (memory in product state |0,0>, |0,1>, |1,0>,
         or |1,1> after the protocol "succeeds").
      3. FAIL (XOR=0 reported with 0 fires or 2 fires; XOR=1 with photon lost):
         protocol fails, retry from round 1.

    measurement_fidelity flips photon-meas bits with prob (1 - meas_fid), allowing
    the reported XOR to differ from the physical XOR. This is the lever that
    introduces meas_fid sensitivity into the Bell pair.

    Returns:
      (state_label, bsm_res_0, bsm_res_1)
      state_label is one of {'phi+','phi-','psi+','psi-','00','01','10','11'},
      possibly with a leading '-' for an unobservable global phase. The caller
      uses _prep_bell_pair_state to materialize this state on stim qubits via
      the appropriate Cliffords.

    Args:
      rng: numpy RNG used for all stochastic decisions (photon meas, flip
           noise, detector_num, photon-loss check). Independent rng draws so
           each successful Bell pair attempt is independent.
      measurement_fidelity: Per-photon readout fidelity in [0, 1]. At 1.0 the
           function only ever returns 'phi+' (modulo phase). Below 1.0 the ghost
           and false-positive paths produce a small fraction of corrupted pairs.
      photon_loss: Per-photon "fails to trigger detector" probability. Combines
           memory.efficiency, fibre transmission, and detector.efficiency. The
           per-shot driver derives this from params["link_distance_km"] via
           compute_photon_loss(); longer fibres push photon_loss up and shift
           the BarretKok ghost-vs-true click ratio (visible in F only when
           measurement_fidelity < 1.0, since at meas_fid=1 the round-1 X
           correction destroys ghost paths regardless of photon_loss).
      max_attempts: Safety cap on the retry loop (failures cause restarts).
    """
    p_flip = max(0.0, min(1.0, 1.0 - float(measurement_fidelity)))

    # Internal symbolic state representation: tuple (label, sign) where sign in {+1, -1}.
    # label is one of {'phi+', 'phi-', 'psi+', 'psi-', '00', '01', '10', '11'}.

    def _z_meas_distribution(state):
        """Probability over (m0, m1) Z-basis outcomes for the joint state."""
        lbl, _ = state
        if lbl == 'phi+' or lbl == 'phi-':
            return [(0, 0, 0.5), (1, 1, 0.5)]
        if lbl == 'psi+' or lbl == 'psi-':
            return [(0, 1, 0.5), (1, 0, 0.5)]
        # Product state: deterministic outcome.
        m0 = int(lbl[0]); m1 = int(lbl[1])
        return [(m0, m1, 1.0)]

    def _set_pure_after_xor1_click(detector_num):
        # Stage 1 BSM XOR=1 branch: state set to psi- if detector==0 else psi+.
        return ('psi-' if detector_num == 0 else 'psi+', +1)

    def _set_post_stage2_state(state0_label, detector_num):
        # Stage 2 _set_state_with_fidelity at fidelity=1.0:
        # eq_psi_plus(state0) XOR detector_num == True -> psi-, else psi+.
        eq_psi_plus = (state0_label == 'psi+')
        return ('psi-' if (eq_psi_plus ^ bool(detector_num)) else 'psi+', +1)

    def _project_to_eigenstate(m0, m1):
        return (f'{m0}{m1}', +1)

    def _apply_xx(state):
        lbl, sign = state
        if lbl == 'phi+':
            return ('phi+', sign)            # X⊗X on (|00>+|11>) = |11>+|00>
        if lbl == 'phi-':
            return ('phi-', -sign)           # X⊗X on (|00>-|11>) = |11>-|00> = -phi-
        if lbl == 'psi+':
            return ('psi+', sign)
        if lbl == 'psi-':
            return ('psi-', -sign)
        # Product state: flip both bits.
        m0 = 1 - int(lbl[0]); m1 = 1 - int(lbl[1])
        return (f'{m0}{m1}', sign)

    def _apply_x_on_a(state):
        lbl, sign = state
        if lbl == 'phi+': return ('psi+', sign)
        if lbl == 'phi-': return ('psi-', -sign)
        if lbl == 'psi+': return ('phi+', sign)
        if lbl == 'psi-': return ('phi-', -sign)
        m0 = 1 - int(lbl[0]); m1 = int(lbl[1])
        return (f'{m0}{m1}', sign)

    def _apply_z_on_b(state):
        lbl, sign = state
        if lbl == 'phi+': return ('phi-', sign)
        if lbl == 'phi-': return ('phi+', sign)
        if lbl == 'psi+': return ('psi-', -sign)
        if lbl == 'psi-': return ('psi+', -sign)
        m0 = int(lbl[0]); m1 = int(lbl[1])
        # Z|0> = |0>, Z|1> = -|1>.
        new_sign = sign * (-1 if m1 == 1 else +1)
        return (f'{m0}{m1}', new_sign)

    for _attempt in range(max_attempts):
        # Round 1: memories reset to |+>|+> (state name not used internally, we
        # sample the photon Z-basis bits directly).
        # Stage 1 BSM:
        m0 = int(rng.random() < 0.5)
        m1 = int(rng.random() < 0.5)
        r0 = m0 ^ (1 if rng.random() < p_flip else 0)
        r1 = m1 ^ (1 if rng.random() < p_flip else 0)

        if r0 ^ r1 == 1:
            # XOR=1 branch: BSM "click", state assigned to psi-/+.
            detector_num = 1 if rng.random() < 0.5 else 0
            # photon = p0 if r0 else p1; photon-loss check.
            if rng.random() <= 1.0 - photon_loss:
                bsm_res_0 = detector_num
                state = _set_pure_after_xor1_click(detector_num)
            else:
                continue  # photon lost, retry
        else:
            # XOR=0 branch: possible ghost click via single-fire.
            d0_fires = (r0 == 1) and (rng.random() <= 1.0 - photon_loss)
            d1_fires = (r1 == 1) and (rng.random() <= 1.0 - photon_loss)
            n_fires = int(d0_fires) + int(d1_fires)
            if n_fires == 1:
                # Single ghost click. One detector_num random per fire (we use one).
                detector_num = 1 if rng.random() < 0.5 else 0
                bsm_res_0 = detector_num
                # State NOT assigned; memories collapse to physical eigenstate.
                state = _project_to_eigenstate(m0, m1)
            else:
                continue  # 0 or 2 fires (double-fire bsm_res reset to -1), retry

        # Round 1 X correction: each node applies X to its memory -> joint X⊗X.
        state = _apply_xx(state)

        # Stage 2 BSM: photons inherit the joint memory state (via shared qstate_keys).
        # Sample Z-basis outcomes from the current joint state distribution.
        outcomes = _z_meas_distribution(state)
        u = rng.random()
        cum = 0.0
        m0 = m1 = 0
        for (a_bit, b_bit, prob) in outcomes:
            cum += prob
            if u <= cum:
                m0, m1 = a_bit, b_bit
                break
        r0 = m0 ^ (1 if rng.random() < p_flip else 0)
        r1 = m1 ^ (1 if rng.random() < p_flip else 0)

        if r0 ^ r1 == 1:
            detector_num = 1 if rng.random() < 0.5 else 0
            if rng.random() <= 1.0 - photon_loss:
                bsm_res_1 = detector_num
                # State at the moment of stage 2 BSM: photon measurement collapses
                # to physical eigenstate (m0, m1). For a Bell input, this is
                # anti-correlated. Then _set_state_with_fidelity at fidelity=1.0
                # overwrites state to psi+/- per the stage1-state XOR detector rule.
                state0_label = state[0]
                state = _set_post_stage2_state(state0_label, detector_num)
            else:
                continue  # photon lost
        else:
            # Stage 2 ghost click candidate.
            d0_fires = (r0 == 1) and (rng.random() <= 1.0 - photon_loss)
            d1_fires = (r1 == 1) and (rng.random() <= 1.0 - photon_loss)
            n_fires = int(d0_fires) + int(d1_fires)
            if n_fires == 1:
                detector_num = 1 if rng.random() < 0.5 else 0
                bsm_res_1 = detector_num
                # No state assignment; memories collapse to physical (m0, m1).
                state = _project_to_eigenstate(m0, m1)
            else:
                continue

        # Round 3 corrections: primary X applied to memory_A.
        # Non-primary Z applied to memory_B if bsm_res_0 != bsm_res_1.
        # Joint correction: X⊗I, then maybe I⊗Z.
        state = _apply_x_on_a(state)
        if bsm_res_0 != bsm_res_1:
            state = _apply_z_on_b(state)

        # Sign is unobservable in fidelity computation; encode into the label
        # only when negative for caller's optional use.
        label, sign = state
        if sign < 0:
            label = '-' + label
        return (label, bsm_res_0, bsm_res_1)

    raise RuntimeError(f"BarretKok simulation failed to converge in {max_attempts} attempts (meas_fid={measurement_fidelity}).")


def _prep_bell_pair_state(
    sim: stim.TableauSimulator,
    a: int, b: int,
    state_label: str,
    p1: float, p2: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
    pauli_2q: Sequence[float] | None,
    bell_pair_fidelity: float,
) -> None:
    """Prepare qubits (a, b) in `state_label` from |00> via stim Cliffords (with gate noise).

    Used by the BarretKok per-pair path (when measurement_fidelity < 1.0).
    Given the symbolic state label produced by _simulate_barret_kok_pair, this
    materializes that exact stabilizer state on the live qubits using a small
    Clifford circuit. All gates carry the configured 1q/2q noise, and Bell
    states (phi+/-, psi+/-) additionally pick up bell_pair_fidelity Werner
    noise via a trailing DEPOLARIZE2.

    State preparation recipes (all starting from |0>|0>):
      'phi+': H(a), CX(a,b)
      'phi-': H(a), CX(a,b), Z(a)            (= apply Z to phi+ on first qubit)
      'psi+': H(a), CX(a,b), X(b)            (= apply X to phi+ on second qubit)
      'psi-': H(a), CX(a,b), X(b), Z(a)
      '00': identity
      '01': X(b)
      '10': X(a)
      '11': X(a), X(b)

    state_label may carry an optional '-' sign prefix encoding an unobservable
    global phase; we strip it. The phase doesn't affect any expectation value
    we care about (Bell-state fidelity correlators).
    """
    if state_label.startswith('-'):
        state_label = state_label[1:]

    circ = stim.Circuit()

    if state_label in {'phi+', 'phi-', 'psi+', 'psi-'}:
        _append_one_qubit_gate(circ, "H", a, p1, channel, pauli_1q)
        _append_two_qubit_gate(circ, "CX", a, b, p2, channel, pauli_2q)
        if state_label == 'phi-':
            _append_one_qubit_gate(circ, "Z", a, p1, channel, pauli_1q)
        elif state_label == 'psi+':
            _append_one_qubit_gate(circ, "X", b, p1, channel, pauli_1q)
        elif state_label == 'psi-':
            _append_one_qubit_gate(circ, "X", b, p1, channel, pauli_1q)
            _append_one_qubit_gate(circ, "Z", a, p1, channel, pauli_1q)
        if bell_pair_fidelity < 1.0:
            f = max(0.0, min(1.0, float(bell_pair_fidelity)))
            p = max(0.0, min(1.0, 16.0 * (1.0 - f) / 15.0))
            if p > 0.0:
                circ.append("DEPOLARIZE2", [a, b], p)
    elif state_label in {'00', '01', '10', '11'}:
        if state_label[0] == '1':
            _append_one_qubit_gate(circ, "X", a, p1, channel, pauli_1q)
        if state_label[1] == '1':
            _append_one_qubit_gate(circ, "X", b, p1, channel, pauli_1q)
    else:
        raise ValueError(f"unknown state_label: {state_label!r}")

    sim.do(circ)


def _build_bell_circuit(
    layout: TwoNodeLayout,
    p1: float, p2: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
    pauli_2q: Sequence[float] | None,
    bell_pair_fidelity: float,
    measurement_fidelity: float = 1.0,
) -> stim.Circuit:
    """Construct 7 |Phi+> Bell pairs across (alice_comm_i, bob_comm_i).

    Used for the FAST PATH at measurement_fidelity=1.0, where no BarretKok BSM
    measurement noise is present and the resulting Bell pair is guaranteed to
    be |Phi+> (modulo bell_pair_fidelity Werner noise). At measurement_fidelity
    < 1.0 the per-shot driver instead calls _simulate_barret_kok_pair +
    _prep_bell_pair_state per pair, which faithfully reproduces the BarretKok
    BSM noise pattern (sequence/components/bsm.py:512-580 +
    barret_kok.py:69-92). The `measurement_fidelity` argument is retained for
    API compatibility but unused here.
    """
    del measurement_fidelity  # noqa: F841 -- consumed by the per-shot path instead
    circ = stim.Circuit()
    for i in range(CODE_N):
        a = layout.alice_comm[i]
        b = layout.bob_comm[i]
        _append_one_qubit_gate(circ, "H", a, p1, channel, pauli_1q)
        _append_two_qubit_gate(circ, "CX", a, b, p2, channel, pauli_2q)
        if bell_pair_fidelity < 1.0:
            f = max(0.0, min(1.0, float(bell_pair_fidelity)))
            p = max(0.0, min(1.0, 16.0 * (1.0 - f) / 15.0))
            if p > 0.0:
                circ.append("DEPOLARIZE2", [a, b], p)
    return circ


def _build_phase_a(
    layout: TwoNodeLayout,
    p2: float,
    channel: str,
    pauli_2q: Sequence[float] | None,
) -> stim.Circuit:
    """TCNOT Phase A: CX(alice_data_i, alice_comm_i) for each i (TeleportedCNOT.py:207-211)."""
    circ = stim.Circuit()
    for i in range(CODE_N):
        _append_two_qubit_gate(circ, "CX", layout.alice_data[i], layout.alice_comm[i], p2, channel, pauli_2q)
    return circ


def _build_phase_b_cx_then_h(
    layout: TwoNodeLayout,
    p1: float, p2: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
    pauli_2q: Sequence[float] | None,
    alice_bits: Sequence[int],
) -> stim.Circuit:
    """TCNOT Phase B: CX(bob_comm_i, bob_data_i) + X(bob_data_i) if alice_bit_i + H(bob_comm_i).

    Alice's measurement bits are baked in (we rebuild this small circuit per
    shot; it's only n=7 CX + ~3.5 X + 7 H).
    """
    circ = stim.Circuit()
    for i in range(CODE_N):
        _append_two_qubit_gate(circ, "CX", layout.bob_comm[i], layout.bob_data[i], p2, channel, pauli_2q)
    for i in range(CODE_N):
        if alice_bits[i] == 1:
            _append_one_qubit_gate(circ, "X", layout.bob_data[i], p1, channel, pauli_1q)
    for i in range(CODE_N):
        _append_one_qubit_gate(circ, "H", layout.bob_comm[i], p1, channel, pauli_1q)
    return circ


def _build_phase_c(
    layout: TwoNodeLayout,
    p1: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
    bob_bits: Sequence[int],
) -> stim.Circuit:
    """TCNOT Phase C: Z(alice_data_i) if bob_bit_i (TeleportedCNOT.py:312-316)."""
    circ = stim.Circuit()
    for i in range(CODE_N):
        if bob_bits[i] == 1:
            _append_one_qubit_gate(circ, "Z", layout.alice_data[i], p1, channel, pauli_1q)
    return circ


def _build_ft_check_circuit(
    block_qubits: Sequence[int],
    ancilla: int,
    support: Sequence[int],
    p2: float,
    channel: str,
    pauli_2q: Sequence[float] | None,
) -> stim.Circuit:
    """One FT check: noisy CX from each support qubit to the ancilla."""
    circ = stim.Circuit()
    for local_idx in support:
        _append_two_qubit_gate(circ, "CX", block_qubits[local_idx], ancilla, p2, channel, pauli_2q)
    return circ


def _make_block_pauli(num_qubits: int, block_qubits: Sequence[int], pauli_string: str) -> stim.PauliString:
    """Build a PauliString supported on `block_qubits` (rest identity)."""
    obs = stim.PauliString(num_qubits)
    for q, p in zip(block_qubits, pauli_string):
        if p != "I":
            obs[q] = p
    return obs


def _make_joint_pauli(
    num_qubits: int,
    left_block: Sequence[int],
    right_block: Sequence[int],
    pauli_string: str,
) -> stim.PauliString:
    """Build a joint PauliString supported on both blocks with the same pattern."""
    obs = stim.PauliString(num_qubits)
    for q, p in zip(left_block, pauli_string):
        if p != "I":
            obs[q] = p
    for q, p in zip(right_block, pauli_string):
        if p != "I":
            obs[q] = p
    return obs


# ---------------------------------------------------------------------------
# Precompiled pipeline (reused across shots)
# ---------------------------------------------------------------------------

@dataclass
class _CompiledPipeline:
    """All stim objects needed to run shots without per-shot Python work."""
    layout: TwoNodeLayout
    num_qubits: int
    alice_ancilla: int
    bob_ancilla: int

    # Noise / channel parameters (cached so we can rebuild per-shot phase B/C circuits cheaply).
    p1: float
    p2: float
    channel: str
    pauli_1q: Sequence[float] | None
    pauli_2q: Sequence[float] | None
    measurement_flip_prob: float
    state_prep_flip_prob: float    # X_ERROR probability after each set_to_zero (mirrors quantum_manager.py:1948).
    bell_pair_fidelity: float       # Werner-state fidelity used by the BarretKok per-shot path.
    photon_loss: float              # Per-photon "fails to trigger" probability for BarretKok BSM (depends on link_distance_km).
    ft_prep_mode: str
    ft_max_retries: int
    correction_mode: str

    # Precompiled stim circuits (constant per call to simulate_logical_pairs_2node).
    encoder_alice: stim.Circuit
    encoder_bob: stim.Circuit
    transversal_h_alice: stim.Circuit  # |0>_L -> |+>_L
    bell_circuit: stim.Circuit
    phase_a_circuit: stim.Circuit
    ft_check_circuits_alice: tuple[stim.Circuit, ...]
    ft_check_circuits_bob: tuple[stim.Circuit, ...]
    ft_supports_alice: tuple[tuple[int, ...], ...]  # used only when ft_prep_mode != "none"
    ft_supports_bob: tuple[tuple[int, ...], ...]

    # Precompiled Pauli observables for the noiseless recovery + fidelity peek.
    alice_x_stab_paulis: tuple[stim.PauliString, ...]
    alice_z_stab_paulis: tuple[stim.PauliString, ...]
    bob_x_stab_paulis: tuple[stim.PauliString, ...]
    bob_z_stab_paulis: tuple[stim.PauliString, ...]
    logical_xx: stim.PauliString
    logical_yy: stim.PauliString
    logical_zz: stim.PauliString


def _compile_pipeline(params: dict[str, object]) -> _CompiledPipeline:
    """Build the precompiled-circuits bundle for one parameter set."""
    layout = make_two_node_layout()
    num_qubits = layout.total_data_and_comm + 2  # +2 for FT-prep ancillas
    alice_ancilla = layout.total_data_and_comm
    bob_ancilla = layout.total_data_and_comm + 1

    p1 = one_qubit_error_prob(float(params["gate_fidelity"]))
    p2 = two_qubit_error_prob(float(params["two_qubit_gate_fidelity"]))
    channel = str(params["gate_error_channel"])
    pauli_1q = params["pauli_1q_weights"]
    pauli_2q = params["pauli_2q_weights"]
    measurement_flip_prob = max(0.0, 1.0 - float(params["measurement_fidelity"]))
    state_prep_flip_prob = max(0.0, 1.0 - float(params["state_preparation_fidelity"]))
    photon_loss = compute_photon_loss(float(params["link_distance_km"]))
    ft_prep_mode = str(params["ft_prep_mode"])
    ft_max_retries = int(params["ft_max_retries"])
    correction_mode = str(params["correction_mode"])
    bell_fid = float(params["physical_bell_pair_fidelity"])

    if ft_prep_mode not in {"none", "minimal", "standard"}:
        raise ValueError(f"Unknown ft_prep_mode {ft_prep_mode!r}.")
    if correction_mode not in {"none", "cec", "qec", "qec+cec"}:
        raise ValueError(f"Unknown correction_mode {correction_mode!r}.")

    if ft_prep_mode == "none":
        ft_supports: tuple[tuple[int, ...], ...] = ()
    elif ft_prep_mode == "minimal":
        ft_supports = (STEANE_FT_CHECK_SUPPORTS_STANDARD[1],)
    else:  # "standard"
        ft_supports = STEANE_FT_CHECK_SUPPORTS_STANDARD

    encoder_alice = _build_steane_encoder(layout.alice_data, p1, p2, channel, pauli_1q, pauli_2q)
    encoder_bob = _build_steane_encoder(layout.bob_data, p1, p2, channel, pauli_1q, pauli_2q)
    transversal_h_alice = _build_transversal_h(layout.alice_data, p1, channel, pauli_1q)
    bell_circuit = _build_bell_circuit(layout, p1, p2, channel, pauli_1q, pauli_2q, bell_fid,
                                       measurement_fidelity=float(params["measurement_fidelity"]))
    phase_a_circuit = _build_phase_a(layout, p2, channel, pauli_2q)

    ft_check_circuits_alice = tuple(
        _build_ft_check_circuit(layout.alice_data, alice_ancilla, support, p2, channel, pauli_2q)
        for support in ft_supports
    )
    ft_check_circuits_bob = tuple(
        _build_ft_check_circuit(layout.bob_data, bob_ancilla, support, p2, channel, pauli_2q)
        for support in ft_supports
    )

    alice_x_stab_paulis = tuple(_make_block_pauli(num_qubits, layout.alice_data, s) for s in STEANE_X_STABILIZERS)
    alice_z_stab_paulis = tuple(_make_block_pauli(num_qubits, layout.alice_data, s) for s in STEANE_Z_STABILIZERS)
    bob_x_stab_paulis = tuple(_make_block_pauli(num_qubits, layout.bob_data, s) for s in STEANE_X_STABILIZERS)
    bob_z_stab_paulis = tuple(_make_block_pauli(num_qubits, layout.bob_data, s) for s in STEANE_Z_STABILIZERS)
    logical_xx = _make_joint_pauli(num_qubits, layout.alice_data, layout.bob_data, STEANE_LX)
    logical_yy = _make_joint_pauli(num_qubits, layout.alice_data, layout.bob_data, STEANE_LY)
    logical_zz = _make_joint_pauli(num_qubits, layout.alice_data, layout.bob_data, STEANE_LZ)

    return _CompiledPipeline(
        layout=layout,
        num_qubits=num_qubits,
        alice_ancilla=alice_ancilla,
        bob_ancilla=bob_ancilla,
        p1=p1, p2=p2,
        channel=channel,
        pauli_1q=pauli_1q, pauli_2q=pauli_2q,
        measurement_flip_prob=measurement_flip_prob,
        state_prep_flip_prob=state_prep_flip_prob,
        bell_pair_fidelity=bell_fid,
        photon_loss=photon_loss,
        ft_prep_mode=ft_prep_mode,
        ft_max_retries=ft_max_retries,
        correction_mode=correction_mode,
        encoder_alice=encoder_alice,
        encoder_bob=encoder_bob,
        transversal_h_alice=transversal_h_alice,
        bell_circuit=bell_circuit,
        phase_a_circuit=phase_a_circuit,
        ft_check_circuits_alice=ft_check_circuits_alice,
        ft_check_circuits_bob=ft_check_circuits_bob,
        ft_supports_alice=ft_supports,
        ft_supports_bob=ft_supports,
        alice_x_stab_paulis=alice_x_stab_paulis,
        alice_z_stab_paulis=alice_z_stab_paulis,
        bob_x_stab_paulis=bob_x_stab_paulis,
        bob_z_stab_paulis=bob_z_stab_paulis,
        logical_xx=logical_xx,
        logical_yy=logical_yy,
        logical_zz=logical_zz,
    )


# ---------------------------------------------------------------------------
# Shot driver
# ---------------------------------------------------------------------------

def _peek_sign_bit(sim: stim.TableauSimulator, observable: stim.PauliString) -> int:
    """Read a stabilizer's sign without collapsing the state.

    Used during noiseless endpoint recovery to read syndromes off a copy of
    the live tableau. peek_observable_expectation returns +1 / -1 (or 0 if
    indeterminate); we map +1 -> 0 (no syndrome) and -1 -> 1 (syndrome). For
    Pauli noise channels the stabilizers are always deterministic so we
    never expect 0; if we ever see it, the simulator state has drifted out
    of the codespace which is a bug.
    """
    expectation = float(sim.peek_observable_expectation(observable))
    if expectation > 0.5:
        return 0
    if expectation < -0.5:
        return 1
    raise RuntimeError(f"Non-deterministic observable expectation = {expectation}")


def _reset_with_prep_noise(sim: stim.TableauSimulator, qubits: Sequence[int], state_prep_flip_prob: float) -> None:
    """Reset each qubit, then apply X_ERROR(state_prep_flip_prob) (mirrors quantum_manager.py:1948).

    At state_preparation_fidelity = 1.0 (state_prep_flip_prob = 0) this is just a no-noise reset
    matching the cheating's previous behavior; otherwise each freshly reset qubit ends up in
    |1> instead of |0> with probability state_prep_flip_prob, identical to SeQUeNCe's set_to_zero.
    """
    if state_prep_flip_prob <= 0.0:
        for q in qubits:
            sim.reset(q)
        return
    circ = stim.Circuit()
    for q in qubits:
        circ.append("R", [q])
        circ.append("X_ERROR", [q], state_prep_flip_prob)
    sim.do(circ)


def _prep_block(
    sim: stim.TableauSimulator,
    pipe: _CompiledPipeline,
    block_qubits: Sequence[int],
    ancilla: int,
    encoder: stim.Circuit,
    ft_checks: Sequence[stim.Circuit],
    transversal_h: stim.Circuit | None,
    rng: np.random.Generator | None = None,
) -> bool:
    """Encode one Steane block, optionally with FT-prep retry. Mirrors
    css_codes.run_encode_ft_prep (css_codes.py:213-296).

    Outline of one retry shot:
      1. Reset the 7 data qubits + the ancilla, applying X_ERROR with probability
         (1 - state_preparation_fidelity) to mirror SeQUeNCe's set_to_zero.
      2. Run the Paetznick-Reichardt encoder -> data block now in |0>_L.
      3. For each FT-check support: reset ancilla, run the support's CNOTs,
         measure ancilla (with flip noise). If any check measures 1, the shot
         is rejected and we retry. If all measure 0, the shot is accepted.
      4. If accepted and transversal_h is provided, apply transversal H to
         convert |0>_L -> |+>_L.

    All measurement noise sources mirror SeQUeNCe's behavior:
      * Reset noise: state_preparation_fidelity -> X_ERROR (quantum_manager.py:1948)
      * Ancilla readout noise: measurement_fidelity -> bit flip
        (quantum_manager_tableau.py:184). PRIOR TO FIXING this, the cheating
        was silent on FT-ancilla readout noise -- the flip code below was the
        bug fix that closed the meas_fid<1 + ft-prep gap.

    Returns True if any of the up-to-ft_max_retries shots was accepted.
    `rng` must be supplied when measurement_flip_prob > 0.
    """
    measurement_flip_prob = pipe.measurement_flip_prob
    state_prep_flip_prob = pipe.state_prep_flip_prob
    for _shot in range(pipe.ft_max_retries):
        _reset_with_prep_noise(sim, list(block_qubits) + [ancilla], state_prep_flip_prob)

        sim.do(encoder)

        if not ft_checks:
            if transversal_h is not None:
                sim.do(transversal_h)
            return True

        accepted = True
        for check in ft_checks:
            _reset_with_prep_noise(sim, [ancilla], state_prep_flip_prob)
            sim.do(check)
            bit = int(sim.measure(ancilla))
            if measurement_flip_prob > 0.0:
                if rng is None:
                    raise RuntimeError("_prep_block: rng must be provided when measurement_flip_prob > 0")
                if float(rng.random()) < measurement_flip_prob:
                    bit ^= 1
            if bit != 0:
                accepted = False
                break
        if accepted:
            if transversal_h is not None:
                sim.do(transversal_h)
            return True
    return False


def _run_shot(pipe: _CompiledPipeline, sim_seed: int | None, rng: np.random.Generator) -> float:
    """Run one 2-node logical-pair shot through the precompiled pipeline.

    Sequence (per-pair, mirrors n_node_logical_pair_with_app for line_2_2G.json):
      1. Encode Alice in |+>_L and Bob in |0>_L (both with optional FT-prep).
      2. Generate 7 Bell pairs across (alice_comm, bob_comm).
      3. Phase A: CX(alice_data -> alice_comm), measure alice_comm; report alice_bits.
      4. Phase B: CX(bob_comm -> bob_data); X(bob_data) if alice_bit==1; H(bob_comm);
                  measure bob_comm; report bob_bits.
      5. Phase C: Z(alice_data) if bob_bit==1.
      6. Noiseless endpoint recovery on each block (peek stabilizers, decode,
         apply correction).
      7. Compute end-to-end fidelity from logical XX, YY, ZZ correlators
         between alice_data and bob_data:  F = (1 + <XbarXbar> - <YbarYbar> + <ZbarZbar>) / 4.

    Mirrors RequestLogicalPairApp + TeleportedCNOT exactly. Returns the per-shot
    Bell-state fidelity in {0, 0.5, 1}; only 0 and 1 occur for noiseless or pure
    Pauli noise. Average over many shots = avg_end_to_end_logical_corrected.
    """
    sim = stim.TableauSimulator(seed=sim_seed)
    sim.set_num_qubits(pipe.num_qubits)

    # Step 1. Encode Alice (right-peer-facing block -> |+>_L) and Bob (|0>_L).
    if not _prep_block(sim, pipe, pipe.layout.alice_data, pipe.alice_ancilla,
                       pipe.encoder_alice, pipe.ft_check_circuits_alice, pipe.transversal_h_alice,
                       rng=rng):
        raise RuntimeError("FT prep failed within ft_max_retries shots (Alice).")
    if not _prep_block(sim, pipe, pipe.layout.bob_data, pipe.bob_ancilla,
                       pipe.encoder_bob, pipe.ft_check_circuits_bob, None,
                       rng=rng):
        raise RuntimeError("FT prep failed within ft_max_retries shots (Bob).")

    # Bell pairs. At measurement_fidelity = 1.0 the precompiled bell_circuit is
    # exact (|Phi+>^7 modulo bell_pair_fidelity Werner noise). At < 1.0 we drop
    # to the per-pair BarretKok simulation that mirrors sequence/components/bsm.py.
    if pipe.measurement_flip_prob > 0.0:
        for i in range(CODE_N):
            label, _r0, _r1 = _simulate_barret_kok_pair(
                rng, 1.0 - pipe.measurement_flip_prob, photon_loss=pipe.photon_loss,
            )
            _prep_bell_pair_state(sim, pipe.layout.alice_comm[i], pipe.layout.bob_comm[i],
                                  label, pipe.p1, pipe.p2, pipe.channel,
                                  pipe.pauli_1q, pipe.pauli_2q,
                                  bell_pair_fidelity=pipe.bell_pair_fidelity)
    else:
        sim.do(pipe.bell_circuit)

    # TCNOT Phase A.
    sim.do(pipe.phase_a_circuit)
    alice_bits = [int(sim.measure(pipe.layout.alice_comm[i])) for i in range(CODE_N)]
    if pipe.measurement_flip_prob > 0.0:
        flips = rng.random(CODE_N) < pipe.measurement_flip_prob
        alice_bits = [b ^ int(f) for b, f in zip(alice_bits, flips)]

    # Phase B (per-shot circuit because the X corrections depend on alice_bits).
    phase_b = _build_phase_b_cx_then_h(
        pipe.layout, pipe.p1, pipe.p2, pipe.channel,
        pipe.pauli_1q, pipe.pauli_2q, alice_bits,
    )
    sim.do(phase_b)
    bob_bits = [int(sim.measure(pipe.layout.bob_comm[i])) for i in range(CODE_N)]
    if pipe.measurement_flip_prob > 0.0:
        flips = rng.random(CODE_N) < pipe.measurement_flip_prob
        bob_bits = [b ^ int(f) for b, f in zip(bob_bits, flips)]

    # Phase C (per-shot circuit because Z corrections depend on bob_bits).
    phase_c = _build_phase_c(pipe.layout, pipe.p1, pipe.channel, pipe.pauli_1q, bob_bits)
    sim.do(phase_c)

    # Noiseless endpoint recovery (Alice + Bob blocks).
    if pipe.correction_mode in {"cec", "qec", "qec+cec"}:
        # Alice block.
        x_synd_a = tuple(_peek_sign_bit(sim, obs) for obs in pipe.alice_x_stab_paulis)
        z_synd_a = tuple(_peek_sign_bit(sim, obs) for obs in pipe.alice_z_stab_paulis)
        x_err_a = STEANE_DECODE_TABLE[z_synd_a]   # Z stabilizers locate X errors
        z_err_a = STEANE_DECODE_TABLE[x_synd_a]   # X stabilizers locate Z errors
        if x_err_a is not None:
            sim.x(pipe.layout.alice_data[int(x_err_a)])
        if z_err_a is not None:
            sim.z(pipe.layout.alice_data[int(z_err_a)])
        # Bob block.
        x_synd_b = tuple(_peek_sign_bit(sim, obs) for obs in pipe.bob_x_stab_paulis)
        z_synd_b = tuple(_peek_sign_bit(sim, obs) for obs in pipe.bob_z_stab_paulis)
        x_err_b = STEANE_DECODE_TABLE[z_synd_b]
        z_err_b = STEANE_DECODE_TABLE[x_synd_b]
        if x_err_b is not None:
            sim.x(pipe.layout.bob_data[int(x_err_b)])
        if z_err_b is not None:
            sim.z(pipe.layout.bob_data[int(z_err_b)])

    cx = float(sim.peek_observable_expectation(pipe.logical_xx))
    cy = float(sim.peek_observable_expectation(pipe.logical_yy))
    cz = float(sim.peek_observable_expectation(pipe.logical_zz))
    return (1.0 + cx - cy + cz) / 4.0


# ---------------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------------

def _resolve_params(overrides: dict | None) -> dict[str, object]:
    """Merge `overrides` into DEFAULT_PARAMS without mutating the default."""
    merged = dict(DEFAULT_PARAMS)
    if overrides:
        for k, v in overrides.items():
            merged[k] = v
    return merged


def simulate_logical_pair_2node(
    params: dict | None = None,
    seed: int | None = None,
) -> tuple[float, _CompiledPipeline]:
    """Single-shot helper. Returns (fidelity, pipeline) -- pipeline is for reuse."""
    p = _resolve_params(params)
    pipe = _compile_pipeline(p)
    rng = np.random.default_rng(None if seed is None else seed + 0xC0FFEE)
    return _run_shot(pipe, seed, rng), pipe


def simulate_logical_pairs_2node(
    num_pairs: int,
    params: dict | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Run `num_pairs` independent shots and aggregate metrics.

    Output schema mirrors main_test's per-worker metrics so the parallel
    sweep can swap this in unchanged.
    """
    p = _resolve_params(params)
    if num_pairs <= 0:
        return _empty_metrics(p)

    pipe = _compile_pipeline(p)
    rng = np.random.default_rng(seed)
    fidelities = np.empty(num_pairs, dtype=np.float64)
    completed = 0
    for shot in range(num_pairs):
        shot_seed = None if seed is None else int(rng.integers(0, 2**63 - 1))
        try:
            value = _run_shot(pipe, shot_seed, rng)
        except RuntimeError:
            fidelities[shot] = np.nan
            continue
        fidelities[shot] = value
        completed += 1

    finite = fidelities[np.isfinite(fidelities)]
    avg = float(np.mean(finite)) if finite.size else float("nan")

    return {
        "num_workers": 1,
        "num_logical_pairs_requested": int(num_pairs),
        "num_logical_pairs_completed": int(completed),
        "avg_end_to_end_logical_raw": float("nan"),
        "avg_end_to_end_logical_corrected": avg,
        "avg_latency_ps": float("nan"),
        "avg_latency_s": float("nan"),
        "avg_throughput_pairs_per_s": float("nan"),
        "two_qubit_gate_fidelity": float(p["two_qubit_gate_fidelity"]),
        "per_shot_fidelities": fidelities.tolist(),
    }


def _empty_metrics(p: dict[str, object]) -> dict[str, object]:
    return {
        "num_workers": 0,
        "num_logical_pairs_requested": 0,
        "num_logical_pairs_completed": 0,
        "avg_end_to_end_logical_raw": float("nan"),
        "avg_end_to_end_logical_corrected": float("nan"),
        "avg_latency_ps": float("nan"),
        "avg_latency_s": float("nan"),
        "avg_throughput_pairs_per_s": float("nan"),
        "two_qubit_gate_fidelity": float(p["two_qubit_gate_fidelity"]),
        "per_shot_fidelities": [],
    }


# ---------------------------------------------------------------------------
# N-node generalization (mirrors n_node_logical_pair_with_app + QREProtocol)
# ---------------------------------------------------------------------------
#
# Protocol (matches RequestLogicalPairApp.py + QREProtocol.py for an N-router
# linear chain, e.g. config/standard_configs/line_6_2G.json):
#
#   1. Encoding (RequestLogicalPairApp.encode_data_qubits, line 830-924):
#      Each block is Steane-encoded; the block facing the right peer goes to
#      |+>_L (transversal H after |0>_L), the block facing the left peer
#      stays |0>_L. Edge nodes hold one block; middle nodes hold two (L+R).
#   2. Per-hop teleported CNOT (TeleportedCNOT.py, hop k between routers
#      k and k+1, control = right block of k, target = left block of k+1).
#      Identical Phase A / B / C as the 2-node case, replicated per hop.
#   3. Middle-node encoded swap (QREProtocol.run_encoded_swapping, line 216-310):
#      transversal CX(L_i -> R_i), then H on each L_i qubit, then measure all
#      14 qubits. Decode via Steane syndromes (decode_middle_bsm, css_codes.py:491).
#      Output (b_x_corrected, b_z_corrected) becomes the per-middle-node frame
#      contribution: bx_i = b_z_corrected, bz_i = b_x_corrected.
#   4. Initiator frame correction (QREProtocol.update_pauli_frame, line 339-380):
#      final_bx = sum(bx_i) mod 2; final_bz = sum(bz_i) mod 2. Apply X^final_bx
#      and Z^final_bz transversally on router_0's data block (transversal X = LX,
#      transversal Z = LZ for the Steane code).
#   5. Noiseless endpoint stabilizer recovery on both endpoint blocks
#      (RequestLogicalPairApp._calculate_pair_fidelity recover_block, line 395-462).
#   6. Logical fidelity F = (1 + <X̄X̄> - <ȲȲ> + <Z̄Z̄>) / 4 between the two
#      endpoint blocks (line 502).
#
# correction_mode handling:
#   "cec":      no pre-swap QEC; classical correction inside decode_middle_bsm; endpoint recovery
#   "none":     no pre-swap QEC; no classical correction at swap; no endpoint recovery
#   "qec"/"qec+cec": NOT supported in cheating mode (would require pre-swap QEC at every middle node)

# Steane parity-check rows (mirror css_codes.py STEANE_SYNDROME_ROWS).
# Row i is the support of the i-th X/Z stabilizer (CSS code: X- and Z-stabilizers
# share supports). With STEANE_DECODE_TABLE indexing (s0, s1, s2), a single
# X (or Z) error on physical qubit q produces syndrome
#     (parity over row 0 of {q in row?}, parity over row 1, parity over row 2)
# which decodes back to q.
STEANE_SYNDROME_ROWS: tuple[tuple[int, ...], ...] = (
    (3, 4, 5, 6),  # IIIXXXX
    (1, 2, 5, 6),  # IXXIIXX
    (0, 2, 4, 6),  # XIXIXIX
)


def _decode_middle_bsm(
    left_x_bits: Sequence[int],
    right_z_bits: Sequence[int],
    correction_mode: str,
) -> tuple[int, int]:
    """Decode an encoded BSM at one middle node (mirrors css_codes.py:491-528).

    The middle-node encoded swap reads 14 bits: 7 from X-basis measurement of the
    left block (which detects Z errors on the left block) and 7 from Z-basis
    measurement of the right block (which detects X errors on the right block).
    From these 14 bits we want two outputs:

      b_x_corrected = parity of the left-block X-basis bits, possibly with one
                      bit XOR-flipped to undo a single-qubit Z error inferred
                      from the syndrome.
      b_z_corrected = parity of the right-block Z-basis bits, similarly with
                      possible single-qubit X-error correction applied.

    These become the X-frame and Z-frame contributions of this middle node.
    Multiple middle nodes' frame contributions XOR-accumulate at the initiator,
    which then applies a transversal X^bx Z^bz correction to pin the final
    end-to-end Bell pair to |Phi+>_L.

    Correction modes:
      "cec" / "qec+cec": classical-only correction is applied inside this function
                          (the parity gets XOR'd to undo the inferred error).
      "none" / "qec":     parities are returned as-is, no correction.

    The XOR-by-1 trick at the end is mathematically equivalent to css_codes.py's
    "flip the bit at the syndrome-decoded position and recompute the parity"
    formulation, but skips the bit-flip step.

    Returns:
        (b_x_corrected, b_z_corrected) -- the per-middle-node frame contributions.
        The caller does the QREProtocol.py:280-281 mapping
        (bx <- b_z_corrected, bz <- b_x_corrected) before XOR-accumulating.
    """
    # Compute the 3-bit Steane syndromes by parity over each stabilizer's support.
    s_x = tuple(sum(int(left_x_bits[i]) for i in row) & 1 for row in STEANE_SYNDROME_ROWS)
    s_z = tuple(sum(int(right_z_bits[i]) for i in row) & 1 for row in STEANE_SYNDROME_ROWS)

    # Raw parity (without any correction) of the 7 measurement bits per side.
    x_parity = sum(int(b) for b in left_x_bits) & 1
    z_parity = sum(int(b) for b in right_z_bits) & 1

    if correction_mode in {"cec", "qec+cec"}:
        # If the syndrome is non-zero, the decoder identifies a single qubit to
        # correct. Flipping that one bit changes the parity by exactly 1.
        if STEANE_DECODE_TABLE.get(s_x) is not None:
            x_parity ^= 1
        if STEANE_DECODE_TABLE.get(s_z) is not None:
            z_parity ^= 1

    return x_parity, z_parity


@dataclass(frozen=True)
class NNodeLayout:
    """Qubit indices for an N-node linear chain (router_0 ... router_{N-1}).

    Per-hop accessors (k = 0 .. N-2):
      alice_data[k] : right-facing block of router_k (TCNOT control)
      alice_comm[k] : right-facing comm register of router_k
      bob_data[k]   : left-facing block of router_{k+1} (TCNOT target)
      bob_comm[k]   : left-facing comm register of router_{k+1}
      alice_ancilla[k], bob_ancilla[k] : per-block FT-prep ancilla qubits

    For middle node i (1 <= i <= N-2), the left block is bob_data[i-1] and
    the right block is alice_data[i] (both are 7-qubit blocks at the same
    router but with disjoint qubit indices).

    Endpoints expose a single block:
      router_0 (initiator) block      = alice_data[0]
      router_{N-1} (responder) block  = bob_data[N-2]
    """
    num_nodes: int
    alice_data: tuple[tuple[int, ...], ...]
    alice_comm: tuple[tuple[int, ...], ...]
    bob_data: tuple[tuple[int, ...], ...]
    bob_comm: tuple[tuple[int, ...], ...]
    alice_ancilla: tuple[int, ...]
    bob_ancilla: tuple[int, ...]
    total_qubits: int

    @property
    def num_links(self) -> int:
        return self.num_nodes - 1

    @property
    def initiator_block(self) -> tuple[int, ...]:
        return self.alice_data[0]

    @property
    def responder_block(self) -> tuple[int, ...]:
        return self.bob_data[-1]


def make_n_node_layout(num_nodes: int) -> NNodeLayout:
    """Allocate qubit indices for an N-node chain (matches the line_N_2G.json topology).

    Each router gets 7 data + 7 comm + 1 ancilla per neighbor it has. So:
      router_0:        7 data + 7 comm + 1 ancilla              = 15 qubits (right-facing only)
      router_i (mid):  14 data + 14 comm + 2 ancillas           = 30 qubits (left+right)
      router_{N-1}:    7 data + 7 comm + 1 ancilla              = 15 qubits (left-facing only)
    For N nodes total: 30 + 30*(N-2) = 30*(N-1) physical qubits.
    """
    if num_nodes < 2:
        raise ValueError(f"num_nodes must be >= 2, got {num_nodes}")

    n = CODE_N
    next_q = 0

    def _alloc_block() -> tuple[int, ...]:
        nonlocal next_q
        block = tuple(range(next_q, next_q + n))
        next_q += n
        return block

    def _alloc_one() -> int:
        nonlocal next_q
        q = next_q
        next_q += 1
        return q

    # per_router[k] holds the qubit allocations for router k. Endpoints have only
    # one set of (data, comm, ancilla). We allocate left-facing first, then
    # right-facing, so middle-router qubit ranges stay contiguous.
    per_router_left: list[tuple[tuple[int, ...], tuple[int, ...], int] | None] = [None] * num_nodes
    per_router_right: list[tuple[tuple[int, ...], tuple[int, ...], int] | None] = [None] * num_nodes

    for k in range(num_nodes):
        is_left_edge = (k == 0)
        is_right_edge = (k == num_nodes - 1)
        if not is_left_edge:
            data = _alloc_block()
            comm = _alloc_block()
            anc = _alloc_one()
            per_router_left[k] = (data, comm, anc)
        if not is_right_edge:
            data = _alloc_block()
            comm = _alloc_block()
            anc = _alloc_one()
            per_router_right[k] = (data, comm, anc)

    alice_data: list[tuple[int, ...]] = []
    alice_comm: list[tuple[int, ...]] = []
    bob_data: list[tuple[int, ...]] = []
    bob_comm: list[tuple[int, ...]] = []
    alice_anc: list[int] = []
    bob_anc: list[int] = []
    for k in range(num_nodes - 1):
        right = per_router_right[k]
        left = per_router_left[k + 1]
        if right is None or left is None:
            raise RuntimeError("internal: missing router allocation in make_n_node_layout")
        alice_data.append(right[0]); alice_comm.append(right[1]); alice_anc.append(right[2])
        bob_data.append(left[0]);    bob_comm.append(left[1]);    bob_anc.append(left[2])

    return NNodeLayout(
        num_nodes=num_nodes,
        alice_data=tuple(alice_data),
        alice_comm=tuple(alice_comm),
        bob_data=tuple(bob_data),
        bob_comm=tuple(bob_comm),
        alice_ancilla=tuple(alice_anc),
        bob_ancilla=tuple(bob_anc),
        total_qubits=next_q,
    )


def _build_swap_bsm_circuit(
    left_block: Sequence[int],
    right_block: Sequence[int],
    p1: float, p2: float,
    channel: str,
    pauli_1q: Sequence[float] | None,
    pauli_2q: Sequence[float] | None,
) -> stim.Circuit:
    """Encoded BSM circuit at a middle node (QREProtocol.py:247-258).

    Transversal CX(left_i -> right_i), then transversal H on left, then
    measure all 14 qubits. The 7 left bits are X-basis outcomes, the 7
    right bits are Z-basis outcomes. The per-shot driver runs this circuit
    and reads the 14 measurement results in order.
    """
    if len(left_block) != CODE_N or len(right_block) != CODE_N:
        raise ValueError("BSM circuit requires Steane-sized blocks (CODE_N qubits each).")
    circ = stim.Circuit()
    for i in range(CODE_N):
        _append_two_qubit_gate(circ, "CX", left_block[i], right_block[i], p2, channel, pauli_2q)
    for i in range(CODE_N):
        _append_one_qubit_gate(circ, "H", left_block[i], p1, channel, pauli_1q)
    return circ


@dataclass
class _CompiledPipelineN:
    """Precompiled n-node pipeline; built once per (num_nodes, params) combo."""
    layout: NNodeLayout
    num_qubits: int

    p1: float
    p2: float
    channel: str
    pauli_1q: Sequence[float] | None
    pauli_2q: Sequence[float] | None
    measurement_flip_prob: float
    state_prep_flip_prob: float    # X_ERROR probability after each set_to_zero (mirrors quantum_manager.py:1948).
    bell_pair_fidelity: float       # Werner-state fidelity used by the BarretKok per-shot path.
    photon_loss: float              # Per-photon "fails to trigger" probability for BarretKok BSM (depends on link_distance_km).
    ft_prep_mode: str
    ft_max_retries: int
    correction_mode: str

    # Per-block encoders (length 2*(num_nodes-1)) and matching transversal-H circuits.
    # Order: for hop k, encoder_alice[k] / encoder_bob[k] paired with transversal_h_alice[k] (None-equivalent for bob).
    encoder_alice: tuple[stim.Circuit, ...]
    encoder_bob: tuple[stim.Circuit, ...]
    transversal_h_alice: tuple[stim.Circuit, ...]  # |0>_L -> |+>_L on the alice block of each hop

    # Per-hop bell-pair + TCNOT-Phase-A circuits.
    bell_circuits: tuple[stim.Circuit, ...]
    phase_a_circuits: tuple[stim.Circuit, ...]

    # Per-block FT-check circuits.
    ft_check_circuits_alice: tuple[tuple[stim.Circuit, ...], ...]  # length num_links; each inner tuple length len(ft_supports)
    ft_check_circuits_bob: tuple[tuple[stim.Circuit, ...], ...]
    ft_supports: tuple[tuple[int, ...], ...]

    # Per-middle-node encoded BSM circuit (length num_nodes - 2; empty if num_nodes == 2).
    bsm_circuits: tuple[stim.Circuit, ...]
    # Per-middle-node block references (left, right) for measurement reads (length num_nodes - 2).
    bsm_left_blocks: tuple[tuple[int, ...], ...]
    bsm_right_blocks: tuple[tuple[int, ...], ...]

    # Endpoint-recovery observables (Pauli strings on the full register).
    initiator_x_stab_paulis: tuple[stim.PauliString, ...]
    initiator_z_stab_paulis: tuple[stim.PauliString, ...]
    responder_x_stab_paulis: tuple[stim.PauliString, ...]
    responder_z_stab_paulis: tuple[stim.PauliString, ...]
    logical_xx: stim.PauliString
    logical_yy: stim.PauliString
    logical_zz: stim.PauliString


def _compile_pipeline_n_node(num_nodes: int, params: dict[str, object]) -> _CompiledPipelineN:
    """Build the precompiled-circuits bundle for an N-node chain."""
    layout = make_n_node_layout(num_nodes)
    num_qubits = layout.total_qubits

    p1 = one_qubit_error_prob(float(params["gate_fidelity"]))
    p2 = two_qubit_error_prob(float(params["two_qubit_gate_fidelity"]))
    channel = str(params["gate_error_channel"])
    pauli_1q = params["pauli_1q_weights"]
    pauli_2q = params["pauli_2q_weights"]
    measurement_flip_prob = max(0.0, 1.0 - float(params["measurement_fidelity"]))
    state_prep_flip_prob = max(0.0, 1.0 - float(params["state_preparation_fidelity"]))
    photon_loss = compute_photon_loss(float(params["link_distance_km"]))
    ft_prep_mode = str(params["ft_prep_mode"])
    ft_max_retries = int(params["ft_max_retries"])
    correction_mode = str(params["correction_mode"])
    bell_fid = float(params["physical_bell_pair_fidelity"])

    if ft_prep_mode not in {"none", "minimal", "standard"}:
        raise ValueError(f"Unknown ft_prep_mode {ft_prep_mode!r}.")
    if correction_mode not in {"none", "cec", "qec", "qec+cec"}:
        raise ValueError(f"Unknown correction_mode {correction_mode!r}.")
    if num_nodes > 2 and correction_mode in {"qec", "qec+cec"}:
        # Pre-swap QEC at middle nodes is not implemented in the cheating module.
        # The default sweep uses correction_mode='cec', so this is a deliberate
        # NotImplemented rather than a silent skip. Implement run_qec_round-equivalent
        # before re-enabling.
        raise NotImplementedError(
            f"correction_mode={correction_mode!r} is not implemented for num_nodes>2 in the cheating module; "
            "use 'cec' or 'none', or extend _run_shot_n_node with pre-swap QEC."
        )

    if ft_prep_mode == "none":
        ft_supports: tuple[tuple[int, ...], ...] = ()
    elif ft_prep_mode == "minimal":
        ft_supports = (STEANE_FT_CHECK_SUPPORTS_STANDARD[1],)
    else:
        ft_supports = STEANE_FT_CHECK_SUPPORTS_STANDARD

    encoder_alice = tuple(
        _build_steane_encoder(layout.alice_data[k], p1, p2, channel, pauli_1q, pauli_2q)
        for k in range(layout.num_links)
    )
    encoder_bob = tuple(
        _build_steane_encoder(layout.bob_data[k], p1, p2, channel, pauli_1q, pauli_2q)
        for k in range(layout.num_links)
    )
    transversal_h_alice = tuple(
        _build_transversal_h(layout.alice_data[k], p1, channel, pauli_1q)
        for k in range(layout.num_links)
    )

    # Per-hop: bell circuit + phase-A circuit. We reuse the 2-node helpers by
    # constructing a per-hop TwoNodeLayout (only alice_data/comm + bob_data/comm
    # are read by those helpers; the dataclass's other fields are unused).
    bell_circuits: list[stim.Circuit] = []
    phase_a_circuits: list[stim.Circuit] = []
    for k in range(layout.num_links):
        hop_layout = TwoNodeLayout(
            alice_data=layout.alice_data[k],
            alice_comm=layout.alice_comm[k],
            bob_comm=layout.bob_comm[k],
            bob_data=layout.bob_data[k],
        )
        bell_circuits.append(_build_bell_circuit(
            hop_layout, p1, p2, channel, pauli_1q, pauli_2q, bell_fid,
            measurement_fidelity=float(params["measurement_fidelity"]),
        ))
        phase_a_circuits.append(_build_phase_a(hop_layout, p2, channel, pauli_2q))

    ft_check_circuits_alice = tuple(
        tuple(
            _build_ft_check_circuit(layout.alice_data[k], layout.alice_ancilla[k], support, p2, channel, pauli_2q)
            for support in ft_supports
        )
        for k in range(layout.num_links)
    )
    ft_check_circuits_bob = tuple(
        tuple(
            _build_ft_check_circuit(layout.bob_data[k], layout.bob_ancilla[k], support, p2, channel, pauli_2q)
            for support in ft_supports
        )
        for k in range(layout.num_links)
    )

    # Per-middle-node BSM (one per intermediate router). Middle node i is router
    # index 1 .. N-2; its left block is bob_data[i-1] and right block is
    # alice_data[i].
    bsm_circuits: list[stim.Circuit] = []
    bsm_left_blocks: list[tuple[int, ...]] = []
    bsm_right_blocks: list[tuple[int, ...]] = []
    for i in range(1, num_nodes - 1):
        left_block = layout.bob_data[i - 1]
        right_block = layout.alice_data[i]
        bsm_circuits.append(
            _build_swap_bsm_circuit(left_block, right_block, p1, p2, channel, pauli_1q, pauli_2q)
        )
        bsm_left_blocks.append(left_block)
        bsm_right_blocks.append(right_block)

    initiator_block = layout.initiator_block
    responder_block = layout.responder_block
    initiator_x_stab_paulis = tuple(_make_block_pauli(num_qubits, initiator_block, s) for s in STEANE_X_STABILIZERS)
    initiator_z_stab_paulis = tuple(_make_block_pauli(num_qubits, initiator_block, s) for s in STEANE_Z_STABILIZERS)
    responder_x_stab_paulis = tuple(_make_block_pauli(num_qubits, responder_block, s) for s in STEANE_X_STABILIZERS)
    responder_z_stab_paulis = tuple(_make_block_pauli(num_qubits, responder_block, s) for s in STEANE_Z_STABILIZERS)
    logical_xx = _make_joint_pauli(num_qubits, initiator_block, responder_block, STEANE_LX)
    logical_yy = _make_joint_pauli(num_qubits, initiator_block, responder_block, STEANE_LY)
    logical_zz = _make_joint_pauli(num_qubits, initiator_block, responder_block, STEANE_LZ)

    return _CompiledPipelineN(
        layout=layout,
        num_qubits=num_qubits,
        p1=p1, p2=p2,
        channel=channel,
        pauli_1q=pauli_1q, pauli_2q=pauli_2q,
        measurement_flip_prob=measurement_flip_prob,
        state_prep_flip_prob=state_prep_flip_prob,
        bell_pair_fidelity=bell_fid,
        photon_loss=photon_loss,
        ft_prep_mode=ft_prep_mode,
        ft_max_retries=ft_max_retries,
        correction_mode=correction_mode,
        encoder_alice=encoder_alice,
        encoder_bob=encoder_bob,
        transversal_h_alice=transversal_h_alice,
        bell_circuits=tuple(bell_circuits),
        phase_a_circuits=tuple(phase_a_circuits),
        ft_check_circuits_alice=ft_check_circuits_alice,
        ft_check_circuits_bob=ft_check_circuits_bob,
        ft_supports=ft_supports,
        bsm_circuits=tuple(bsm_circuits),
        bsm_left_blocks=tuple(bsm_left_blocks),
        bsm_right_blocks=tuple(bsm_right_blocks),
        initiator_x_stab_paulis=initiator_x_stab_paulis,
        initiator_z_stab_paulis=initiator_z_stab_paulis,
        responder_x_stab_paulis=responder_x_stab_paulis,
        responder_z_stab_paulis=responder_z_stab_paulis,
        logical_xx=logical_xx,
        logical_yy=logical_yy,
        logical_zz=logical_zz,
    )


def _prep_block_n(
    sim: stim.TableauSimulator,
    block_qubits: Sequence[int],
    ancilla: int,
    encoder: stim.Circuit,
    ft_checks: Sequence[stim.Circuit],
    transversal_h: stim.Circuit | None,
    ft_max_retries: int,
    measurement_flip_prob: float,
    state_prep_flip_prob: float,
    rng: np.random.Generator | None,
) -> bool:
    """Encode one block with optional FT-prep retry; standalone (no 2-node pipeline).

    The FT-prep ancilla measurement carries the same readout-fidelity noise as
    every other measurement (mirrors quantum_manager_tableau.py:184). Resets
    carry an X_ERROR with probability state_prep_flip_prob (mirrors set_to_zero
    at quantum_manager.py:1948). `rng` must be provided whenever
    measurement_flip_prob > 0.
    """
    for _shot in range(ft_max_retries):
        _reset_with_prep_noise(sim, list(block_qubits) + [ancilla], state_prep_flip_prob)
        sim.do(encoder)
        if not ft_checks:
            if transversal_h is not None:
                sim.do(transversal_h)
            return True
        accepted = True
        for check in ft_checks:
            _reset_with_prep_noise(sim, [ancilla], state_prep_flip_prob)
            sim.do(check)
            bit = int(sim.measure(ancilla))
            if measurement_flip_prob > 0.0:
                if rng is None:
                    raise RuntimeError("_prep_block_n: rng must be provided when measurement_flip_prob > 0")
                if float(rng.random()) < measurement_flip_prob:
                    bit ^= 1
            if bit != 0:
                accepted = False
                break
        if accepted:
            if transversal_h is not None:
                sim.do(transversal_h)
            return True
    return False


def _run_shot_n_node(
    pipe: _CompiledPipelineN,
    sim_seed: int | None,
    rng: np.random.Generator,
) -> float:
    """Run one logical-pair shot on an N-node chain through the precompiled pipeline.

    The full sequence per shot:
      1. Encode every data block. router_0 has 1 right-facing block (|+>_L);
         each middle router has 2 blocks (left-facing |0>_L + right-facing |+>_L);
         router_{N-1} has 1 left-facing block (|0>_L). Each block goes through
         FT-prep with up to ft_max_retries retries.
      2. Per-hop teleported CNOT. For hop k between router_k and router_{k+1}:
           - Generate 7 Bell pairs across (alice_comm[k], bob_comm[k]). When
             measurement_fidelity < 1.0 we run the BarretKok simulation per pair
             to faithfully reproduce SeQUeNCe's BSM-induced corruption.
           - Phase A: CX(alice_data, alice_comm), measure alice_comm.
           - Phase B: CX(bob_comm, bob_data), feed-forward X based on alice bits,
             H(bob_comm), measure bob_comm.
           - Phase C: feed-forward Z on alice_data based on bob bits.
      3. Encoded-swap BSM at each middle router. The swap measures both data
         blocks of the middle node in a Bell-state basis (transversal CX(L,R) +
         transversal H(L) + measurement). The 14 bits decode to a per-node
         (b_x_corrected, b_z_corrected) frame contribution.
      4. XOR-accumulate the middle-node frame contributions into (final_bx,
         final_bz) and apply X^final_bx Z^final_bz transversally to router_0's
         data block. Transversal X = LX, transversal Z = LZ for Steane.
      5. Noiseless endpoint stabilizer recovery on both endpoint blocks.
      6. Compute end-to-end fidelity from logical XX, YY, ZZ correlators.

    Returns the per-shot end-to-end Bell-state fidelity in [0, 1].
    """
    sim = stim.TableauSimulator(seed=sim_seed)
    sim.set_num_qubits(pipe.num_qubits)
    layout = pipe.layout

    # Step 1. Encode every block (right-facing blocks -> |+>_L for use as TCNOT
    # control on the next hop; left-facing blocks -> |0>_L for use as target).
    # Mirrors RequestLogicalPairApp.encode_data_qubits (line 830-924).
    for k in range(layout.num_links):
        if not _prep_block_n(
            sim, layout.alice_data[k], layout.alice_ancilla[k],
            pipe.encoder_alice[k], pipe.ft_check_circuits_alice[k],
            pipe.transversal_h_alice[k], pipe.ft_max_retries,
            pipe.measurement_flip_prob, pipe.state_prep_flip_prob, rng,
        ):
            raise RuntimeError(f"FT prep failed (alice block of hop {k}).")
        if not _prep_block_n(
            sim, layout.bob_data[k], layout.bob_ancilla[k],
            pipe.encoder_bob[k], pipe.ft_check_circuits_bob[k],
            None, pipe.ft_max_retries,
            pipe.measurement_flip_prob, pipe.state_prep_flip_prob, rng,
        ):
            raise RuntimeError(f"FT prep failed (bob block of hop {k}).")

    # 2. Per-hop teleported CNOT (Phases A/B/C). Hops are on disjoint qubits so
    # the order is irrelevant for the final stim state; we run them sequentially
    # for clarity (matches the SeQUeNCe per-link execution).
    for k in range(layout.num_links):
        # Bell pair generation: at measurement_fidelity = 1 the precompiled
        # circuit is exact; otherwise per-pair BarretKok simulation.
        if pipe.measurement_flip_prob > 0.0:
            for i in range(CODE_N):
                label, _r0, _r1 = _simulate_barret_kok_pair(
                rng, 1.0 - pipe.measurement_flip_prob, photon_loss=pipe.photon_loss,
            )
                _prep_bell_pair_state(sim, layout.alice_comm[k][i], layout.bob_comm[k][i],
                                      label, pipe.p1, pipe.p2, pipe.channel,
                                      pipe.pauli_1q, pipe.pauli_2q,
                                      bell_pair_fidelity=pipe.bell_pair_fidelity)
        else:
            sim.do(pipe.bell_circuits[k])
        sim.do(pipe.phase_a_circuits[k])
        alice_bits = [int(sim.measure(layout.alice_comm[k][i])) for i in range(CODE_N)]
        if pipe.measurement_flip_prob > 0.0:
            flips = rng.random(CODE_N) < pipe.measurement_flip_prob
            alice_bits = [b ^ int(f) for b, f in zip(alice_bits, flips)]

        hop_layout = TwoNodeLayout(
            alice_data=layout.alice_data[k],
            alice_comm=layout.alice_comm[k],
            bob_comm=layout.bob_comm[k],
            bob_data=layout.bob_data[k],
        )
        phase_b = _build_phase_b_cx_then_h(
            hop_layout, pipe.p1, pipe.p2, pipe.channel,
            pipe.pauli_1q, pipe.pauli_2q, alice_bits,
        )
        sim.do(phase_b)
        bob_bits = [int(sim.measure(layout.bob_comm[k][i])) for i in range(CODE_N)]
        if pipe.measurement_flip_prob > 0.0:
            flips = rng.random(CODE_N) < pipe.measurement_flip_prob
            bob_bits = [b ^ int(f) for b, f in zip(bob_bits, flips)]

        phase_c = _build_phase_c(hop_layout, pipe.p1, pipe.channel, pipe.pauli_1q, bob_bits)
        sim.do(phase_c)

    # Step 3. Per-middle-node encoded BSM (the "swap" in QRE language). Each
    # middle router has two encoded blocks (left + right) entangled with its
    # neighbours by the per-hop TCNOTs above. The encoded BSM transversally
    # CXes left -> right, applies transversal H on left, and measures both
    # blocks in the Z basis (X-basis on left because of the H, Z-basis on right).
    # Mirrors QREProtocol.run_encoded_swapping (lines 216-310).
    #
    # The 14 measurement bits decode into a per-node Pauli-frame contribution
    # (b_x_corrected, b_z_corrected). Frame contributions XOR-accumulate across
    # all middle nodes; the final (bx, bz) is applied as a transversal correction
    # on the initiator block to pin the resulting end-to-end Bell pair to |Phi+>_L.
    final_bx = 0
    final_bz = 0
    for idx, bsm_circuit in enumerate(pipe.bsm_circuits):
        sim.do(bsm_circuit)
        left_block = pipe.bsm_left_blocks[idx]
        right_block = pipe.bsm_right_blocks[idx]
        left_x_bits = [int(sim.measure(q)) for q in left_block]
        right_z_bits = [int(sim.measure(q)) for q in right_block]
        if pipe.measurement_flip_prob > 0.0:
            # Apply readout flip noise to the 14 BSM bits (mirrors
            # quantum_manager_tableau.py:184).
            flips_l = rng.random(CODE_N) < pipe.measurement_flip_prob
            flips_r = rng.random(CODE_N) < pipe.measurement_flip_prob
            left_x_bits = [b ^ int(f) for b, f in zip(left_x_bits, flips_l)]
            right_z_bits = [b ^ int(f) for b, f in zip(right_z_bits, flips_r)]
        b_x_corrected, b_z_corrected = _decode_middle_bsm(left_x_bits, right_z_bits, pipe.correction_mode)
        # QREProtocol.py:280-281 maps the decoded parities into frame contributions
        # with X/Z swapped: bx <- b_z_corrected (X-frame from Z-basis right),
        # bz <- b_x_corrected (Z-frame from X-basis left).
        final_bx ^= b_z_corrected
        final_bz ^= b_x_corrected

    # Step 4. Initiator applies the accumulated frame correction transversally
    # on its data block. For Steane, transversal X = LX and transversal Z = LZ,
    # so this is exactly the logical-X^bx logical-Z^bz the protocol needs.
    # The correction is built as a per-shot stim.Circuit so the trailing 1q noise
    # (DEPOLARIZE1 / PAULI_CHANNEL_1) is sampled by stim's own RNG, identical to
    # how Phase B/C corrections are applied.
    if final_bx or final_bz:
        correction_circuit = stim.Circuit()
        if final_bx:
            for q in layout.initiator_block:
                _append_one_qubit_gate(correction_circuit, "X", q, pipe.p1, pipe.channel, pipe.pauli_1q)
        if final_bz:
            for q in layout.initiator_block:
                _append_one_qubit_gate(correction_circuit, "Z", q, pipe.p1, pipe.channel, pipe.pauli_1q)
        sim.do(correction_circuit)

    # Step 5. Noiseless endpoint stabilizer recovery on each endpoint block. We
    # peek the X- and Z-stabilizer eigenvalues without collapsing (the state is
    # already a stabilizer state), decode each syndrome to a single qubit, and
    # apply the corresponding X or Z to the data qubit identified by the
    # decoder. Mirrors RequestLogicalPairApp._calculate_pair_fidelity recover_block
    # (line 395-462). Skipped entirely when correction_mode == "none".
    if pipe.correction_mode in {"cec", "qec", "qec+cec"}:
        # Initiator block.
        x_synd = tuple(_peek_sign_bit(sim, obs) for obs in pipe.initiator_x_stab_paulis)
        z_synd = tuple(_peek_sign_bit(sim, obs) for obs in pipe.initiator_z_stab_paulis)
        x_err = STEANE_DECODE_TABLE[z_synd]
        z_err = STEANE_DECODE_TABLE[x_synd]
        if x_err is not None:
            sim.x(layout.initiator_block[int(x_err)])
        if z_err is not None:
            sim.z(layout.initiator_block[int(z_err)])
        # Responder block.
        x_synd = tuple(_peek_sign_bit(sim, obs) for obs in pipe.responder_x_stab_paulis)
        z_synd = tuple(_peek_sign_bit(sim, obs) for obs in pipe.responder_z_stab_paulis)
        x_err = STEANE_DECODE_TABLE[z_synd]
        z_err = STEANE_DECODE_TABLE[x_synd]
        if x_err is not None:
            sim.x(layout.responder_block[int(x_err)])
        if z_err is not None:
            sim.z(layout.responder_block[int(z_err)])

    cx = float(sim.peek_observable_expectation(pipe.logical_xx))
    cy = float(sim.peek_observable_expectation(pipe.logical_yy))
    cz = float(sim.peek_observable_expectation(pipe.logical_zz))
    return (1.0 + cx - cy + cz) / 4.0


def simulate_logical_pair_nnode(
    num_nodes: int,
    params: dict | None = None,
    seed: int | None = None,
) -> tuple[float, _CompiledPipelineN]:
    """Single-shot helper for the N-node chain. Returns (fidelity, pipeline)."""
    p = _resolve_params(params)
    pipe = _compile_pipeline_n_node(num_nodes, p)
    rng = np.random.default_rng(None if seed is None else seed + 0xC0FFEE)
    return _run_shot_n_node(pipe, seed, rng), pipe


def simulate_logical_pairs_nnode(
    num_nodes: int,
    num_pairs: int,
    params: dict | None = None,
    seed: int | None = None,
) -> dict[str, object]:
    """Run `num_pairs` independent shots on an N-node chain and aggregate metrics.

    This is the main entry point used by main_test_parallel_cheating.py and
    main_test_cheating.py. The pipeline is compiled once (precompiled stim
    circuits + Pauli observables for each link, BSM, and endpoint), then the
    per-shot driver _run_shot_n_node is called num_pairs times.

    Args:
      num_nodes: Length of the linear chain. N=2 reproduces line_2_2G.json,
                 N=6 reproduces line_6_2G.json, etc.
      num_pairs: Number of Bell pair generations (Monte Carlo sample size).
      params: Override dict for the run parameters; merges over DEFAULT_PARAMS.
      seed: Reproducibility seed; None -> nondeterministic.

    Returns:
      Metrics dict with the same schema as simulate_logical_pairs_2node so the
      parallel sweep can dispatch on num_nodes without changing how it consumes
      results. Per-shot fidelities are stored in "per_shot_fidelities" for
      downstream histogramming; NaN entries indicate FT-prep retry exhaustion.
    """
    p = _resolve_params(params)
    if num_pairs <= 0:
        out = _empty_metrics(p)
        out["num_nodes"] = int(num_nodes)
        return out

    pipe = _compile_pipeline_n_node(num_nodes, p)
    rng = np.random.default_rng(seed)
    fidelities = np.empty(num_pairs, dtype=np.float64)
    completed = 0
    for shot in range(num_pairs):
        shot_seed = None if seed is None else int(rng.integers(0, 2**63 - 1))
        try:
            value = _run_shot_n_node(pipe, shot_seed, rng)
        except RuntimeError:
            fidelities[shot] = np.nan
            continue
        fidelities[shot] = value
        completed += 1

    finite = fidelities[np.isfinite(fidelities)]
    avg = float(np.mean(finite)) if finite.size else float("nan")

    return {
        "num_workers": 1,
        "num_nodes": int(num_nodes),
        "num_logical_pairs_requested": int(num_pairs),
        "num_logical_pairs_completed": int(completed),
        "avg_end_to_end_logical_raw": float("nan"),
        "avg_end_to_end_logical_corrected": avg,
        "avg_latency_ps": float("nan"),
        "avg_latency_s": float("nan"),
        "avg_throughput_pairs_per_s": float("nan"),
        "two_qubit_gate_fidelity": float(p["two_qubit_gate_fidelity"]),
        "per_shot_fidelities": fidelities.tolist(),
    }


# ---------------------------------------------------------------------------
# Helpers reused by the parallel runner
# ---------------------------------------------------------------------------

def split_pair_counts(total_pairs: int, workers: int) -> list[int]:
    """Split a total pair budget across workers (mirrors main_test:746-758)."""
    if workers <= 0:
        raise ValueError("workers must be positive.")
    base = total_pairs // workers
    extra = total_pairs % workers
    return [base + (1 if i < extra else 0) for i in range(workers)]


def aggregate_worker_metrics(worker_metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    """Combine per-chunk metrics. Same schema as main_test's summarizer."""
    requested = sum(int(m["num_logical_pairs_requested"]) for m in worker_metrics)
    completed = sum(int(m["num_logical_pairs_completed"]) for m in worker_metrics)
    all_fidelities: list[float] = []
    for m in worker_metrics:
        for value in m.get("per_shot_fidelities", []):
            if value is not None and math.isfinite(float(value)):
                all_fidelities.append(float(value))
    avg = float(np.mean(all_fidelities)) if all_fidelities else float("nan")
    return {
        "num_workers": len(worker_metrics),
        "num_logical_pairs_requested": requested,
        "num_logical_pairs_completed": completed,
        "avg_end_to_end_logical_raw": float("nan"),
        "avg_end_to_end_logical_corrected": avg,
        "avg_latency_ps": float("nan"),
        "avg_latency_s": float("nan"),
        "avg_throughput_pairs_per_s": float("nan"),
    }
