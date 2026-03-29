"""QREProtocol module.

This module defines a minimal protocol scaffold for the
quantum repeater with encoding (QRE) workflow.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING
from sequence.message import Message
from sequence.protocol import Protocol
from sequence.utils import log
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.node import QuantumRouter2ndGeneration
from sequence.components.circuit import Circuit

if TYPE_CHECKING:
    from RequestLogicalPairApp import RequestLogicalPairApp

DECODE_TABLE = {
    (0, 0, 0): None,
    (0, 0, 1): 0,
    (0, 1, 0): 1,
    (0, 1, 1): 2,
    (1, 0, 0): 3,
    (1, 0, 1): 4,
    (1, 1, 0): 5,
    (1, 1, 1): 6,
}

class QREMsgType(Enum):
    """Message types used by QREProtocol."""
    FRAME_UPDATE = auto()

class QREMessage(Message):
    """Classical message container used by QREProtocol."""
    def __init__(self, msg_type: QREMsgType, protocol_name: str, **kwargs):
        """Create a QRE classical message.

        Args:
            msg_type: QRE message type.
            protocol_name: Sender protocol name.
            **kwargs: Message payload fields by type.

        Returns:
            None
        """
        super().__init__(msg_type, "request_logical_pair_app")
        self.protocol_name = protocol_name

        if msg_type is QREMsgType.FRAME_UPDATE:
            self.sender_node = str(kwargs["sender_node"])
            self.frame_contrib_bx = int(kwargs["frame_contrib_bx"])
            self.frame_contrib_bz = int(kwargs["frame_contrib_bz"])
        else:
            raise ValueError(f"Unknown QRE message type: {msg_type}")
        
class QREProtocol(Protocol):
    """Link-level QRE protocol for one neighbor relationship.

    This protocol runs local encoded-swapping logic on middle nodes and
    handles frame-update aggregation/correction on endpoint nodes.

    Args:
        owner: Protocol owner node.
        app: Owning RequestLogicalPairApp instance.
        remote_node_name: Neighbor endpoint name.
    """

    def __init__(self, owner: QuantumRouter2ndGeneration, app: RequestLogicalPairApp, remote_node_name: str):
        """Initialize one link-level QRE protocol instance.

        Args:
            owner: Protocol owner node.
            app: Owning RequestLogicalPairApp instance.
            remote_node_name: Neighbor endpoint name.

        Returns:
            None
        """
        super().__init__(owner, name=f"{owner.name}.QRE.{remote_node_name}")

        # Stable handles for app context and remote peer.
        self.app = app
        self.remote_node_name = remote_node_name
        self.n = self.app.n

        # Cached role/path values to avoid repeated app lookups in methods.
        self.is_middle = (self.app._path_role == "middle")
        self.is_initiator = (self.app._path_position == 0)
        self.is_end_responder = (self.app._path_position == len(self.app._path_node_names) - 1)
        self.initiator_name = self.app._path_node_names[0]
        self.expected_frame_updates = len(self.app._path_node_names) - 2

        # Local data-key views for left/right adjacent logical blocks.
        left_peer = self.app._left_peer_name
        right_peer = self.app._right_peer_name
        left_memories = [] if left_peer is None else self.app.data_qubits.get(left_peer, [])
        right_memories = [] if right_peer is None else self.app.data_qubits.get(right_peer, [])
        self.left_data_keys = [m.qstate_key for m in left_memories]
        self.right_data_keys = [m.qstate_key for m in right_memories]

        # Endpoint correction target block:
        # initiator -> right block, end responder -> left block, middle -> none.
        if self.is_initiator:
            self.local_data_keys = list(self.right_data_keys)
        elif self.is_end_responder:
            self.local_data_keys = list(self.left_data_keys)
        else:
            self.local_data_keys = []

        # Runtime protocol state.
        self.is_running = False
        self.current_phase = "IDLE"
        self.is_success = False
        # Cache idle-noise settings from app once; methods use local fields.
        self.idle_pauli_weights: dict[str, float] = dict(self.app.idle_pauli_weights)
        self.idle_data_coherence_time_sec = float(self.app.idle_data_coherence_time_sec)

        # Frame bits used for local contribution and final aggregate correction.
        self.bx = 0
        self.bz = 0

        # Protocol output payload forwarded to app callback.
        self.result: dict[str, object] = {}
        log.logger.debug(f"{self.name}: init middle={self.is_middle} initiator={self.is_initiator} end_responder={self.is_end_responder} expected_frame_updates={self.expected_frame_updates}")

    def start(self) -> None:
        """Start QRE after TCNOT completion for this link.

        Args:
            None.

        Returns:
            None
        """
        if self.is_running:
            raise RuntimeError(f"{self.name}: start called more than once")

        self.is_running = True
        self.current_phase = "START"
        log.logger.info(f"{self.name}: start role middle={self.is_middle} initiator={self.is_initiator} phase={self.current_phase}")

        # Refresh local data-key views at start to avoid stale init-time snapshots.
        left_peer = self.app._left_peer_name
        right_peer = self.app._right_peer_name
        left_memories = [] if left_peer is None else self.app.data_qubits.get(left_peer, [])
        right_memories = [] if right_peer is None else self.app.data_qubits.get(right_peer, [])
        self.left_data_keys = [m.qstate_key for m in left_memories]
        self.right_data_keys = [m.qstate_key for m in right_memories]

        if self.is_initiator:
            self.local_data_keys = list(self.right_data_keys)
        elif self.is_end_responder:
            self.local_data_keys = list(self.left_data_keys)
        else:
            self.local_data_keys = []
        log.logger.info(f"{self.name}: qre_start run_id={self.app.current_run['run_id']} initiator={self.is_initiator} middle={self.is_middle} expected_updates={self.expected_frame_updates} local_data_keys={self.local_data_keys}")

        # Initiator owns the frame accumulator, so reset it per attempt.
        if self.is_initiator:
            self.app.current_run["frame_updates_by_src"].clear()

        # Middle-node path: execute encoded swapping locally.
        if self.is_middle:
            self.current_phase = "SWAP"
            time_now = self.owner.timeline.now()
            process = Process(self, "run_encoded_swapping", [])
            priority = self.owner.timeline.schedule_counter
            event = Event(time_now, process, priority)
            self.owner.timeline.schedule(event)
            log.logger.info(f"{self.name}: scheduling run_encoded_swapping")
            return

        # Initiator endpoint waits for frame updates from middle nodes.
        if self.is_initiator:
            if self.expected_frame_updates == 0:
                self.result = {
                    "final_bx": 0,
                    "final_bz": 0,
                    "frame_updates_received": 0,
                }
                self.current_phase = "NO_MIDDLE_NODES"
                time_now = self.owner.timeline.now()
                process = Process(self, "complete_qre", [])
                priority = self.owner.timeline.schedule_counter
                event = Event(time_now, process, priority)
                self.owner.timeline.schedule(event)
                return

            self.current_phase = "WAIT_FRAME_BITS"
            log.logger.info(f"{self.name}: waiting for frame bits expected={self.expected_frame_updates}")
            return

        # End responder completes immediately in this message flow.
        self.result = {
            "final_bx": 0,
            "final_bz": 0,
            "frame_updates_received": 0,
        }
        self.current_phase = "ENDPOINT_COMPLETE"
        log.logger.info(f"{self.name}: endpoint completes immediately")
        time_now = self.owner.timeline.now()
        process = Process(self, "complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)

    def received_message(self, src: str, msg: object) -> None:
        """Handle incoming QRE messages.

        Args:
            src: Source node name.
            msg: Incoming message.

        Returns:
            None
        """
        if not isinstance(msg, QREMessage):
            return
        log.logger.debug(f"{self.name}: recv src={src} type={getattr(msg, 'msg_type', type(msg).__name__)} phase={self.current_phase}")

        # Explicit type dispatch keeps this method extensible for future QRE messages.
        if msg.msg_type is QREMsgType.FRAME_UPDATE:
            self.current_phase = f"FRAME_UPDATE_RECEIVED from {src}"
            time_now = self.owner.timeline.now()
            process = Process(self, "update_pauli_frame", [src, int(msg.frame_contrib_bx), int(msg.frame_contrib_bz)])
            priority = self.owner.timeline.schedule_counter
            event = Event(time_now, process, priority)
            self.owner.timeline.schedule(event)
            return

        log.logger.warning(f"{self.name}: unknown message type {msg.msg_type} from {src}")

    def run_encoded_swapping(self) -> None:
        """Run middle-node encoded swapping phase.

        Args:
            None.

        Returns:
            None
        """
        if not self.is_middle:
            raise RuntimeError(f"{self.name}: run_encoded_swapping called on non-middle node")

        self.current_phase = "SWAP"

        run_keys = self.left_data_keys + self.right_data_keys
        if len(self.left_data_keys) != self.n or len(self.right_data_keys) != self.n:
            raise RuntimeError(f"{self.name}: expected {self.n} keys per side, got "
                f"left={len(self.left_data_keys)}, right={len(self.right_data_keys)}")

        circ = Circuit(2 * self.n)

        # 1) Build encoded Bell-measurement circuit: transversal CX plus block measurements.
        for i in range(self.n):
            circ.cx(i, self.n + i)

        # Left block measured in X basis (H+M); right block measured in Z basis (M).
        for i in range(self.n):
            circ.h(i)
            circ.measure(i)
        for i in range(self.n):
            circ.measure(self.n + i)

        # Apply time-based idle decoherence before encoded swapping consumes data keys.
        now_ps = int(self.owner.timeline.now())
        coherence_time_sec_by_key = {key: self.idle_data_coherence_time_sec for key in run_keys}
        self.owner.timeline.quantum_manager.apply_idling_decoherence(
            keys=run_keys,
            now_ps=now_ps,
            coherence_time_sec_by_key=coherence_time_sec_by_key,
            pauli_weights=self.idle_pauli_weights)

        meas_samp = self.owner.get_generator().random()
        results = self.owner.timeline.quantum_manager.run_circuit(circ, run_keys, meas_samp)

        left_x_bits = [int(results[self.left_data_keys[i]]) for i in range(self.n)]
        right_z_bits = [int(results[self.right_data_keys[i]]) for i in range(self.n)]

        decoded = self.decode_middle_bsm(left_x_bits, right_z_bits)
        log.logger.info(f"{self.name}: swap_decode raw_left_x={left_x_bits} raw_right_z={right_z_bits} s_x={decoded['s_x']} s_z={decoded['s_z']} x_flip={decoded['x_flip_qubit']} z_flip={decoded['z_flip_qubit']} x_corrected={decoded['x_corrected']} z_corrected={decoded['z_corrected']} b_x_corrected={decoded['b_x_corrected']} b_z_corrected={decoded['b_z_corrected']}")

        # 2) Decode Steane syndromes and map corrected parities to frame contributions.
        # Frame convention: frame_bx <- corrected Z parity, frame_bz <- corrected X parity.
        self.bx = int(decoded["b_z_corrected"])
        self.bz = int(decoded["b_x_corrected"])
        log.logger.info(f"{self.name}: frame_mapping b_x_corrected={decoded['b_x_corrected']} b_z_corrected={decoded['b_z_corrected']} mapped_bx={self.bx} mapped_bz={self.bz}")
        self.s_x = list(decoded["s_x"])
        self.s_z = list(decoded["s_z"])
        log.logger.info(f"{self.name}: swap decoded frame_contrib_bx={self.bx} frame_contrib_bz={self.bz}")

        self.current_phase = "SWAP_COMPLETE"

        # 3) Forward this middle-node frame contribution to the initiator.
        initiator_name = self.initiator_name
        if initiator_name != self.owner.name:
            msg = QREMessage(QREMsgType.FRAME_UPDATE,
                protocol_name=self.name,
                sender_node=self.owner.name,
                frame_contrib_bx=self.bx,
                frame_contrib_bz=self.bz)
            
            self.owner.send_message(initiator_name, msg)
            log.logger.info(f"{self.name}: emit_frame_update run_id={self.app.current_run['run_id']} initiator={initiator_name} bx={self.bx} bz={self.bz}")

        # Middle-node QRE completes after sending its frame contribution.
        self.result = {"frame_contrib_bx": int(self.bx), "frame_contrib_bz": int(self.bz)}

        time_now = self.owner.timeline.now()
        process = Process(self, "complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)

    def decode_middle_bsm(self, left_x_bits: list[int], right_z_bits: list[int]) -> dict[str, object]:
        """Decode Steane syndromes from middle-node Bell-measurement bitstrings.

        Args:
            left_x_bits: Seven X-basis bits from left logical block.
            right_z_bits: Seven Z-basis bits from right logical block.

        Returns:
            dict[str, object]: Syndrome bits, flip indices, corrected strings, and corrected parity bits.
        """

        s_x = [
            left_x_bits[3] ^ left_x_bits[4] ^ left_x_bits[5] ^ left_x_bits[6],
            left_x_bits[1] ^ left_x_bits[2] ^ left_x_bits[5] ^ left_x_bits[6],
            left_x_bits[0] ^ left_x_bits[2] ^ left_x_bits[4] ^ left_x_bits[6],
        ]
        s_z = [
            right_z_bits[3] ^ right_z_bits[4] ^ right_z_bits[5] ^ right_z_bits[6],
            right_z_bits[1] ^ right_z_bits[2] ^ right_z_bits[5] ^ right_z_bits[6],
            right_z_bits[0] ^ right_z_bits[2] ^ right_z_bits[4] ^ right_z_bits[6],
        ]

        x_flip_qubit = DECODE_TABLE[tuple(s_x)]
        z_flip_qubit = DECODE_TABLE[tuple(s_z)]

        x_corrected = list(left_x_bits)
        z_corrected = list(right_z_bits)
        if x_flip_qubit is not None:
            x_corrected[x_flip_qubit] ^= 1
        if z_flip_qubit is not None:
            z_corrected[z_flip_qubit] ^= 1

        b_x_corrected = sum(x_corrected) & 1
        b_z_corrected = sum(z_corrected) & 1

        return {
            "s_x": s_x,
            "s_z": s_z,
            "x_flip_qubit": x_flip_qubit,
            "z_flip_qubit": z_flip_qubit,
            "x_corrected": x_corrected,
            "z_corrected": z_corrected,
            "b_x_corrected": b_x_corrected,
            "b_z_corrected": b_z_corrected,
        }

    def update_pauli_frame(self, src: str, bx: int, bz: int) -> None:
        """Accumulate frame contributions and apply final logical correction.

        Args:
            src: Source node name for this frame update.
            bx: Frame X contribution from the source node.
            bz: Frame Z contribution from the source node.

        Returns:
            None
        """
        # Cache one source-tagged frame contribution.
        self.bx = int(bx)
        self.bz = int(bz)
        self.app.current_run["frame_updates_by_src"][src] = (self.bx, self.bz)
        log.logger.info(f"{self.name}: frame_accumulate run_id={self.app.current_run['run_id']} src={src} stored={self.app.current_run['frame_updates_by_src']} count={len(self.app.current_run['frame_updates_by_src'])}/{self.expected_frame_updates}")

        # Wait until all middle-node contributions for this attempt are available.
        if len(self.app.current_run["frame_updates_by_src"]) != self.expected_frame_updates:
            return

        final_bx = sum(bx for bx, _ in self.app.current_run["frame_updates_by_src"].values()) % 2
        final_bz = sum(bz for _, bz in self.app.current_run["frame_updates_by_src"].values()) % 2
        log.logger.info(f"{self.name}: frame_final run_id={self.app.current_run['run_id']} final_bx={final_bx} final_bz={final_bz} local_data_keys={self.local_data_keys}")
        # Apply final logical frame on the local endpoint block when non-trivial.
        if (final_bx or final_bz) and self.local_data_keys:
            # Apply time-based idle decoherence before endpoint frame-correction circuit.
            now_ps = int(self.owner.timeline.now())
            coherence_time_sec_by_key = {key: self.idle_data_coherence_time_sec for key in self.local_data_keys}
            self.owner.timeline.quantum_manager.apply_idling_decoherence(
                keys=self.local_data_keys,
                now_ps=now_ps,
                coherence_time_sec_by_key=coherence_time_sec_by_key,
                pauli_weights=self.idle_pauli_weights)
            corr = Circuit(len(self.local_data_keys))
            for i in range(len(self.local_data_keys)):
                if final_bx:
                    corr.x(i)
                if final_bz:
                    corr.z(i)
            log.logger.info(f"{self.name}: apply_final_frame final_bx={final_bx} final_bz={final_bz} local_data_keys={self.local_data_keys}")
            self.owner.timeline.quantum_manager.run_circuit(corr, self.local_data_keys)

        # Persist completion payload for app-level callback/reporting.
        self.result = {
            "final_bx": final_bx,
            "final_bz": final_bz,
            "frame_updates_received": len(self.app.current_run["frame_updates_by_src"]),
        }

        self.current_phase = "ALL_FRAME_UPDATES_RECEIVED"
        time_now = self.owner.timeline.now()
        process = Process(self, "complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)

    def complete_qre(self) -> None:
        """Mark the QRE attempt complete.

        Args:
            None.

        Returns:
            None
        """
        self.is_success = True
        self.current_phase = "COMPLETE"
        log.logger.info(f"{self.name}: COMPLETE result={self.result}")

        time_now = self.owner.timeline.now()
        process = Process(self.app, "logical_pair_complete", [self.remote_node_name, self.result])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)
