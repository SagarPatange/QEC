"""QREProtocol module.

This module defines a minimal protocol scaffold for the
quantum repeater with encoding (QRE) workflow.
"""

from __future__ import annotations

from enum import Enum, auto
from sequence.message import Message
from sequence.protocol import Protocol
from sequence.utils import log
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.topology.node import QuantumRouter2ndGeneration
from RequestLogicalPairApp import RequestLogicalPairApp
from stim import Circuit

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
        super().__init__(msg_type, protocol_name)
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

    def __init__(self, owner: "QuantumRouter2ndGeneration", app: "RequestLogicalPairApp", remote_node_name: str):
        """Initialize one link-level QRE protocol instance.

        Args:
            owner: Protocol owner node.
            app: Owning RequestLogicalPairApp instance.
            remote_node_name: Neighbor endpoint name.

        Returns:
            None
        """
        super().__init__(owner, name=f"{owner.name}.QRE.{remote_node_name}")

        # Stable handles for app context and neighbor identity.
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
        self.left_data_keys = [] if left_peer is None else [m.qstate_key for m in self.app.data_qubits[left_peer]]
        self.right_data_keys = [] if right_peer is None else [m.qstate_key for m in self.app.data_qubits[right_peer]]

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

        # Frame bits used for local contribution and final aggregate correction.
        self.bx = 0
        self.bz = 0

        # Protocol output payload forwarded to app callback.
        self.result: dict[str, object] = {}

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

        # Initiator owns the frame accumulator, so reset it per attempt.
        if self.is_initiator:
            self.app.frame_updates_by_src.clear()

        # Middle-node path: execute encoded swapping locally.
        if self.is_middle:
            self.current_phase = "SWAP"
            time_now = self.owner.timeline.now()
            process = Process(self, "_run_encoded_swapping", [])
            priority = self.owner.timeline.schedule_counter
            event = Event(time_now, process, priority)
            self.owner.timeline.schedule(event)
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
                process = Process(self, "_complete_qre", [])
                priority = self.owner.timeline.schedule_counter
                event = Event(time_now, process, priority)
                self.owner.timeline.schedule(event)
                return

            self.current_phase = "WAIT_FRAME_BITS"
            return

        # End responder completes immediately in this message flow.
        self.result = {
            "final_bx": 0,
            "final_bz": 0,
            "frame_updates_received": 0,
        }
        self.current_phase = "ENDPOINT_COMPLETE"
        time_now = self.owner.timeline.now()
        process = Process(self, "_complete_qre", [])
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

        # Explicit type dispatch keeps this method extensible for future QRE messages.
        if msg.msg_type is QREMsgType.FRAME_UPDATE:
            self.bx = int(msg.frame_contrib_bx)
            self.bz = int(msg.frame_contrib_bz)
            self.current_phase = f"FRAME_UPDATE_RECEIVED from {src}"
            self.update_pauli_frame(src=src)
            return

        log.logger.warning(f"{self.name}: unknown message type {msg.msg_type} from {src}")

    def _run_encoded_swapping(self) -> None:
        """Run middle-node encoded swapping phase.

        Args:
            None.

        Returns:
            None
        """
        if not self.is_middle:
            raise RuntimeError(f"{self.name}: _run_encoded_swapping called on non-middle node")

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

        meas_samp = self.owner.get_generator().random()
        results = self.owner.timeline.quantum_manager.run_circuit(circ, run_keys, meas_samp)

        left_x_bits = [int(results[self.left_data_keys[i]]) for i in range(self.n)]
        right_z_bits = [int(results[self.right_data_keys[i]]) for i in range(self.n)]

        decoded = self._decode_middle_bsm(left_x_bits, right_z_bits)

        # 2) Decode Steane syndromes and map corrected parities to frame contributions.
        # Frame convention: frame_bx <- corrected Z parity, frame_bz <- corrected X parity.
        self.bx = int(decoded["b_z_corrected"])
        self.bz = int(decoded["b_x_corrected"])
        self.s_x = list(decoded["s_x"])
        self.s_z = list(decoded["s_z"])

        self.current_phase = "SWAP_COMPLETE"

        # 3) Forward this middle-node frame contribution to the initiator.
        initiator_name = self.initiator_name
        if initiator_name != self.owner.name:
            msg = QREMessage(QREMsgType.FRAME_UPDATE,
                protocol_name=self.name,
                sender_node=self.owner.name,
                frame_contrib_bx=self.bx,
                frame_contrib_bz=self.bz,
                )
            
            msg.receiver = f"{initiator_name}.QRE.{self.owner.name}"
            self.owner.send_message(initiator_name, msg)

        # Middle-node QRE completes after sending its frame contribution.
        self.result = {"frame_contrib_bx": int(self.bx), "frame_contrib_bz": int(self.bz)}

        time_now = self.owner.timeline.now()
        process = Process(self, "_complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)


    def _decode_middle_bsm(self, left_x_bits: list[int], right_z_bits: list[int]) -> dict[str, object]:
        """Decode Steane syndromes from middle-node Bell-measurement bitstrings.

        Args:
            left_x_bits: Seven X-basis bits from left logical block.
            right_z_bits: Seven Z-basis bits from right logical block.

        Returns:
            dict[str, object]: Syndrome bits, flip indices, corrected strings, and corrected parity bits.
        """
        decode_table = {
            (0, 0, 0): None,
            (0, 0, 1): 0,
            (0, 1, 0): 1,
            (0, 1, 1): 2,
            (1, 0, 0): 3,
            (1, 0, 1): 4,
            (1, 1, 0): 5,
            (1, 1, 1): 6,
        }

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

        x_flip_qubit = decode_table[tuple(s_x)]
        z_flip_qubit = decode_table[tuple(s_z)]

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

    def update_pauli_frame(self, src: str) -> None:
        """Accumulate frame contributions and apply final logical correction.

        Args:
            src: Source node name for this frame update.

        Returns:
            None
        """
        # Cache one source-tagged frame contribution.
        self.app.frame_updates_by_src[src] = (self.bx, self.bz)

        # Wait until all middle-node contributions for this attempt are available.
        if len(self.app.frame_updates_by_src) != self.expected_frame_updates:
            return

        final_bx = sum(bx for bx, _ in self.app.frame_updates_by_src.values()) % 2
        final_bz = sum(bz for _, bz in self.app.frame_updates_by_src.values()) % 2

        # Apply final logical frame on the local endpoint block when non-trivial.
        if (final_bx or final_bz) and self.local_data_keys:
            corr = Circuit(len(self.local_data_keys))
            for i in range(len(self.local_data_keys)):
                if final_bx:
                    corr.x(i)
                if final_bz:
                    corr.z(i)
            self.owner.timeline.quantum_manager.run_circuit(corr, self.local_data_keys)

        # Persist completion payload for app-level callback/reporting.
        self.result = {
            "final_bx": final_bx,
            "final_bz": final_bz,
            "frame_updates_received": len(self.app.frame_updates_by_src),
        }

        self.current_phase = "ALL_FRAME_UPDATES_RECEIVED"
        time_now = self.owner.timeline.now()
        process = Process(self, "_complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)


    def _complete_qre(self) -> None:
        """Mark the QRE attempt complete.

        Args:
            None.

        Returns:
            None
        """
        self.is_success = True
        self.current_phase = "COMPLETE"

        time_now = self.owner.timeline.now()
        process = Process(self.app, "_on_qre_complete", [self.remote_node_name, self.result])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)
