"""QREProtocol module.

This module defines a minimal protocol scaffold for the
quantum repeater with encoding (QRE) workflow.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from sequence.message import Message
from sequence.protocol import Protocol
from sequence.utils import log
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from stim import Circuit


if TYPE_CHECKING:
    from RequestLogicalPairApp import RequestLogicalPairApp
    from sequence.topology.node import QuantumRouter2ndGeneration


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
    """Minimal protocol scaffold for one link-level QRE attempt.

    Args:
        owner: Protocol owner node.
        app: Owning RequestLogicalPairApp instance.
        remote_node_name: Neighbor endpoint name.
        memory_keys: Optional local memory keys used by this attempt.
        metadata: Optional context metadata for this attempt.
    """

    def __init__(self, owner: "QuantumRouter2ndGeneration", app: "RequestLogicalPairApp", remote_node_name: Optional[str] = None, memory_keys: Optional[list[int]] = None, metadata: Optional[dict[str, object]] = None):
        suffix = remote_node_name if remote_node_name is not None else "unbound"
        super().__init__(owner, name=f"{owner.name}.QRE.{suffix}")
        self.app = app
        self.remote_node_name = remote_node_name
        self.memory_keys = list(memory_keys) if memory_keys is not None else []
        self.metadata = dict(metadata) if metadata is not None else {}

        self.is_running = False
        self.is_success: Optional[bool] = None
        self.result: dict[str, object] | None = None

        self.n = self.app.n
        self.node_role = str(self.metadata.get("node_role", self.metadata.get("role", "edge")))
        self.left_data_keys = list(self.metadata.get("left_data_keys", []))
        self.right_data_keys = list(self.metadata.get("right_data_keys", []))

        self.current_phase = "IDLE"
        self.local_bx = 0
        self.local_bz = 0
        self.frame_contrib_bx = 0
        self.frame_contrib_bz = 0
        self.final_bx = 0
        self.final_bz = 0

        #TODO: fix the retrival of data qubits and try to eliminate the metadata dependency for data qubits if possible; for now we rely on the app to pass the correct data qubit keys in metadata, but ideally the protocol should be able to determine which qubits to operate on based on the reservation and memory allocation results without needing the app to tell it explicitly which qubits are for this link and which side they're on.

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
        self.current_phase = "SWAP_ENTRY"

        if self.node_role != "middle":
            self.current_phase = "WAIT_FRAME_BITS"
            return

        self.current_phase = "SWAP"
        time_now = self.owner.timeline.now()
        process = Process(self, "_run_encoded_swapping", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)

    def received_message(self, src: str, msg: object):
        """Handle incoming QRE messages.

        Args:
            src: Source node name.
            msg: Incoming message.

        Returns:
            None
        """
        if not isinstance(msg, QREMessage):
            return

        if msg.msg_type is QREMsgType.FRAME_UPDATE:
            self.frame_contrib_bx = int(msg.frame_contrib_bx)
            self.frame_contrib_bz = int(msg.frame_contrib_bz)
            self.current_phase = f"FRAME_UPDATE_RECEIVED from {msg.src}"
            self.update_pauli_frame(self.frame_contrib_bx, self.frame_contrib_bz, src)
        else:
            log.logger.warning(f"{self.name}: received unknown message type {msg.type} from {src}")

    def _run_encoded_swapping(self) -> None:
        """Run middle-node encoded swapping phase.

        Args:
            None.

        Returns:
            None
        """
        if self.node_role != "middle":
            raise RuntimeError(f"{self.name}: _run_encoded_swapping called on non-middle node")

        self.current_phase = "SWAP"

        run_keys = self.left_data_keys + self.right_data_keys
        if len(self.left_data_keys) != self.n or len(self.right_data_keys) != self.n:
            raise RuntimeError(f"{self.name}: expected {self.n} keys per side, got "
                f"left={len(self.left_data_keys)}, right={len(self.right_data_keys)}")

        circ = Circuit(2 * self.n)

        # Encoded swap: transversal CX(left_i -> right_i).
        for i in range(self.n):
            circ.cx(i, self.n + i)

        # Logical Bell measurement: left in X basis (H+M), right in Z basis (M).
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

        self.local_bx = int(decoded["b_x_corrected"])
        self.local_bz = int(decoded["b_z_corrected"])
        self.frame_contrib_bx = self.local_bz
        self.frame_contrib_bz = self.local_bx
        self.s_x = list(decoded["s_x"])
        self.s_z = list(decoded["s_z"])

        self.current_phase = "SWAP_COMPLETE"

        initiator_name = self.app._path_node_names[0]
        if initiator_name != self.owner.name:
            msg = QREMessage(
                QREMsgType.FRAME_UPDATE,
                protocol_name=self.name,
                sender_node=self.owner.name,
                frame_contrib_bx=self.frame_contrib_bx,
                frame_contrib_bz=self.frame_contrib_bz,
            )
            msg.receiver = f"{initiator_name}.QRE.{self.owner.name}"
            self.owner.send_message(initiator_name, msg)

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

    def update_pauli_frame(self, frame_contrib_bx: int, frame_contrib_bz: int, src: str, local_data_keys: list[int]) -> None:
        """Accumulate frame contributions and apply the net logical correction.

        Args:
            frame_contrib_bx: X-frame contribution received from one middle node.
            frame_contrib_bz: Z-frame contribution received from one middle node.
            src: Source node name for this frame update.
            local_data_keys: Local logical block qstate keys to correct.

        Returns:
            None
        """
        self.app.frame_updates_by_src[src] = (frame_contrib_bx, frame_contrib_bz)

        expected_updates = len(self.app._path_node_names) - 2
        if len(self.app.frame_updates_by_src) != expected_updates:
            log.logger.info(f"{self.name}: received frame update from {src}, waiting for {expected_updates - len(self.app.frame_updates_by_src)} more...")
            return
        else:
            bx = sum(bx for bx, _ in self.app.frame_updates_by_src.values()) % 2
            bz = sum(bz for _, bz in self.app.frame_updates_by_src.values()) % 2

            self.final_bx = bx
            self.final_bz = bz

            if bx or bz:
                corr = Circuit(len(local_data_keys))
                for i in range(len(local_data_keys)):
                    if bx:
                        corr.x(i)
                    if bz:
                        corr.z(i)
                self.owner.timeline.quantum_manager.run_circuit(corr, local_data_keys)

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

