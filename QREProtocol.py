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

        # Protocol context for message routing and app callbacks.
        self.app = app
        self.remote_node_name = remote_node_name
        self.n = self.app.n

        # Cached path-role metadata for phase control.
        self.is_middle = self.app._path_role == "middle"
        self.is_initiator = self.app._path_position == 0
        self.is_end_responder = self.app._path_position == len(self.app._path_node_names) - 1
        self.initiator_name = self.app._path_node_names[0]
        self.expected_frame_updates = len(self.app._path_node_names) - 2

        # Store data key allocations for QRE phases.
        self.left_data_keys: list[int] = []
        self.right_data_keys: list[int] = []
        self.left_ancilla_keys: list[int] = []
        self.right_ancilla_keys: list[int] = []
        self.local_data_keys: list[int] = []    # Endpoint block keys that may receive the final frame correction.

        # Per-run protocol execution state.
        self.is_running = False
        self.current_phase = "IDLE"
        self.is_success = False

        # Idle-noise parameters derived from the app.
        self.idle_pauli_weights: dict[str, float] = dict(self.app.idle_pauli_weights)
        self.idle_data_coherence_time_sec = float(self.app.idle_data_coherence_time_sec)
        self.correction_mode = str(self.app.correction_mode)

        # Frame contributions for local and final Pauli-frame corrections.
        self.bx = 0
        self.bz = 0

        # Result data returned to the app when QRE completes.
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

        # Starting phase
        self.is_running = True
        self.current_phase = "START"
        log.logger.info(f"{self.name}: start role middle={self.is_middle} initiator={self.is_initiator} phase={self.current_phase}")

        # Record the local data/ancilla assignments at start to avoid later allocation changes affecting this run.
        left_peer = self.app._left_peer_name
        right_peer = self.app._right_peer_name
        left_memories = [] if left_peer is None else self.app.data_qubits.get(left_peer, [])     # Own memories associated with left neighbor
        right_memories = [] if right_peer is None else self.app.data_qubits.get(right_peer, [])  # Own memories associated with right neighbor
        left_ancilla_memories = [] if left_peer is None else self.app.ancilla_qubits.get(left_peer, [])
        right_ancilla_memories = [] if right_peer is None else self.app.ancilla_qubits.get(right_peer, [])
        self.left_data_keys = [m.qstate_key for m in left_memories]
        self.right_data_keys = [m.qstate_key for m in right_memories]
        self.left_ancilla_keys = [m.qstate_key for m in left_ancilla_memories]
        self.right_ancilla_keys = [m.qstate_key for m in right_ancilla_memories]

        # Endpoints keep the local block that may receive the final frame correction.
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
                self.result = {"final_bx": 0, "final_bz": 0, "frame_updates_received": 0}
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
        self.result = { "final_bx": 0, "final_bz": 0, "frame_updates_received": 0,}
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

        # Keep left/right block order fixed for the swap decoder.
        run_keys = self.left_data_keys + self.right_data_keys
        if len(self.left_data_keys) != self.n or len(self.right_data_keys) != self.n:
            raise RuntimeError(f"{self.name}: expected {self.n} keys per side, got " f"left={len(self.left_data_keys)}, right={len(self.right_data_keys)}")

        qm = self.owner.timeline.quantum_manager
        coherence_time_by_key = {key: self.idle_data_coherence_time_sec for key in run_keys}
        
        # Apply accumulated idle noise before the local swap circuit.
        qm.apply_idling_decoherence(keys=run_keys, now_ps=int(self.owner.timeline.now()), coherence_time_sec_by_key=coherence_time_by_key, pauli_weights=self.idle_pauli_weights)

        pre_swap_duration_ps = 0
        if self.correction_mode in {"qec", "qec+cec"}: # TODO: Make these constants?
            # Run one QEC round on each local block before swapping.
            left_qec = self.run_qec_cycle(self.left_data_keys, self.left_ancilla_keys)
            right_qec = self.run_qec_cycle(self.right_data_keys, self.right_ancilla_keys)
            pre_swap_duration_ps = int(left_qec.get("duration_ps", 0)) + int(right_qec.get("duration_ps", 0))
            log.logger.info(f"{self.name}: pre_swap_qec left_x={left_qec['x_syndrome']} left_z={left_qec['z_syndrome']} right_x={right_qec['x_syndrome']} right_z={right_qec['z_syndrome']}")

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
        # Execute the encoded Bell-basis measurement across both blocks.
        results = qm.run_circuit(circ, run_keys, meas_samp)
        # Delay swap outputs by the full local pre-swap processing time.
        finish_t = int(self.owner.timeline.now()) + pre_swap_duration_ps + qm.get_circuit_duration(circ)
        log.logger.info(f"{self.name}: swap_timing now={int(self.owner.timeline.now())} pre_swap_duration_ps={pre_swap_duration_ps} swap_duration_ps={qm.get_circuit_duration(circ)} finish_t={finish_t}")
        # Mark both local data blocks busy until the swap circuit is considered complete.
        for key in run_keys:
            qm.last_idle_time_ps_by_key[key] = finish_t

        # The decoder expects X-basis bits from the left block and Z-basis bits from the right block.
        left_x_bits = [int(results[self.left_data_keys[i]]) for i in range(self.n)]
        right_z_bits = [int(results[self.right_data_keys[i]]) for i in range(self.n)]

        # Decode raw swap outcomes into corrected frame contributions.
        decoded = self.app.code.decode_middle_bsm(left_x_bits, right_z_bits, self.correction_mode)
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
            # Endpoints apply the final frame; middle nodes only report their contribution.
            msg = QREMessage(QREMsgType.FRAME_UPDATE,
                protocol_name=self.name,
                sender_node=self.owner.name,
                frame_contrib_bx=self.bx,
                frame_contrib_bz=self.bz)

            process = Process(self.owner, "send_message", [initiator_name, msg])
            event = Event(finish_t, process, self.owner.timeline.schedule_counter)
            self.owner.timeline.schedule(event)
            log.logger.info(f"{self.name}: emit_frame_update run_id={self.app.current_run['run_id']} initiator={initiator_name} bx={self.bx} bz={self.bz}")

        # Middle-node QRE completes after sending its frame contribution.
        self.result = {"frame_contrib_bx": int(self.bx), "frame_contrib_bz": int(self.bz)}

        process = Process(self, "complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(finish_t, process, priority)
        self.owner.timeline.schedule(event)

    def run_qec_cycle(self, data_keys: list[int], ancilla_keys: list[int]) -> dict[str, object]:
        """Run one local QEC round on a middle-node block before swap.

        Args:
            data_keys: Data-block quantum-manager keys.
            ancilla_keys: Ancilla quantum-manager keys associated with the block.

        Returns:
            dict[str, object]: Measured syndromes and decoded correction data.
        """
        if len(data_keys) != self.n:
            raise RuntimeError(f"{self.name}: expected {self.n} data keys for pre-swap QEC, got {len(data_keys)}")
        if len(ancilla_keys) < 3:
            raise RuntimeError(f"{self.name}: correction_mode={self.correction_mode} requires at least 3 ancillas per block")

        qm = self.owner.timeline.quantum_manager
        used_ancilla_keys = list(ancilla_keys[:3])
        qec_result = self.app.code.run_qec_round(qm=qm, data_keys=data_keys, ancilla_keys=used_ancilla_keys, meas_samp=self.owner.get_generator().random(), apply_physical_correction=True)

        now_ps = int(self.owner.timeline.now()) + int(qec_result.get("duration_ps", 0))
        for key in data_keys:
            qm.last_idle_time_ps_by_key[key] = now_ps
        for key in used_ancilla_keys:
            qm.last_idle_time_ps_by_key[key] = now_ps

        return qec_result

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
        finish_t = int(self.owner.timeline.now())
        # Apply final logical frame on the local endpoint block when non-trivial.
        if (final_bx or final_bz) and self.local_data_keys:
            # Apply time-based idle decoherence before endpoint frame-correction circuit.
            now_ps = int(self.owner.timeline.now())
            coherence_time_sec_by_key = {key: self.idle_data_coherence_time_sec for key in self.local_data_keys}
            self.owner.timeline.quantum_manager.apply_idling_decoherence(keys=self.local_data_keys, now_ps=now_ps, coherence_time_sec_by_key=coherence_time_sec_by_key, pauli_weights=self.idle_pauli_weights)
            correction_circuit = Circuit(len(self.local_data_keys))
            for i in range(len(self.local_data_keys)):
                if final_bx:
                    correction_circuit.x(i)
                if final_bz:
                    correction_circuit.z(i)
            log.logger.info(f"{self.name}: apply_final_frame final_bx={final_bx} final_bz={final_bz} local_data_keys={self.local_data_keys}")
            self.owner.timeline.quantum_manager.run_circuit(correction_circuit, self.local_data_keys)
            # Delay endpoint completion by the estimated local frame-correction time.
            finish_t = int(self.owner.timeline.now()) + self.owner.timeline.quantum_manager.get_circuit_duration(correction_circuit)
            log.logger.info(f"{self.name}: final_frame_timing now={int(self.owner.timeline.now())} duration_ps={self.owner.timeline.quantum_manager.get_circuit_duration(correction_circuit)} finish_t={finish_t}")
            # Mark the endpoint block busy until the frame correction is considered complete.
            for key in self.local_data_keys:
                self.owner.timeline.quantum_manager.last_idle_time_ps_by_key[key] = finish_t

        # Persist completion payload for app-level callback/reporting.
        self.result = {"final_bx": final_bx, "final_bz": final_bz, "frame_updates_received": len(self.app.current_run["frame_updates_by_src"])}
        self.current_phase = "ALL_FRAME_UPDATES_RECEIVED"
        process = Process(self, "complete_qre", [])
        priority = self.owner.timeline.schedule_counter
        event = Event(finish_t, process, priority)
        self.owner.timeline.schedule(event)

    def complete_qre(self) -> None:
        """Mark the QRE attempt complete.

        Args:
            None.

        Returns:
            None
        """
        # Mark the QRE attempt as successful.
        self.is_success = True
        self.current_phase = "COMPLETE"
        log.logger.info(f"{self.name}: COMPLETE result={self.result}")

        # Schedule the completion event.
        time_now = self.owner.timeline.now()
        process = Process(self.app, "logical_pair_complete", [self.remote_node_name, self.result])
        priority = self.owner.timeline.schedule_counter
        event = Event(time_now, process, priority)
        self.owner.timeline.schedule(event)
