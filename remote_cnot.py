from enum import Enum, auto
from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.utils import log
from sequence.protocol import Protocol
import numpy as np
from sequence.topology.node import Node


class RemoteCNOTMsgType(Enum):
    """
    Enum representing types of messages exchanged during a teleported CNOT protocol.
    """
    TCNOT_START = auto()          # Message to initiate the teleported CNOT
    MEASUREMENT_RESULTS = auto()  # Message carrying measurement results
    APPLY_MEASUREMENT = auto()    # Message indicating when to perform measurement
    TCNOT_END = auto()            # Message signaling the end of the protocol


class RemoteCNOTMessage(Message):
    """
    Message container for RemoteCNOT protocol, with dynamic content depending on message type.
    """

    def __init__(self, msg_type: RemoteCNOTMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = RemoteCNOT  # reference to the protocol this message belongs to

        # Dynamically set the content depending on message type
        if msg_type is RemoteCNOTMsgType.TCNOT_START:
            self.start_tcnot = kwargs.get("start_tcnot")
        elif msg_type is RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            self.meas_result = kwargs.get("meas_result")
        elif msg_type is RemoteCNOTMsgType.APPLY_MEASUREMENT:
            self.start_measurement = kwargs.get("start_measurement")
        elif msg_type is RemoteCNOTMsgType.TCNOT_END:
            self.end_tcnot = kwargs.get("end_tcnot")
        else:
            raise ValueError(f"Invalid message type {msg_type}")

    def __repr__(self):
        """
        Returns a readable string representation of the message for debugging/logging.
        """
        if self.msg_type is RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            return f"type: {self.msg_type}, meas_result: {self.meas_result}"
        if self.msg_type is RemoteCNOTMsgType.TCNOT_START:
            return f"type: {self.msg_type}, meas_result: {self.start_tcnot}"
        if self.msg_type is RemoteCNOTMsgType.APPLY_MEASUREMENT:
            return f"type: {self.msg_type}, meas_status: {self.start_measurement}"
        if self.msg_type is RemoteCNOTMsgType.TCNOT_END:
            return f"type: {self.msg_type}, meas_status: {self.end_tcnot}"


class RemoteCNOT(Protocol):
    """
    Protocol for performing a teleported CNOT gate using entanglement between two nodes.
    
    Based on the relative node names, one node acts as the control and the other as the target.
    """

    def __init__(self, owner: "Node", name: str, other: str):
        super().__init__(owner, name)

        # Keys used to refer to qubits in quantum manager
        self.owner_storage_qubit_key = owner.storage_qubit.qstate_key
        self.owner_communication_qubit_key = owner.communication_qubit.qstate_key

        self.remote_node_name = other  # name of the remote node in the protocol
        self.remote_protocol_name: str = None

        self.meas_result = None
        self.activated = False

        # Determine if this node is the control (lexicographically smaller name)
        self.control = self.owner.name < self.remote_node_name
        self.target = not self.control

    def set_others(self, protocol, node, memories):
        """
        Sets up the reference to the remote protocol instance (remote node-side protocol name).
        """
        self.remote_protocol_name = protocol

    def start(self):
        """
        Entry point for starting the teleportation-based CNOT protocol.
        Sends a start message to the other node after readiness is verified.
        """
        log.logger.info(f"{self.name} protocol start for teleported CNOT.")
        self.activated = True

        if not self.is_ready():
            return "Other node is not ready"

        # Inform the other node to start
        message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_START, self.remote_protocol_name, start_tcnot=True)
        self.owner.send_message(self.remote_node_name, message)

    def received_message(self, src, msg):
        """
        Handles all incoming messages and branches to appropriate handlers.
        """
        if msg.msg_type == RemoteCNOTMsgType.TCNOT_START:
            log.logger.info(f"{self.owner.name} received confirmation to start teleported cnot operation {src}: {msg.start_tcnot}")
            if msg.start_tcnot:
                self.apply_local_tcnot()

        elif msg.msg_type == RemoteCNOTMsgType.APPLY_MEASUREMENT:
            log.logger.info(f"{self.owner.name} received confirmation to start measurement protocol {src}: {msg.start_measurement}")
            if msg.start_measurement:
                self.apply_measurement()

        elif msg.msg_type == RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            log.logger.info(f"{self.owner.name} received measurement result from {src}: {msg.meas_result}")
            self.apply_correction(msg)

        elif msg.msg_type == RemoteCNOTMsgType.TCNOT_END:
            log.logger.info(f"{self.owner.name} received confirmation to end protocol from {src}")
            if msg.end_tcnot:
                self.end_tcnot()

    def apply_local_tcnot(self):
        """
        Applies the local CNOT circuit between storage and communication qubit.
        Direction depends on whether this node is the control or target.
        """
        log.logger.info(f"{self.name} executing local CNOT circuit on {self.owner.name}")

        circuit_teleport = Circuit(size=2)
        circuit_teleport.cx(0, 1)

        # Choose qubit ordering based on control/target role
        if self.control:
            keys = [self.owner_storage_qubit_key, self.owner_communication_qubit_key]
        else:
            keys = [self.owner_communication_qubit_key, self.owner_storage_qubit_key]

        self.owner.timeline.quantum_manager.run_circuit(circuit_teleport, keys=keys)

        # Target sends message to control to begin measurement step
        if self.target:
            message = RemoteCNOTMessage(RemoteCNOTMsgType.APPLY_MEASUREMENT, self.remote_protocol_name, start_measurement=True)
            self.owner.send_message(self.remote_node_name, message)

    def apply_measurement(self):
        """
        Applies measurement to communication qubit depending on role.
        Sends result to remote node.
        """
        log.logger.info(f"{self.owner.name} measuring communication qubits")
        circuit_measurement = Circuit(size=1)

        if self.control:
            # Z-basis measurement
            circuit_measurement.measure(0)
        else:
            # X-basis measurement via Hadamard + Z
            circuit_measurement.h(0)
            circuit_measurement.measure(0)

        measurement = self.owner.timeline.quantum_manager.run_circuit(
            circuit_measurement,
            [self.owner_communication_qubit_key],
            meas_samp=np.random.rand()
        )

        self.meas_result = measurement[self.owner_communication_qubit_key]
        message = RemoteCNOTMessage(RemoteCNOTMsgType.MEASUREMENT_RESULTS, self.remote_protocol_name, meas_result=self.meas_result)
        self.owner.send_message(self.remote_node_name, message)

    def apply_correction(self, msg):
        """
        Applies correction to the storage qubit based on received measurement result.
        Also sends protocol end signal if applicable.
        """
        if self.control:
            # Apply Z correction to storage if needed
            if msg.meas_result == 1:
                correction_gate = Circuit(size=1)
                correction_gate.z(0)
                self.owner.timeline.quantum_manager.run_circuit(correction_gate, [self.owner_storage_qubit_key])

            # Inform target that protocol is complete
            message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_END, self.remote_protocol_name, end_tcnot=True)
            self.owner.send_message(self.remote_node_name, message)
            self.end_tcnot()

        else:
            # Apply X correction if measurement result is 1
            if msg.meas_result == 1:
                correction_gate = Circuit(size=1)
                correction_gate.x(0)
                self.owner.timeline.quantum_manager.run_circuit(correction_gate, [self.owner_storage_qubit_key])

            # Now perform measurement
            self.apply_measurement()

    def end_tcnot(self):
        """
        Finalizes and deactivates the protocol.
        """
        self.activated = False

    def is_ready(self):
        """
        Returns True if the remote protocol reference is set.
        """
        return self.remote_protocol_name is not None
