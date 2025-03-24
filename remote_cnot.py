

from enum import Enum, auto
from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.utils import log
from sequence.protocol import Protocol
import numpy as np
from sequence.topology.node import Node



class RemoteCNOTMsgType(Enum):
    TCNOT_START = auto()
    MEASUREMENT_RESULTS = auto()
    APPLY_MEASUREMENT = auto()
    TCNOT_END = auto()


class RemoteCNOTMessage(Message):
    def __init__(self, msg_type: RemoteCNOTMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = RemoteCNOT

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
        if self.msg_type is RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            return f"type: {self.msg_type}, meas_result: {self.meas_result}"
        if self.msg_type is RemoteCNOTMsgType.TCNOT_START:
            return f"type: {self.msg_type}, meas_result: {self.start_tcnot}"
        if self.msg_type is RemoteCNOTMsgType.APPLY_MEASUREMENT:
            return f"type: {self.msg_type}, meas_status: {self.start_measurement}"
        if self.msg_type is RemoteCNOTMsgType.TCNOT_END:
            return f"type: {self.msg_type}, meas_status: {self.end_tcnot}"
        
class RemoteCNOT(Protocol):
    """Protocol for performing a teleported CNOT operation using pre-shared entanglement."""

    def __init__(self, owner: "Node", name: str, other: str):
        super().__init__(owner, name)
        self.owner_storage_qubit_key = owner.storage_qubit.qstate_key
        self.owner_communication_qubit_key = owner.communication_qubit.qstate_key
        self.remote_node_name = other
        self.remote_protocol_name: str = None

        self.meas_result = None
        self.remote_protocol_name = None

        self.activated = False

        # misc
        self.control = self.owner.name < self.remote_node_name
        self.target = not self.control


    def set_others(self, protocol, node, memories):
        """Link with remote protocol instance."""
        self.remote_protocol_name = protocol


    def start(self):
        """Initiate the teleported CNOT operation."""
        log.logger.info(f"{self.name} protocol start for teleported CNOT.")
        # Ensure both nodes have the necessary entanglement before starting

        self.activated = True

        if not self.is_ready(): # TODO: make sure that both nodes have the required bell state 
            return "Other node is not ready"
        
        # Tell other node to start teleported cnot operation 
        message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_START, self.remote_protocol_name, start_tcnot=True)
        self.owner.send_message(self.remote_node_name, message)

    def received_message(self, src, msg):
        """Handle received messages """
        if msg.msg_type == RemoteCNOTMsgType.TCNOT_START:
            log.logger.info(f"{self.owner.name} received confirmation to start teleported cnot operation {src}: {msg.start_tcnot}")
            if msg.start_tcnot:
                self.apply_local_tcnot()

        if msg.msg_type == RemoteCNOTMsgType.APPLY_MEASUREMENT:
            log.logger.info(f"{self.owner.name} received confirmation to start measurement protocol {src}: {msg.start_measurement}")

            # Apply corrections based on received measurement results
            if msg.start_measurement:
                self.apply_measurement()


        if msg.msg_type == RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            log.logger.info(f"{self.owner.name} received measurement result from {src}: {msg.meas_result}")
            self.apply_correction(msg)

        if msg.msg_type == RemoteCNOTMsgType.TCNOT_END:
            log.logger.info(f"{self.owner.name} received confirmation to end protocol from {src}")
            if msg.end_tcnot:
                self.end_tcnot()

    def apply_local_tcnot(self): ## TODO: rename

        log.logger.info(f"{self.name} executing local CNOT circuit on {self.owner.name}")
        
        if  self.control:
            circuit_teleport = Circuit(size=2)  
            circuit_teleport.cx(0, 1)
            self.owner.timeline.quantum_manager.run_circuit(circuit_teleport, keys = [self.owner_storage_qubit_key, self.owner_communication_qubit_key])
    

        else:
            circuit_teleport = Circuit(size=2)  
            circuit_teleport.cx(0, 1)
            self.owner.timeline.quantum_manager.run_circuit(circuit_teleport, keys = [self.owner_communication_qubit_key, self.owner_storage_qubit_key])

            # Send measurement result to remote node
            message = RemoteCNOTMessage(RemoteCNOTMsgType.APPLY_MEASUREMENT, self.remote_protocol_name, start_measurement=True)
            self.owner.send_message(self.remote_node_name, message)

    def apply_measurement(self): ## TODO:

        log.logger.info(f"{self.owner.name} measuring communication qubits")

        circuit_measurement = Circuit(size=1)  

        if self.control:
            circuit_measurement.measure(0)  # Measure C1 in the Z basis
            measurement = self.owner.timeline.quantum_manager.run_circuit(circuit_measurement, [self.owner_communication_qubit_key], meas_samp=np.random.rand())
            self.meas_result = measurement[self.owner_communication_qubit_key]

            message = RemoteCNOTMessage(RemoteCNOTMsgType.MEASUREMENT_RESULTS, self.remote_protocol_name, meas_result = self.meas_result)
            self.owner.send_message(self.remote_node_name, message)

        else: 
            circuit_measurement.h(0)
            circuit_measurement.measure(0)  # Measure C2 in the Z basis 
            measurement = self.owner.timeline.quantum_manager.run_circuit(circuit_measurement, [self.owner_communication_qubit_key], meas_samp=np.random.rand())
            self.meas_result = measurement[self.owner_communication_qubit_key]

            message = RemoteCNOTMessage(RemoteCNOTMsgType.MEASUREMENT_RESULTS, self.remote_protocol_name, meas_result = self.meas_result)
            self.owner.send_message(self.remote_node_name, message)


    def apply_correction(self, msg):
    
        if self.control:
            if msg.meas_result == 1:
                correction_gate = Circuit(size=1)  
                correction_gate.z(0)  # Apply Z correction to node2 storage qubit 
                self.owner.timeline.quantum_manager.run_circuit(correction_gate, [self.owner_storage_qubit_key])

            message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_END, self.remote_protocol_name, end_tcnot = True)
            self.owner.send_message(self.remote_node_name, message)

            self.end_tcnot()
            
        else:
            if msg.meas_result == 1:
                correction_gate = Circuit(size=1)  
                correction_gate.x(0)  # Apply X correction to node1 storage qubit
                self.owner.timeline.quantum_manager.run_circuit(correction_gate, [self.owner_storage_qubit_key])
            self.apply_measurement()

    def end_tcnot(self):
        self.activated = False
        
    def is_ready(self):
        """Check if the protocol is ready to start."""
        return self.remote_protocol_name is not None



