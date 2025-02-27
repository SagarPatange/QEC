from enum import Enum, auto
from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.utils import log
from sequence.protocol import Protocol
import numpy as np



class RemoteCNOTMsgType(Enum):
    TCNOT_START = auto()
    MEASUREMENT_RESULTS = auto()


class RemoteCNOTMessage(Message):
    def __init__(self, msg_type: RemoteCNOTMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = RemoteCNOT

        if msg_type is RemoteCNOTMsgType.TCNOT_START:
            self.start_tcnot = kwargs.get("start_tcnot")  
        elif msg_type is RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            self.meas_result = kwargs.get("meas_result")
        else:
            raise ValueError(f"Invalid message type {msg_type}")

    def __repr__(self):
        if self.msg_type is RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            return f"type: {self.msg_type}, meas_result: {self.meas_result}"
        if self.msg_type is RemoteCNOTMsgType.TCNOT_START:
            return f"type: {self.msg_type}, meas_result: {self.start_tcnot}"


class RemoteCNOT(Protocol):
    """Protocol for performing a teleported CNOT operation using pre-shared entanglement."""

    def __init__(self, owner, name, other):
        super().__init__(owner, name)
        self.storage_qubit_key = owner.storage_qubit.qstate_key
        self.communication_qubit_key = owner.communication_qubit.qstate_key
        self.another = other
        self.meas_result = None
        self.remote_protocol_name = None

    def set_others(self, protocol, node, memories):
        """Link with remote protocol instance."""
        self.remote_protocol_name = protocol


    def start(self):
        """Initiate the teleported CNOT operation."""
        log.logger.info(f"{self.name} protocol start for teleported CNOT.")
        # Ensure both nodes have the necessary entanglement before starting

        if not self.is_ready(): # TODO: make sure that both nodes have the required bell state 
            return
        
        # Tell other node to start teleported cnot operation 
        message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_START, self.remote_protocol_name, start_tcnot=True)
        self.owner.send_message(self.another, message)

    
    def apply_local_tcnot(self): ## TODO: rename

        log.logger.info(f"{self.name} executing local CNOT circuit on {self.owner.name}")

        if self.owner.name == 'node1':
            circuit_teleport = Circuit(size=2)  
            circuit_teleport.cx(0, 1)
            circuit_teleport.measure(1)  # Measurement in H + X measurement = measurement in Z basis
        else:
            circuit_teleport = Circuit(size=2)  
            circuit_teleport.cx(1, 0)
            circuit_teleport.h(1)
            circuit_teleport.measure(1) # Measurement in X basis  
        
        meas_results = self.owner.timeline.quantum_manager.run_circuit(circuit_teleport, keys = [self.storage_qubit_key, self.communication_qubit_key], meas_samp=np.random.rand())
        self.meas_result = meas_results[self.communication_qubit_key]

        # Send measurement result to remote node
        message = RemoteCNOTMessage(RemoteCNOTMsgType.MEASUREMENT_RESULTS, self.remote_protocol_name, meas_result=self.meas_result)
        self.owner.send_message(self.another, message)

    def received_message(self, src, msg):
        """Handle received messages """
        if msg.msg_type == RemoteCNOTMsgType.TCNOT_START:
            log.logger.info(f"{self.owner.name} received confirmation to start teleported cnot operation {src}: {msg.start_tcnot}")
            if msg.start_tcnot:
                self.apply_local_tcnot()

        if msg.msg_type == RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            log.logger.info(f"{self.owner.name} received measurement result from {src}: {msg.meas_result}")

            # Apply corrections based on received measurement results
            if msg.meas_result == 1:
                self.apply_corrections(src)

    def apply_corrections(self, src):
        """Apply necessary quantum corrections after receiving measurement results."""
        circuit_correction = Circuit(size=1)  
        if src == 'node1':
            circuit_correction.x(0)  # Apply X correction to node2 storage qubit 
        elif src == 'node2' :
            circuit_correction.z(0)  # Apply Z correction to node1 storage qubit

        log.logger.info(f"{self.owner.name} applying correction to target qubit")
        self.owner.timeline.quantum_manager.run_circuit(circuit_correction, [self.storage_qubit_key])

    def is_ready(self):
        """Check if the protocol is ready to start."""
        return self.remote_protocol_name is not None

