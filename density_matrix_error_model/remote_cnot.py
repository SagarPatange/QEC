# --- IMPORTS ---
# Import enum for message types, message passing class, circuit operations, logging utilities, base protocol class, NumPy, and Node structure
from enum import Enum, auto
from sequence.message import Message
from sequence.components.circuit import Circuit
from sequence.utils import log
from sequence.protocol import Protocol
import numpy as np
from sequence.topology.node import Node
# from noise import Noise
from sequence.utils.noise import Noise
import numpy as np



class RemoteCNOTMsgType(Enum):
    """
    Enumeration for different stages in the teleported CNOT protocol communication.
    """
    TCNOT_START = auto()          # Request to initiate teleported CNOT
    MEASUREMENT_RESULTS = auto()  # Communicate measurement results
    APPLY_MEASUREMENT = auto()    # Signal to apply measurement
    TCNOT_END = auto()             # Signal protocol completion


class RemoteCNOTMessage(Message):
    """
    Message object used in RemoteCNOT protocol.
    Encapsulates different kinds of payload depending on the stage of the protocol.
    """

    def __init__(self, msg_type: RemoteCNOTMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = RemoteCNOT  # Message belongs to RemoteCNOT protocol

        # Dynamically add message-specific fields
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
        """Formatted string output for better logging and debugging."""
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
    Protocol implementing a teleported CNOT gate between two nodes
    based on distributed entanglement.
    """

    def __init__(self, owner: "Node", name: str, other: str, depolarization_noise):
        super().__init__(owner, name)

        # Qubit references in the owner's quantum manager
        self.owner_storage_qubit_key = owner.storage_qubit.qstate_key
        self.owner_communication_qubit_key = owner.communication_qubit.qstate_key

        # Other node's name and protocol reference
        self.remote_node_name = other
        self.remote_protocol_name: str = None

        # Tracking variables
        self.meas_result = None
        self.activated = False

        # Determine roles: control node has a lexicographically smaller name
        self.control = self.owner.name < self.remote_node_name
        self.target = not self.control

        # Determines noise-free or noisy-prone circuit:
        self.depol_noise = depolarization_noise

    def set_others(self, protocol, node, memories):
        """Set the remote protocol name for communication."""
        self.remote_protocol_name = protocol


    def start(self):
        """
        Start the teleported CNOT protocol.
        Inform the remote node if ready.
        """
        log.logger.info(f"{self.name} protocol start for teleported CNOT.")
        self.activated = True



        # Send start command to other node
        message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_START, self.remote_protocol_name, start_tcnot=True)
        self.owner.send_message(self.remote_node_name, message)

    def received_message(self, src, msg):
        """
        Handle incoming protocol messages and dispatch to corresponding steps.
        """
        if msg.msg_type == RemoteCNOTMsgType.TCNOT_START:
            log.logger.info(f"{self.owner.name} received start signal {src}: {msg.start_tcnot}")
            if msg.start_tcnot:
                self.apply_local_tcnot()

        elif msg.msg_type == RemoteCNOTMsgType.APPLY_MEASUREMENT:
            log.logger.info(f"{self.owner.name} received measurement trigger {src}: {msg.start_measurement}")
            if msg.start_measurement:
                self.apply_measurement()

        elif msg.msg_type == RemoteCNOTMsgType.MEASUREMENT_RESULTS:
            log.logger.info(f"{self.owner.name} received measurement result {src}: {msg.meas_result}")
            self.apply_correction(msg)

        elif msg.msg_type == RemoteCNOTMsgType.TCNOT_END:
            log.logger.info(f"{self.owner.name} received end signal from {src}")
            if msg.end_tcnot:
                self.end_tcnot()

    def apply_local_tcnot(self):
        """
        Locally apply CNOT gate between storage and communication qubits.
        Control and target order depend on node's role.
        """
        log.logger.info(f"{self.name} executing local CNOT on {self.owner.name}")
        circuit_teleport = Circuit(size=2)
        circuit_teleport.cx(0, 1)


        if self.control:
            keys = [self.owner_storage_qubit_key, self.owner_communication_qubit_key]
        else:
            keys = [self.owner_communication_qubit_key, self.owner_storage_qubit_key]

        # --- Run CNOT gate ---
        self.owner.timeline.quantum_manager.run_circuit(circuit_teleport, keys=keys)

        


        if self.target:
            if self.depol_noise == True:
                state_info = self.owner.timeline.quantum_manager

                noisy_rho = Noise.apply_depolarizing_noise(rho = state_info.get(0).state, p = 0.5, qubits=[0,1], keys=state_info.get(0).keys)
                noisy_rho = Noise.apply_depolarizing_noise(rho = noisy_rho, p = 0.5, qubits=[2,3], keys=state_info.get(0).keys)

                state_info.set(keys=state_info.get(0).keys, state = noisy_rho)


            message = RemoteCNOTMessage(RemoteCNOTMsgType.APPLY_MEASUREMENT, self.remote_protocol_name, start_measurement=True)
            self.owner.send_message(self.remote_node_name, message)


    def apply_measurement(self):
        """
        Measure the communication qubit.
        Control measures in Z basis, target in X basis (H + Z).
        """
        log.logger.info(f"{self.owner.name} measuring communication qubit")
        circuit_measurement = Circuit(size=1)

        if self.control:
            # Add Measurement Noise
            circuit_measurement = Noise.apply_measurement_noise(circuit_measurement, meas_error_rate=0)
            circuit_measurement.measure(0)
        else:
            circuit_measurement.h(0)
            # Add Measurement Noise
            circuit_measurement = Noise.apply_measurement_noise(circuit_measurement, meas_error_rate=0)
            circuit_measurement.measure(0)


        measurement = self.owner.timeline.quantum_manager.run_circuit( circuit_measurement, [self.owner_communication_qubit_key], meas_samp=np.random.rand())
        


        # self.meas_result = noisy_meas
        message = RemoteCNOTMessage(RemoteCNOTMsgType.MEASUREMENT_RESULTS, self.remote_protocol_name, meas_result=measurement)
        self.owner.send_message(self.remote_node_name, message)

    def apply_correction(self, msg):
        """
        Apply correction gates to the storage qubit based on measurement outcome.
        """
        if self.control:
            if next(iter(msg.meas_result.values())) == 1:
                correction_gate = Circuit(size=1)
                correction_gate.z(0)
                self.owner.timeline.quantum_manager.run_circuit(correction_gate, [self.owner_storage_qubit_key])

                if self.depol_noise == True:
                    state_info = self.owner.timeline.quantum_manager
                    noisy_rho = Noise.apply_depolarizing_noise(state_info.get(0).state, p = 0.5, qubits = [self.owner_storage_qubit_key], keys = state_info.get(0).keys)
                    state_info.set(keys = state_info.get(0).keys, state=noisy_rho)

            message = RemoteCNOTMessage(RemoteCNOTMsgType.TCNOT_END, self.remote_protocol_name, end_tcnot=True)
                
            self.owner.send_message(self.remote_node_name, message)
            self.end_tcnot()

        else:
            if next(iter(msg.meas_result.values())) == 1:
                correction_gate = Circuit(size=1)
                correction_gate.x(0)

                # --- Inject memory noise before applying correction ---
                # Noise.apply_memory_noise(correction_gate, 0, memory_error_rate=0.5)

                self.owner.timeline.quantum_manager.run_circuit(correction_gate, [self.owner_storage_qubit_key])
                # self.display_density_matrix("After second correction")
                if self.depol_noise == True:
                    state_info = self.owner.timeline.quantum_manager.get(0)
                    # Noise.apply_single_qubit_depolarization(state_info.state, state_info.keys, self.owner_storage_qubit_key, 0.5, self.owner.timeline.quantum_manager)
            self.apply_measurement()


    def end_tcnot(self):
        """
        Deactivate protocol after CNOT teleportation completed.
        """
        self.activated = False

    def is_ready(self):
        """
        Return True if remote protocol reference is set (handshake ready).
        """
        return self.remote_protocol_name is not None
