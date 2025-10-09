"""QEC_App.py
Minimal QEC Application following teleport_app pattern.
Handles coordination and resource management while delegating quantum operations to QEC_protocol.

Architecture with QuantumRouter2ndGeneration (default sizes are perfect for QEC!):
- Communication memories (memo_arr): 7 memories for distributed QEC operations  
- Data memories (data_memo_arr): 7 memories for one [[7,1,3]] logical qubit
- Ancilla memories (ancilla_memo_arr): 6 memories for syndrome measurements

The app automatically detects if running on a 2nd gen router and uses the
appropriate memory arrays. Falls back gracefully to standard QuantumRouter.

NOTE: To enable message routing for QEC, add the following to 
      QuantumRouter2ndGeneration.receive_message() after the resource_manager check:
      
    elif msg.receiver == "qec_app":
        self.qec_app.received_message(src, msg)
"""

from sequence.app.request_app import RequestApp
from sequence.utils import log
from QEC_protocol import QECProtocol, QECMessage, QECMsgType
from sequence.topology.node import QuantumRouter2ndGeneration
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from typing import Dict, List


class QECApp(RequestApp): # TODO: rename this to RequestLogicalPairApp   
    """Quantum Error Correction Application
    
    Manages logical qubits and coordinates error correction protocols.
    Similar to TeleportApp but for continuous error correction rather than one-shot teleportation.
    Works with both QuantumRouter and QuantumRouter2ndGeneration nodes.
    
    Attributes:
        node: The quantum router node this app is attached to
        name (str): Name of the QEC application  
        qec_protocols (Dict): Active QEC protocol instances {logical_id: protocol}
        results (List): Results from error corrections [(timestamp, syndrome_data)]
    """
    
    def __init__(self, node):
        """Initialize QEC Application
        
        Args:
            node: Either QuantumRouter or QuantumRouter2ndGeneration
                  2nd gen provides separate data and ancilla memory arrays
        """
        super().__init__(node)
        self.name = f"{self.node.name}.QECApp"
        
        # Register with node for message routing (like teleport_app)
        node.qec_app = self  
        
        self.qec_protocols: Dict[int, QECProtocol] = {}
        self.results = []  # Store correction history like teleport_app stores teleported states
        self.next_logical_id = 0
        
        # Check if we have a 2nd generation router with additional memory arrays
        self.is_2nd_gen = isinstance(node, QuantumRouter2ndGeneration)
        
        if self.is_2nd_gen:
            log.logger.debug(f"{self.name}: Detected QuantumRouter2ndGeneration with:")
            log.logger.debug(f"  - Communication memories: {node.memo_arr_name}")
            log.logger.debug(f"  - Data memories: {node.data_memo_arr_name}")  
            log.logger.debug(f"  - Ancilla memories: {node.ancilla_memo_arr_name}")
        else:
            log.logger.debug(f"{self.name}: Using standard QuantumRouter")
        
        log.logger.debug(f"{self.name}: initialized")
    
    def start(self, physical_indices: List[int] = None, initial_state=None, 
              syndrome_interval: int = 100 * 1e9, use_data_memories: bool = True) -> int:
        """Start QEC protection for a logical qubit
        
        Similar to TeleportApp.start() but creates a persistent protocol instead of one-shot.
        
        Args:
            physical_indices: List of 7 physical qubit indices (if None, auto-allocate)
            initial_state: Optional initial quantum state (default |0⟩)
            syndrome_interval: Time between syndrome measurements in ps
            use_data_memories: If True and node is 2nd gen, use data memories instead of comm memories
            
        Returns:
            logical_id of the created logical qubit
        """
        # Auto-allocate qubits if not specified
        if physical_indices is None:
            if use_data_memories and self.is_2nd_gen:
                # Use data memories on 2nd gen router (default size is 7 - perfect!)
                physical_indices = list(range(7))
                log.logger.debug(f"{self.name}: Using data memories [0-6] for logical qubit")
            else:
                # Use communication memories on standard router
                physical_indices = list(range(7))
                log.logger.debug(f"{self.name}: Using communication memories [0-6] for logical qubit")
        
        if len(physical_indices) != 7:
            log.logger.error(f"{self.name}: Need exactly 7 qubits, got {len(physical_indices)}")
            return None
        
        log.logger.debug(f"{self.name}: start() -> physical_indices={physical_indices}, use_data_mem={use_data_memories}")
        
        # Create logical ID
        logical_id = self.next_logical_id
        self.next_logical_id += 1
        
        # Determine which memory array to use
        memory_array_name = None
        if use_data_memories and self.is_2nd_gen:
            memory_array_name = self.node.data_memo_arr_name
        else:
            memory_array_name = self.node.memo_arr_name
        
        # Create QEC protocol instance (like TeleportProtocol)
        qec_protocol = QECProtocol(
            self.node,
            logical_id=logical_id,
            physical_indices=physical_indices,
            syndrome_interval=syndrome_interval,
            memory_array_name=memory_array_name,
            ancilla_indices=[],  # No ancillas for basic start method
            ancilla_array_name=None
        )
        
        self.qec_protocols[logical_id] = qec_protocol
        
        # Encode the initial state
        qec_protocol.encode(initial_state)
        
        # Start syndrome measurement cycle
        self._schedule_syndrome_measurement(logical_id, 
                                           self.node.timeline.now() + syndrome_interval)
        
        log.logger.info(f"{self.name}: Started QEC for logical qubit {logical_id}")
        return logical_id
    
    def _schedule_syndrome_measurement(self, logical_id: int, time: int):
        """Schedule next syndrome measurement
        
        Args:
            logical_id: ID of logical qubit
            time: Time to perform measurement
        """
        def measure_syndromes():
            if logical_id not in self.qec_protocols:
                return  # Protocol was stopped
            
            protocol = self.qec_protocols[logical_id]
            
            # Measure syndromes
            syndrome_data = protocol.measure_syndromes()
            
            # Store results (like teleport_app stores teleported states)
            self.results.append((self.node.timeline.now(), syndrome_data))
            
            # Apply corrections if needed
            if protocol.needs_correction(syndrome_data):
                corrections = protocol.decode_error(syndrome_data)
                protocol.apply_corrections(corrections)
                log.logger.debug(f"{self.name}: Applied corrections for logical {logical_id}")
            
            # Schedule next measurement
            next_time = self.node.timeline.now() + protocol.syndrome_interval
            self._schedule_syndrome_measurement(logical_id, next_time)
        
        process = Process(self, "syndrome_measurement", [logical_id])
        process.func = measure_syndromes
        event = Event(time, process)
        self.node.timeline.schedule(event)
    
    def stop(self, logical_id: int):
        """Stop QEC and decode the logical qubit
        
        Similar to how TeleportApp cleans up after teleportation completes.
        
        Args:
            logical_id: ID of logical qubit to stop
            
        Returns:
            Decoded quantum state
        """
        if logical_id not in self.qec_protocols:
            log.logger.error(f"{self.name}: Logical qubit {logical_id} not found")
            return None
        
        protocol = self.qec_protocols[logical_id]
        
        # Decode the logical qubit
        final_state = protocol.decode()
        
        # Clean up (like TeleportApp removes completed protocols)
        del self.qec_protocols[logical_id]
        
        log.logger.info(f"{self.name}: Stopped QEC for logical {logical_id}, final state: {final_state}")
        return final_state
    
    def received_message(self, src: str, msg: QECMessage):
        """Handle incoming QEC messages
        
        Similar to TeleportApp.received_message() but for QEC coordination.
        
        Args:
            src: Source node name
            msg: The QEC message received
        """
        log.logger.debug(f"{self.name}: received_message from {src}: {msg}")
        
        if msg.msg_type == QECMsgType.SYNDROME_REQUEST:
            # Another node requesting syndrome measurement participation
            # (For distributed QEC - not implemented in minimal version)
            pass
            
        elif msg.msg_type == QECMsgType.CORRECTION_CMD:
            # Correction command from coordinator node
            if msg.logical_id in self.qec_protocols:
                protocol = self.qec_protocols[msg.logical_id]
                protocol.apply_correction_at_index(msg.qubit_index, msg.correction_type)
                log.logger.debug(f"{self.name}: Applied remote correction")
    
    def perform_logical_gate(self, logical_id: int, gate_type: str):
        """Perform a logical gate operation
        
        Args:
            logical_id: ID of logical qubit
            gate_type: Type of gate ('X', 'Z', 'H', etc.)
        """
        if logical_id not in self.qec_protocols:
            log.logger.error(f"{self.name}: Logical qubit {logical_id} not found")
            return
        
        protocol = self.qec_protocols[logical_id]
        protocol.apply_logical_gate(gate_type)
        
        log.logger.debug(f"{self.name}: Applied logical {gate_type} to qubit {logical_id}")
    
    def start_with_2nd_gen(self, initial_state=None,
                           syndrome_interval: int = 100 * 1e9) -> int:
        """Start QEC using QuantumRouter2ndGeneration's specialized memory arrays
        
        This method specifically leverages the 2nd generation router's architecture:
        - Uses all 7 data memories for the logical qubit (perfect match!)
        - Uses all 6 ancilla memories for syndrome measurements (perfect match!)
        - Keeps communication memories free for distributed operations
        
        Args:
            initial_state: Initial quantum state (default |0⟩)
            syndrome_interval: Time between syndrome measurements in ps
            
        Returns:
            logical_id of the created logical qubit
        """
        if not self.is_2nd_gen:
            log.logger.error(f"{self.name}: This method requires QuantumRouter2ndGeneration")
            return None
        
        # Use all data memories for the logical qubit (router has exactly 7 by default!)
        physical_indices = list(range(7))
        
        # Use all ancilla memories for syndrome measurements (router has exactly 6 by default!)
        ancilla_indices = list(range(6))
        
        log.logger.info(f"{self.name}: Starting QEC with 2nd gen router - "
                       f"using all 7 data memories and all 6 ancilla memories")
        
        # Create logical ID
        logical_id = self.next_logical_id
        self.next_logical_id += 1
        
        # Create protocol with 2nd gen specific configuration
        qec_protocol = QECProtocol(
            self.node,
            logical_id=logical_id,
            physical_indices=physical_indices,
            syndrome_interval=syndrome_interval,
            memory_array_name=self.node.data_memo_arr_name,
            ancilla_indices=ancilla_indices,
            ancilla_array_name=self.node.ancilla_memo_arr_name
        )
        
        self.qec_protocols[logical_id] = qec_protocol
        
        # Encode the initial state
        qec_protocol.encode(initial_state)
        
        # Start syndrome measurement cycle
        self._schedule_syndrome_measurement(logical_id, 
                                           self.node.timeline.now() + syndrome_interval)
        
        log.logger.info(f"{self.name}: Started QEC for logical qubit {logical_id} using 2nd gen features")
        return logical_id
    
    def get_syndrome_history(self, logical_id: int) -> List:
        """Get syndrome measurement history for a logical qubit
        
        Returns:
            List of (timestamp, syndrome_data) tuples
        """
        # Filter results for this logical qubit
        return [r for r in self.results if r[1].get('logical_id') == logical_id]
    
    def get_memory_arrays(self) -> Dict[str, any]:
        """Helper method to access different memory arrays
        
        Returns:
            Dict with available memory arrays
        """
        arrays = {}
        
        # Communication memories (always present)
        arrays['communication'] = self.node.get_component_by_name(self.node.memo_arr_name)
        
        # Data and ancilla memories (for 2nd generation routers)
        if self.is_2nd_gen:
            arrays['data'] = self.node.get_component_by_name(self.node.data_memo_arr_name)
            arrays['ancilla'] = self.node.get_component_by_name(self.node.ancilla_memo_arr_name)
        
        return arrays