"""
Request Logical Pair Application for generating Bell pairs with optional QEC encoding.

This application demonstrates:
1. Generation of 7 Bell pairs using Barrett-Kok entanglement
2. Fidelity calculation using quantum state tomography
3. Synchronization between nodes before fidelity calculation
4. Optional encoding to logical qubits using [[7,1,3]] code

COMPLETE VERSION: Includes proper fidelity calculation via tomography and full synchronization.
"""

import numpy as np
import stim
from typing import TYPE_CHECKING, Dict, List, Optional
from sequence.app.request_app import RequestApp
from sequence.utils import log
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.message import Message
from sequence.resource_management.memory_manager import MemoryInfo
from final_QEC713 import QEC713
from teleported_cnot import TeleportedCNOTProtocol, TeleportedCNOTMessage, TeleportedCNOTMsgType 


if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter
    from sequence.network_management.reservation import Reservation


class RequestLogicalPairApp(RequestApp):
    """Application for generating 7 Bell pairs and optionally encoding to logical qubits.
    
    Workflow:
    1. Request 7 Bell pairs via Barrett-Kok protocol
    2. Synchronize with remote node when all pairs are ready
    3. Calculate fidelities using quantum state tomography
    4. Optionally encode to logical qubits (Alice: |0>_L, Bob: |+>_L)
    """

    def __init__(self, node: "QuantumRouter"):
        super().__init__(node)

        # Register this instance on node so protocols can access it
        self.node.request_logical_pair_app = self  # ADD THIS LINE
        self.remote_node: QuantumRouter = None  
    

        # Bell pair tracking
        self.memo_size = 7  # Need exactly 7 Bell pairs for [[7,1,3]] code
        self.num_completed = 0
        self.results = []  # Store Bell pair results


        # Encoding configuration
        self.encoding_enabled = False
        self.logical_state = '0'  # '0' for |0>_L, '+' for |+>_L
        self.encoding_complete = False
        self.remote_encoding_enabled = False
        self.remote_logical_state = None
    
        # Add these for teleported CNOT:
        self.teleported_cnot_enabled = False
        self.teleported_cnot_protocol = None
        self.teleported_cnot_complete = False
        self.teleported_cnot_start_time = None
        self.teleported_cnot_end_time = None

        # Depolarization parameters
        self.depolarization_time_ms = 500  # Coherence time for depolarization noise in ms TODO: make a coherence time 
        self.depolarization_enabled = False  # Flag to enable/disable idling decoherence
        
        # Timing
        self.first_pair_time = None
        self.last_pair_time = None
        self.encoding_start_time = None
        self.encoding_end_time = None

        # For responder mode
        self.is_initiator = False
        self.is_responder = False

        # Output control
        self.verbose = True  # Print detailed results by default

        # Qubit storage for teleported CNOT and future QEC
        # Allocate memories from each memory array in 2nd generation quantum router
        # - 7 data qubits for encoding logical qubits
        # - 7 communication qubits for Bell pairs
        # - 6 ancilla qubits for QEC syndrome measurements
        
        # Get the memory arrays from the node's components
        data_array_name = f"{self.node.name}.DataMemoryArray"
        comm_array_name = f"{self.node.name}.MemoryArray"
        ancilla_array_name = f"{self.node.name}.AncillaMemoryArray"
        
        # Allocate 7 data memories for encoding logical qubits
        self.data_qubits = []
        if data_array_name in self.node.components:
            data_array = self.node.components[data_array_name]
            if len(data_array.memories) >= 7:
                self.data_qubits = data_array.memories[:7]
                log.logger.info(f"{self.node.name}: Allocated 7 data memories: "
                               f"{[m.name for m in self.data_qubits]}")
            else:
                log.logger.warning(f"{self.node.name}: Only {len(data_array.memories)} "
                                  f"data memories available (need 7)")
        else:
            log.logger.warning(f"{self.node.name}: DataMemoryArray not found in components")
        
        # Allocate 7 communication memories for Bell pairs
        self.communication_qubits = []
        if comm_array_name in self.node.components:
            comm_array = self.node.components[comm_array_name]
            if len(comm_array.memories) >= 7:
                self.communication_qubits = comm_array.memories[:7]
                log.logger.info(f"{self.node.name}: Allocated 7 communication memories: "
                               f"{[m.name for m in self.communication_qubits]}")
            else:
                log.logger.warning(f"{self.node.name}: Only {len(comm_array.memories)} "
                                  f"communication memories available (need 7)")
        else:
            log.logger.warning(f"{self.node.name}: MemoryArray not found in components")
        
        # Allocate 6 ancilla memories for QEC syndrome measurements
        self.ancilla_qubits = []
        if ancilla_array_name in self.node.components:
            ancilla_array = self.node.components[ancilla_array_name]
            if len(ancilla_array.memories) >= 6:
                self.ancilla_qubits = ancilla_array.memories[:6]
                log.logger.info(f"{self.node.name}: Allocated 6 ancilla memories: "
                               f"{[m.name for m in self.ancilla_qubits]}")
            else:
                log.logger.warning(f"{self.node.name}: Only {len(ancilla_array.memories)} "
                                  f"ancilla memories available (need 6)")
        else:
            log.logger.warning(f"{self.node.name}: AncillaMemoryArray not found in components")
    
        # Add logical fidelity tracking
        self.logical_fidelity = {
            'fidelity': None,
            'correlations': {
                'XX': None,
                'YY': None,
                'ZZ': None
            }
        }
   

    def start(self, responder: str, start_time: int, end_time: int, ## TODO: put the coherence time into the config file 
            memory_size: int = 7, target_fidelity: float = 0.8,
            logical_state: str = '+', encoding_enabled: bool = True,
            teleported_cnot_enabled: bool = False, depolarization_enabled: bool = False,
            coherence_time_ms: float = 500): ### TODO: replace memory size with th CSS object, and see how you can put the logical state in the protocol level. Removal of encoding_enabled and depolarization enabled, and tcnot_enabled
        """Start Bell pair generation with optional QEC encoding. 

        Args:
            remote_node_name: Name of the remote quantum router
            start_time: Start time in picoseconds
            end_time: End time in picoseconds
            memory_size: Number of Bell pairs to generate (must be 7 for encoding)
            target_fidelity: Minimum acceptable fidelity
            logical_state: '0' for |0>_L or '+' for |+>_L (if encoding enabled)
            encoding_enabled: Whether to encode to logical qubits after Bell pair generation
            teleported_cnot_enabled: Whether to perform teleported CNOT after encoding
            depolarization_enabled: Whether to apply idling decoherence noise
            coherence_time_ms: Coherence time for depolarization noise in milliseconds
        """
        assert memory_size == 7, "Must request exactly 7 Bell pairs for [[7,1,3]] encoding"
        if teleported_cnot_enabled:
            assert encoding_enabled, "Teleported CNOT requires encoding to be enabled"
            
        self.responder = responder

        # ============================================================================
        # SIMULATION WORKAROUND - NOT LOCC COMPLIANT
        # ============================================================================
        # The following code accesses the remote node object directly from the
        # timeline. This is ONLY required due to how Stim (the stabilizer simulator)
        # structures quantum states and requires all entangled qubits to be grouped
        # into a single shared tableau/circuit before operations can be performed.
        #
        # In a real physical implementation, Alice would NOT have access to Bob's
        # quantum state or qubit references. However, in Stim's simulator:
        # - Alice's encoded qubits exist in one tableau
        # - The Bell pairs exist in another tableau
        # - Bob's encoded qubits exist in yet another tableau
        # - To apply transversal operations across these, Stim requires grouping them
        #
        # This grouping operation (quantum_manager.group_qubits) is a simulator-specific
        # requirement that has no physical analog. In reality:
        # - Each party operates only on their local qubits
        # - Classical communication carries measurement results
        # - No direct access to remote quantum states is needed or possible
        #
        # This reference is used ONLY to call group_qubits() on the shared state,
        # which is a simulation bookkeeping operation, not a physical quantum operation.
        # All actual quantum operations remain strictly local and LOCC-compliant.
        # ============================================================================
        self.remote_node = self.node.timeline.get_entity_by_name(responder)
        if self.remote_node is None:
            log.logger.warning(f"{self.node.name}: Could not find remote node {responder}")

        self.encoding_enabled = encoding_enabled
        self.logical_state = logical_state
        self.teleported_cnot_enabled = teleported_cnot_enabled
        self.depolarization_enabled = depolarization_enabled
        self.depolarization_time_ms = coherence_time_ms
        self.is_initiator = True
        self.is_responder = False
        
        # Start Bell pair generation using parent class
        
        super().start(
            responder=responder,
            start_t=start_time,
            end_t=end_time,
            memo_size=memory_size,
            fidelity=target_fidelity
        )
        
        log.logger.info(f"{self.node.name}: Started Bell pair generation with {responder}, "
                       f"encoding={'enabled' if encoding_enabled else 'disabled'}")
  
 
    def _configure_as_responder(self, initiator_name: str, encoding_enabled: bool,
                                initiator_logical_state: str,
                                teleported_cnot_enabled: bool = True,
                                depolarization_enabled: bool = False,
                                coherence_time_ms: float = 500):
        """Configure this app as a responder to an initiator's request.

        Args:
            initiator_name: Name of the initiating node
            encoding_enabled: Whether encoding is enabled
            initiator_logical_state: Logical state the initiator will encode to
            teleported_cnot_enabled: Whether to perform teleported CNOT
            depolarization_enabled: Whether to apply depolarization noise
            coherence_time_ms: Coherence time for depolarization noise in milliseconds
        """
        self.responder = initiator_name

        # Get remote node reference from timeline
        self.remote_node = self.node.timeline.get_entity_by_name(initiator_name)
        if self.remote_node is None:
            log.logger.warning(f"{self.node.name}: Could not find remote node {initiator_name}")

        self.remote_encoding_enabled = encoding_enabled
        self.remote_logical_state = initiator_logical_state
        self.teleported_cnot_enabled = teleported_cnot_enabled
        self.depolarization_enabled = depolarization_enabled
        self.depolarization_time_ms = coherence_time_ms
        self.is_responder = True
        self.is_initiator = False
        
        # Configure our encoding state
        if encoding_enabled:
            self.encoding_enabled = True
            # Responder (Bob) uses |+>_L if initiator (Alice) uses |0>_L
            self.logical_state = '0' if initiator_logical_state == '+' else '+'
    
    
    def get_other_reservation(self, reservation: "Reservation") -> None:
        """Called when responder (Bob) receives a reservation from initiator (Alice).

        This method is called automatically by SeQUeNCe's reservation system.
        We configure Bob's role and encoding parameters here, then call the parent
        to populate memo_to_reservation so get_memory() can track Bell pairs.
        """
        # Configure as responder FIRST (before calling super)
        self.is_responder = True
        self.is_initiator = False
        self.responder = reservation.initiator  # Bob's "responder" field = who he talks to (Alice)

        # Get remote node reference from timeline
        self.remote_node = self.node.timeline.get_entity_by_name(reservation.initiator)
        if self.remote_node is None:
            log.logger.warning(f"{self.node.name}: Could not find remote node {reservation.initiator}")

        # Configure encoding parameters
        # Alice uses |+>_L, so Bob uses |0>_L for proper Bell pair creation
        self.encoding_enabled = True
        self.logical_state = '0'
        self.teleported_cnot_enabled = False
        self.depolarization_enabled = False

        # CRITICAL: Call parent to populate memo_to_reservation
        # Without this, get_memory() will exit early and Bob won't track Bell pairs
        super().get_other_reservation(reservation)

        log.logger.info(f"{self.node.name}: Configured as responder for {reservation.initiator}, "
                       f"encoding={self.encoding_enabled}, logical_state=|{self.logical_state}>_L")
        

    def _release_memories(self):
        """Release all communication memories to RAW state."""
        for result in self.results:
            info = result['memory_info']
            self.node.resource_manager.update(None, info.memory, "RAW")
        log.logger.info(f"{self.node.name}: Released all memories")
    
    
    def received_message(self, src: str, msg):
        """Handle incoming messages - route to appropriate protocol."""

        # DEBUG: Log ALL incoming messages
        log.logger.info(f"{'@'*70}")
        log.logger.info(f"DEBUG: {self.node.name} received_message() CALLED")
        log.logger.info(f"DEBUG: Source: {src}")
        log.logger.info(f"DEBUG: Message type: {type(msg).__name__}")
        log.logger.info(f"DEBUG: Is TeleportedCNOTMessage? {isinstance(msg, TeleportedCNOTMessage)}")
        log.logger.info(f"DEBUG: Protocol exists? {self.teleported_cnot_protocol is not None}")
        log.logger.info(f"{'@'*70}")

        # Route teleported CNOT messages to the protocol
        if isinstance(msg, TeleportedCNOTMessage):
            log.logger.info(f"{self.node.name}: Detected TeleportedCNOTMessage!")
            # Check if this is a completion message
            if msg.msg_type == TeleportedCNOTMsgType.PROTOCOL_COMPLETE:
                log.logger.info(f"{self.node.name}: Received PROTOCOL_COMPLETE message")
                self._on_teleported_cnot_complete()
            elif self.teleported_cnot_protocol:
                # Route measurement messages to protocol
                log.logger.info(f"{self.node.name}: Routing message to teleported CNOT protocol")
                self.teleported_cnot_protocol.received_message(src, msg)
            else:
                log.logger.warning(f"{self.node.name}: Received teleported CNOT message "
                                f"but protocol not initialized")
        else:
            # Handle other message types if needed
            log.logger.debug(f"{self.node.name}: Received non-teleported-CNOT message")
            pass
         
    ####################### Generate Physical Bell Pairs ##########################
     
    def get_memory(self, info: "MemoryInfo") -> None:
        """Process completed Bell pairs.
        
        Called when a memory becomes entangled. Collects Bell pairs and
        triggers synchronization when all 7 are ready.
        
        Args:
            info: Memory information containing entanglement details
        """
        if info.state != "ENTANGLED":
            return
        
        # Check if this memory is part of our reservation
        if info.index not in self.memo_to_reservation:
            return
                
        # Track timing
        generation_time = self.node.timeline.now()
        if self.first_pair_time is None:
            self.first_pair_time = generation_time
        self.last_pair_time = generation_time
        
        log.logger.info(f"{self.node.name}: Bell pair {self.num_completed} generated "
                       f"at t={generation_time*1e-12:.6f}s on memory {info.index}")
        
        # Store result (fidelity calculated later after sync)
        result = {
            'pair_id': self.num_completed,
            'memory_index': info.index,
            'memory_info': info,
            'generation_time': generation_time * 1e-12,
            'fidelity': None,  # Calculated after synchronization
            'remote_node': info.remote_node,
            'remote_memory': info.remote_memo
        }
        self.results.append(result)
        self.num_completed += 1  ## TODO: we need to continiously generate the bell pair so if we do this we only generate 1 logical bell pair
        
        if self.num_completed >= self.memo_size:
            # Schedule fidelity calculation first
            process = Process(self, '_calculate_physical_bell_pair_fidelities', [])
            event = Event(self.node.timeline.now() + 1000, process, priority=0)
            self.node.timeline.schedule(event)           
            
            if self.encoding_enabled:
                # Schedule encoding AFTER fidelity calculation (additional delay)
                process = Process(self, '_start_encoding', [])
                event = Event(self.node.timeline.now() + 1000, process, priority=1)
                self.node.timeline.schedule(event)
            else:
                # Schedule cleanup after fidelity calculation
                log.logger.info(f"{self.node.name}: No encoding requested - releasing memories")
                process = Process(self, 'print_results', [])
                event = Event(self.node.timeline.now() + 1000, process, priority=1)
                self.node.timeline.schedule(event)
         
    ####################### Generate Single State Encoding ##########################
        
    def _start_encoding(self):
        """Start QEC encoding process to create logical qubits.

        Encodes the pre-allocated data qubits to create either |0>_L or |+>_L.

        Steps:
        1. Initialize data qubits to |0> states
        2. Apply preparation noise (DEPOLARIZE1) based on memory fidelity
        3. Prepare initial state (apply H for |+> if needed)
        4. Apply [[7,1,3]] encoding circuit
        5. Schedule completion callback
        """
        log.logger.info(f"{self.node.name}: Starting [[7,1,3]] encoding to |{self.logical_state}>_L")
        self.encoding_start_time = self.node.timeline.now()

        try:
            # Get quantum manager
            qm = self.node.timeline.quantum_manager

            # Validate we have exactly 7 data qubits
            if len(self.data_qubits) != 7:
                raise ValueError(f"Expected 7 data qubits, got {len(self.data_qubits)}")

            # Get qubit keys from pre-allocated data memories
            log.logger.info(f"{self.node.name}: Using pre-allocated data memories")

            # Get qubit keys from pre-allocated data memories
            data_keys = [memory.qstate_key for memory in self.data_qubits]

            log.logger.info(f"{self.node.name}: Encoding using qubit keys {data_keys}")

            # Apply preparation noise (DEPOLARIZE1) to each data qubit based on its raw_fidelity
            for i, memory in enumerate(self.data_qubits):
                fidelity = getattr(memory, 'raw_fidelity', 1.0)
                if fidelity < 1.0:
                    # Apply single-qubit depolarizing noise
                    noise_param = 1.0 - fidelity
                    noise_circuit = stim.Circuit()
                    noise_circuit.append("DEPOLARIZE1", [data_keys[i]], noise_param)
                    qm.run_circuit(noise_circuit, [data_keys[i]], compute_dm=False)
                    log.logger.debug(f"{self.node.name}: Applied DEPOLARIZE1({noise_param:.4f}) to data qubit {i}")

            # Log the fidelity being used
            avg_fidelity = np.mean([getattr(m, 'raw_fidelity', 1.0) for m in self.data_qubits])
            log.logger.info(f"{self.node.name}: Data qubit avg fidelity = {avg_fidelity:.4f}")

            # Prepare initial state based on desired logical state
            if self.logical_state == '+':
                # Apply H to first qubit to get |+> = (|0> + |1>)/sqrt(2)
                log.logger.info(f"{self.node.name}: Applying H gate to prepare |+> initial state")
                h_circuit = QEC713.create_hadamard_circuit()
                qm.run_circuit(h_circuit, [data_keys[0]], compute_dm = False)

                # Schedule encoding completion callback
                delay = 1  # 1 picosecond
                process = Process(self, '_encoding_completed', [])
                event = Event(self.node.timeline.now() + delay, process)
                self.node.timeline.schedule(event)

                log.logger.info(f"{self.node.name}: Scheduled encoding completion callback")
            else:
                # Bob needs |0>_L: Qubits already in |0> state
                log.logger.info(f"{self.node.name}: Using |0> initial state (no preparation needed)")

            # Apply [[7,1,3]] encoding circuit
            log.logger.info(f"{self.node.name}: Applying [[7,1,3]] encoding circuit")
            encoding_circuit = QEC713.create_encoding_circuit()
            qm.run_circuit(encoding_circuit, data_keys, compute_dm = False)

            log.logger.info(f"{self.node.name}: Encoding circuit applied successfully")

            # Mark encoding as complete for Bob (who doesn't schedule _encoding_completed)
            if self.logical_state != '+':
                self.encoding_complete = True
                self.encoding_end_time = self.node.timeline.now()

        except Exception as e:
            log.logger.error(f"{self.node.name}: Encoding FAILED with error: {e}")
            import traceback
            log.logger.error(traceback.format_exc())
            
            # Mark encoding as failed and clean up
            self.encoding_complete = False
            self._release_memories()
            raise


    def _encoding_completed(self):
        """Called when QEC encoding is complete."""
        # DEBUG: Check if this is being called
        log.logger.info(f"{'*'*70}")
        log.logger.info(f"DEBUG: {self.node.name} _encoding_completed() CALLED")
        log.logger.info(f"DEBUG: Role: {'initiator' if self.is_initiator else 'responder'}")
        log.logger.info(f"DEBUG: teleported_cnot_enabled = {self.teleported_cnot_enabled}")
        log.logger.info(f"{'*'*70}")

        self.encoding_end_time = self.node.timeline.now()
        self.encoding_complete = True

        if self.is_initiator:
            log.logger.info(f"{self.node.name}: Calculating product state fidelity |+>_L (Alice) x |0>_L (Bob)")
            
            # Calculate product state fidelity
            product_fidelity_result = self._calculate_product_state_fidelity()
            
            # Store results
            self.product_state_fidelity = product_fidelity_result


        encoding_time = (self.encoding_end_time - self.encoding_start_time) * 1e-12
        log.logger.info(f"{self.node.name}: Encoding completed in {encoding_time:.6f}s")
        
        # # Expire the reservation to stop generating more pairs 
        # for reservation in self.memo_to_reservation.values():
        #     self.node.resource_manager.expire_rules_by_reservation(reservation)
        #     break  # Only expire once
        
        # Check if we should proceed to teleported CNOT
        if self.teleported_cnot_enabled:
            log.logger.info(f"{self.node.name}: Proceeding to teleported CNOT phase")

            # Verify qubits are ready (already allocated in __init__)
            log.logger.info(f"{self.node.name}: Using {len(self.data_qubits)} encoded data qubits "
                        f"and {len(self.communication_qubits)} communication qubits for teleported CNOT")
            
            # Verify we have the right number of qubits
            if len(self.data_qubits) != 7 or len(self.communication_qubits) != 7:
                raise RuntimeError(f"{self.node.name}: Incorrect number of qubits - "
                                f"data: {len(self.data_qubits)}, comm: {len(self.communication_qubits)}")

            self._initialize_teleported_cnot()
            # DON'T release memories or print yet - protocol needs them!
        else:
            # No teleported CNOT - we're done
            # self._release_memories()
            self.print_results()

    ####################### Teleported CNOT Protocol ##########################

    def _initialize_teleported_cnot(self):
        """Initialize and start the teleported CNOT protocol.

        Alice initializes both her own protocol and Bob's protocol.
        This is a simulation workaround - in reality, each node would
        initialize its own protocol via classical coordination.
        """

        # Only Alice initializes protocols (she handles both sides)
        if not self.is_initiator:
            log.logger.warning(f"{self.node.name}: _initialize_teleported_cnot called on responder - skipping")
            return

        # Create Alice's protocol if not already created
        if self.teleported_cnot_protocol is None:
            protocol_name = f"TeleportedCNOT_{self.node.name}"
            self.teleported_cnot_protocol = TeleportedCNOTProtocol(
                owner=self.node,
                name=protocol_name,
                role='alice',
                remote_node_name=self.responder,
                data_qubits=self.data_qubits,
                communication_qubits=self.communication_qubits,
                remote_node=self.remote_node
            )
            self.node.protocols.append(self.teleported_cnot_protocol)
            log.logger.info(f"{self.node.name}: Alice's teleported CNOT protocol initialized")

        # Create Bob's protocol if not already created
        if self.remote_node and hasattr(self.remote_node, 'request_logical_pair_app'):
            bob_app = self.remote_node.request_logical_pair_app
            if bob_app.teleported_cnot_protocol is None:
                bob_protocol_name = f"TeleportedCNOT_{self.remote_node.name}"
                bob_app.teleported_cnot_protocol = TeleportedCNOTProtocol(
                    owner=self.remote_node,
                    name=bob_protocol_name,
                    role='bob',
                    remote_node_name=self.node.name,
                    data_qubits=bob_app.data_qubits,
                    communication_qubits=bob_app.communication_qubits,
                    remote_node=self.node
                )
                self.remote_node.protocols.append(bob_app.teleported_cnot_protocol)
                log.logger.info(f"{self.remote_node.name}: Bob's teleported CNOT protocol initialized by Alice")

        # Start the protocol
        self.teleported_cnot_start_time = self.node.timeline.now()
        log.logger.info(f"{self.node.name}: Starting teleported CNOT as Alice")
        self.teleported_cnot_protocol.start()


    def _on_teleported_cnot_complete(self):
        """Called when teleported CNOT protocol finishes."""
        self.teleported_cnot_complete = True
        self.teleported_cnot_end_time = self.node.timeline.now()
        
        duration = (self.teleported_cnot_end_time - self.teleported_cnot_start_time) * 1e-12
        
        log.logger.info(f"\n{'='*70}")
        log.logger.info(f"{self.node.name}: LOGICAL BELL PAIR CREATED!")
        log.logger.info(f"{'='*70}")
        
        # MEASURE LOGICAL FIDELITY (only initiator measures to avoid duplication)
        if self.is_initiator:
            log.logger.info(f"{self.node.name}: Calculating logical Bell pair fidelity...")
            logical_fidelity = self._calculate_logical_bell_pair_fidelity()
            self.logical_fidelity['fidelity'] = logical_fidelity
            log.logger.info(f"{self.node.name}: Logical fidelity = {logical_fidelity:.6f}")
        
        log.logger.info(f"Teleported CNOT completed in {duration:.6f}s")
        log.logger.info(f"State: |Phi+>_AB = 1/sqrt(2)(|0>_L^A |0>_L^B + |1>_L^A |1>_L^B)")
        log.logger.info(f"{'='*70}\n")
        
        # NOW release memories and print final results
        self.print_results()

################ Fidelity & Error Calculation Functions  #########################  
        
    def _calculate_logical_bell_pair_fidelity(self, num_samples: int = 100, shots_per_sample: int = 1000) -> float:
        """
        Calculate logical Bell pair fidelity using multi-sample averaging.

        Uses multiple sampling runs with different seeds to properly capture
        stochastic noise from DEPOLARIZE2 channels.

        For |Φ+>_L = (|0>_L|0>_L + |1>_L|1>_L)/√2, measures:
        - <X_L ⊗ X_L> should be +1
        - <Y_L ⊗ Y_L> should be -1
        - <Z_L ⊗ Z_L> should be +1

        Fidelity F = (1 + <XX> - <YY> + <ZZ>) / 4

        For [[7,1,3]] Steane code:
        - X_L = X^⊗7 (X on all 7 physical qubits)
        - Z_L = Z^⊗7 (Z on all 7 physical qubits)

        Args:
            num_samples: Number of independent sampling runs to average
            shots_per_sample: Shots per Pauli basis per sample

        Returns:
            Logical Bell pair fidelity with |Φ⁺>_L
        """
        qm = self.node.timeline.quantum_manager

        # Get Alice's data qubits (this node)
        alice_keys = [mem.qstate_key for mem in self.data_qubits]

        # Get Bob's data qubits
        bob_node = self.node.timeline.get_entity_by_name(self.responder)
        if bob_node is None or not hasattr(bob_node, 'request_logical_pair_app'):
            raise ValueError(f"Cannot access Bob's node {self.responder} for fidelity measurement")
        bob_app = bob_node.request_logical_pair_app
        bob_keys = [mem.qstate_key for mem in bob_app.data_qubits]

        all_keys = alice_keys + bob_keys

        log.logger.info(f"{self.node.name}: Measuring logical Bell pair fidelity...")
        log.logger.info(f"{self.node.name}: Using {num_samples} samples x {shots_per_sample} shots")
        log.logger.info(f"{self.node.name}: Alice keys: {alice_keys}")
        log.logger.info(f"{self.node.name}: Bob keys: {bob_keys}")

        # Get the shared quantum state
        state = qm.states[alice_keys[0]]
        log.logger.info(f"{self.node.name}: State contains qubits: {sorted(state.keys)}")

        # Multi-sample averaging to capture stochastic noise
        correlation_samples = {'X': [], 'Y': [], 'Z': []}

        for sample_idx in range(num_samples):
            seed = (sample_idx * 12345 + hash(f"{self.node.name}_logical")) % (2**31)

            for basis in ['X', 'Y', 'Z']:
                # Build measurement circuit from state
                meas_circuit = state.circuit.copy()

                # Add basis rotation for all qubits
                for key in all_keys:
                    if basis == 'X':
                        meas_circuit.append("H", [key])
                    elif basis == 'Y':
                        meas_circuit.append("S_DAG", [key])
                        meas_circuit.append("H", [key])

                # Add measurements
                for key in all_keys:
                    meas_circuit.append("M", [key])

                # Sample with unique seed (ensure non-negative)
                basis_seed = abs(seed + hash(basis)) % (2**63)
                sampler = meas_circuit.compile_sampler(seed=basis_seed)
                measurements = sampler.sample(shots=shots_per_sample)

                # Find measurement columns for our qubits
                # Measurements are appended in order, so last 14 columns are ours
                num_meas = measurements.shape[1]
                num_our_qubits = len(all_keys)

                # Extract our measurements (last num_our_qubits columns)
                our_measurements = measurements[:, num_meas - num_our_qubits:]

                # Compute logical parities
                # Alice's logical parity = XOR of her 7 physical measurements
                alice_parity = np.sum(our_measurements[:, :7], axis=1) % 2
                # Bob's logical parity = XOR of his 7 physical measurements
                bob_parity = np.sum(our_measurements[:, 7:14], axis=1) % 2

                # Convert to eigenvalues and compute correlation
                alice_eigenvalues = 1 - 2 * alice_parity
                bob_eigenvalues = 1 - 2 * bob_parity
                correlation = float(np.mean(alice_eigenvalues * bob_eigenvalues))

                correlation_samples[basis].append(correlation)

        # Average correlations over all samples
        correlations = {
            'X': float(np.mean(correlation_samples['X'])),
            'Y': float(np.mean(correlation_samples['Y'])),
            'Z': float(np.mean(correlation_samples['Z']))
        }

        # Store correlations
        self.logical_fidelity['correlations']['XX'] = correlations['X']
        self.logical_fidelity['correlations']['YY'] = correlations['Y']
        self.logical_fidelity['correlations']['ZZ'] = correlations['Z']

        # Calculate fidelity: F = (1 + <XX> - <YY> + <ZZ>) / 4
        fidelity = (1 + correlations['X'] - correlations['Y'] + correlations['Z']) / 4

        log.logger.info(f"{self.node.name}: <X_L ⊗ X_L> = {correlations['X']:+.4f}")
        log.logger.info(f"{self.node.name}: <Y_L ⊗ Y_L> = {correlations['Y']:+.4f}")
        log.logger.info(f"{self.node.name}: <Z_L ⊗ Z_L> = {correlations['Z']:+.4f}")
        log.logger.info(f"{self.node.name}: Logical Bell pair fidelity = {fidelity:.6f}")
        log.logger.info(f"{self.node.name}: Formula: (1 + {correlations['X']:+.4f} - ({correlations['Y']:+.4f}) + {correlations['Z']:+.4f}) / 4 = {fidelity:.6f}")

        return float(fidelity)


    def _calculate_physical_bell_pair_fidelities(self, num_samples: int = 100, shots_per_sample: int = 1000):
        """Calculate fidelities for all Bell pairs using multi-sample averaging.

        Uses multiple sampling runs with different seeds to properly capture
        stochastic noise from DEPOLARIZE2 channels.

        Args:
            num_samples: Number of independent sampling runs to average
            shots_per_sample: Shots per Pauli basis per sample
        """
        qm = self.node.timeline.quantum_manager

        log.logger.info(f"{self.node.name}: Calculating fidelities for {self.memo_size} Bell pairs")
        log.logger.info(f"{self.node.name}: Using {num_samples} samples x {shots_per_sample} shots")

        # Calculate fidelity for each pair
        for result in self.results:
            local_key = result['memory_info'].memory.qstate_key
            remote_memory = self.node.timeline.get_entity_by_name(result['remote_memory'])
            remote_key = remote_memory.qstate_key
            keys = [local_key, remote_key]

            # Get the state and its circuit
            state = qm.states[local_key]

            # Sample multiple times with different seeds to capture stochastic noise
            fidelity_samples = []

            for sample_idx in range(num_samples):
                # Use different seed for each sample
                seed = (sample_idx * 12345 + hash(f"{self.node.name}_{result['pair_id']}")) % (2**31)

                # Compute density matrix via Pauli tomography
                # For Bell state fidelity: F = (1 + <XX> + <YY> + <ZZ>) / 4 for |Φ+>
                # But we need to be careful - |Φ+> has <XX>=+1, <YY>=-1, <ZZ>=+1
                # So F = Tr(ρ |Φ+><Φ+|) = (1 + <XX> - <YY> + <ZZ>) / 4

                correlations = {}
                for basis in ['X', 'Y', 'Z']:
                    # Build measurement circuit
                    meas_circuit = state.circuit.copy()

                    # Add basis rotation
                    for key in keys:
                        if basis == 'X':
                            meas_circuit.append("H", [key])
                        elif basis == 'Y':
                            meas_circuit.append("S_DAG", [key])
                            meas_circuit.append("H", [key])

                    # Add measurements
                    for key in keys:
                        meas_circuit.append("M", [key])

                    # Sample with this seed (ensure non-negative)
                    basis_seed = abs(seed + hash(basis)) % (2**63)
                    sampler = meas_circuit.compile_sampler(seed=basis_seed)
                    measurements = sampler.sample(shots=shots_per_sample)

                    # Get the last 2 measurement columns (for our 2 qubits)
                    num_measurements = measurements.shape[1]
                    m0 = measurements[:, num_measurements - 2]
                    m1 = measurements[:, num_measurements - 1]

                    # Compute correlation: <ZZ> after basis rotation
                    eigenvalues = (1 - 2*m0) * (1 - 2*m1)
                    correlations[basis] = float(np.mean(eigenvalues))

                # Fidelity with |Φ+>: F = (1 + <XX> - <YY> + <ZZ>) / 4
                fid = (1 + correlations['X'] - correlations['Y'] + correlations['Z']) / 4
                fidelity_samples.append(fid)

            # Average over all samples
            avg_fidelity = float(np.mean(fidelity_samples))
            std_fidelity = float(np.std(fidelity_samples))

            result['fidelity'] = float(np.clip(avg_fidelity, 0.0, 1.0))
            result['fidelity_std'] = std_fidelity

            log.logger.info(f"{self.node.name}: Pair {result['pair_id']} fidelity = {result['fidelity']:.4f} ± {std_fidelity:.4f}")

        # Log statistics
        fidelities = [r['fidelity'] for r in self.results]
        log.logger.info(f"{self.node.name}: Fidelity stats - mean={np.mean(fidelities):.4f}, std={np.std(fidelities):.4f}")


    def _calculate_product_state_fidelity(self, num_samples: int = 100, shots_per_sample: int = 1000) -> dict:
        """
        Calculate fidelity of |+>_L (Alice) and |0>_L (Bob) product state.

        Uses multi-sample averaging to properly capture stochastic noise.

        For [[7,1,3]] Steane code:
        - Logical Z = Z^{otimes 7} (all physical Z operators)
        - Logical X = X^{otimes 7} (all physical X operators)

        For |+>_L state: X_L eigenvalue should be +1
        For |0>_L state: Z_L eigenvalue should be +1

        Measures:
        - Alice: P(X_L = +1) for |+>_L state
        - Bob: P(Z_L = +1) for |0>_L state
        - Product fidelity: F = P(X_L = +1) * P(Z_L = +1)

        Args:
            num_samples: Number of independent sampling runs to average
            shots_per_sample: Shots per sample

        Returns:
            dict with 'alice_x_prob', 'bob_z_prob', 'fidelity'
        """
        try:
            qm = self.node.timeline.quantum_manager

            # Get Alice's data qubits (this node)
            alice_keys = [mem.qstate_key for mem in self.data_qubits]

            # Get Bob's data qubits
            bob_node = self.node.timeline.get_entity_by_name(self.responder)
            if bob_node is None or not hasattr(bob_node, 'request_logical_pair_app'):
                raise ValueError(f"Cannot access Bob's node {self.responder} for fidelity measurement")
            bob_app = bob_node.request_logical_pair_app
            bob_keys = [mem.qstate_key for mem in bob_app.data_qubits]

            log.logger.info(f"{self.node.name}: Measuring product state fidelity using multi-sample averaging")
            log.logger.info(f"{self.node.name}: Using {num_samples} samples x {shots_per_sample} shots")
            log.logger.info(f"{self.node.name}: Alice keys: {alice_keys}")
            log.logger.info(f"{self.node.name}: Bob keys: {bob_keys}")

            # Get the quantum states
            alice_state = qm.states[alice_keys[0]]
            bob_state = qm.states[bob_keys[0]]

            # Multi-sample averaging
            alice_x_samples = []
            bob_z_samples = []

            for sample_idx in range(num_samples):
                seed = (sample_idx * 12345 + hash(f"{self.node.name}_product")) % (2**63)

                # =====================================================================
                # Measure Alice's X_L (logical X operator = X^{otimes 7})
                # For |+>_L state, X_L|+>_L = +1|+>_L
                # =====================================================================

                # Build measurement circuit for X basis
                alice_meas_circuit = alice_state.circuit.copy()

                # Apply H to measure in X basis
                for key in alice_keys:
                    alice_meas_circuit.append("H", [key])

                # Add measurements
                for key in alice_keys:
                    alice_meas_circuit.append("M", [key])

                # Sample
                alice_seed = abs(seed + hash("alice_x")) % (2**63)
                sampler = alice_meas_circuit.compile_sampler(seed=alice_seed)
                measurements = sampler.sample(shots=shots_per_sample)

                # Get last 7 measurement columns for Alice
                num_meas = measurements.shape[1]
                alice_measurements = measurements[:, num_meas - 7:]

                # Compute logical parity (XOR of all 7 physical measurements)
                alice_parity = np.sum(alice_measurements, axis=1) % 2

                # P(X_L = +1) = fraction where parity is 0
                alice_x_prob = float(np.mean(alice_parity == 0))
                alice_x_samples.append(alice_x_prob)

                # =====================================================================
                # Measure Bob's Z_L (logical Z operator = Z^{otimes 7})
                # For |0>_L state, Z_L|0>_L = +1|0>_L
                # =====================================================================

                # Build measurement circuit for Z basis (no rotation needed)
                bob_meas_circuit = bob_state.circuit.copy()

                # Add measurements (Z basis = computational basis)
                for key in bob_keys:
                    bob_meas_circuit.append("M", [key])

                # Sample
                bob_seed = abs(seed + hash("bob_z")) % (2**63)
                sampler = bob_meas_circuit.compile_sampler(seed=bob_seed)
                measurements = sampler.sample(shots=shots_per_sample)

                # Get last 7 measurement columns for Bob
                num_meas = measurements.shape[1]
                bob_measurements = measurements[:, num_meas - 7:]

                # Compute logical parity
                bob_parity = np.sum(bob_measurements, axis=1) % 2

                # P(Z_L = +1) = fraction where parity is 0
                bob_z_prob = float(np.mean(bob_parity == 0))
                bob_z_samples.append(bob_z_prob)

            # Average over all samples
            alice_x_prob = float(np.mean(alice_x_samples))
            bob_z_prob = float(np.mean(bob_z_samples))
            fidelity = alice_x_prob * bob_z_prob

            log.logger.info(f"{self.node.name}: Alice P(X_L=+1) = {alice_x_prob:.6f} (should be ~1.0 for |+>_L)")
            log.logger.info(f"{self.node.name}: Bob P(Z_L=+1) = {bob_z_prob:.6f} (should be ~1.0 for |0>_L)")
            log.logger.info(f"{self.node.name}: Product state fidelity = {fidelity:.6f}")

            return {
                'alice_x_prob': alice_x_prob,
                'bob_z_prob': bob_z_prob,
                'fidelity': fidelity
            }

        except Exception as e:
            log.logger.error(f"{self.node.name}: Error calculating product state fidelity: {e}")
            import traceback
            traceback.print_exc()
            return {
                'alice_x_prob': None,
                'bob_z_prob': None,
                'fidelity': None
            }
                 
        
    def _calculate_depolarization_probability(self, elapsed_time_ps: float, 
                                         coherence_time_ms: float = 100) -> float:
        """
        Calculate depolarization probability after elapsed time.
        
        Args:
            elapsed_time_ps: Time elapsed since generation (picoseconds)
            coherence_time_ns: T2 coherence time (nanoseconds)
        
        Returns:
            Depolarization probability p âˆˆ [0, 1]
        """
        elapsed_time_ms = elapsed_time_ps * 1e-9
        p = 1.0 - np.exp(- elapsed_time_ms / coherence_time_ms)
        if p < 0.0 or p > 1.0:
            log.logger.warning(f"{self.node.name}: Calculated depolarization probability out of bounds: p={p}")
            p = np.clip(p, 0.0, 1.0)
        return p
    
    ############################ Result Processing and Printing  #########################
        
    def get_results(self) -> Dict:
        """Get structured results for external analysis.
        
        Returns:
            Dictionary containing all results and metrics including
            tomography-based fidelity measurements.
        """
        results = {
            'node_name': self.node.name,
            'role': 'initiator' if self.is_initiator else 'responder',
            'bell_pairs': [],
            'statistics': {},
            'timing': {},
            'encoding': {
                'enabled': self.encoding_enabled,
                'logical_state': self.logical_state if self.encoding_enabled else None,
                'success': self.encoding_complete if self.encoding_enabled else None,
                'encoding_time': None
            },
            'tomography': {
                'method': 'Pauli tomography',
                'target_state': '|ÃŽÂ¦+Ã¢Å¸Â© = (|00Ã¢Å¸Â© + |11Ã¢Å¸Â©)/Ã¢Ë†Å¡2',
                'measurements': '2^2 = 4 Pauli basis measurements per pair'
            }
        }
        
        # Add Bell pair details with tomography fidelities
        for result in self.results:
            results['bell_pairs'].append({
                'pair_id': result['pair_id'],
                'memory_index': result['memory_index'],
                'fidelity': result['fidelity'],  # Tomography-based fidelity
                'generation_time': result['generation_time'],
                'remote_node': result.get('remote_node'),
                'remote_memory': result.get('remote_memory')
            })
        
        # Calculate statistics from tomography results
        if self.results:
            fidelities = [r['fidelity'] for r in self.results if r['fidelity'] is not None]
            if fidelities:
                results['statistics'] = {
                    'num_pairs': len(self.results),
                    'avg_fidelity': float(np.mean(fidelities)),
                    'min_fidelity': float(np.min(fidelities)),
                    'max_fidelity': float(np.max(fidelities)),
                    'std_fidelity': float(np.std(fidelities)),
                    'measurement_method': 'Quantum state tomography'
                }
        
        # Add timing information
        if self.first_pair_time and self.last_pair_time:
            results['timing']['generation_time'] = (self.last_pair_time - self.first_pair_time) * 1e-12
            results['timing']['first_pair_time'] = self.first_pair_time * 1e-12
            results['timing']['last_pair_time'] = self.last_pair_time * 1e-12
        
        if self.encoding_enabled and self.encoding_start_time and self.encoding_end_time:
            results['encoding']['encoding_time'] = (self.encoding_end_time - self.encoding_start_time) * 1e-12
        
        # Add logical fidelity information (always include if teleported_cnot was enabled)
        if self.teleported_cnot_enabled:
            results['logical_bell_pair'] = {
                'fidelity': self.logical_fidelity['fidelity'],  # Will be None if calculation failed
                'correlations': self.logical_fidelity['correlations'],
                'calculation_attempted': True,
                'calculation_successful': self.logical_fidelity['fidelity'] is not None
            }

        # Add product state fidelity if it exists
        if hasattr(self, 'product_state_fidelity'):
            results['product_state_fidelity'] = self.product_state_fidelity

        return results

     
    def print_results(self):
        """Print Bell pair fidelity results."""
        if not self.verbose:
            return

        print(f"\n{'='*70}")
        print(f"Physical Bell Pair Fidelities - {self.node.name}")
        print(f"{'='*70}")

        if self.results:
            # Calculate statistics
            fidelities = [r['fidelity'] for r in self.results if r['fidelity'] is not None]
            if fidelities:
                print(f"\nIndividual Bell Pair Fidelities:")
                for result in self.results:
                    fidelity_str = f"{result['fidelity']:.6f}" if result['fidelity'] is not None else "Error"
                    print(f"  Pair {result['pair_id']}: {fidelity_str}")

                # Summary statistics
                avg_fidelity = np.mean(fidelities)
                min_fidelity = np.min(fidelities)
                max_fidelity = np.max(fidelities)
                std_fidelity = np.std(fidelities)

                print(f"\nStatistics:")
                print(f"  Average: {avg_fidelity:.6f}")
                print(f"  Min:     {min_fidelity:.6f}")
                print(f"  Max:     {max_fidelity:.6f}")
                print(f"  Std Dev: {std_fidelity:.6f}")
            else:
                print(f"\nWARNING: No fidelities calculated")
        else:
            print(f"\nNo Bell pairs generated")

        print(f"{'='*70}\n")
    
