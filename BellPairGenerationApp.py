"""
Request Logical Pair Application for 2nd Generation Quantum Routers.

This module implements an application that:
1. Generates 7 Bell pairs using Barrett-Kok entanglement generation
2. Synchronizes between Alice and Bob before encoding
3. Encodes Alice's 7 data qubits into logical |0⟩_L
4. Encodes Bob's 7 data qubits into logical |+⟩_L

Designed for QuantumRouter2ndGeneration with separate communication, data, 
and ancilla memory arrays.
"""

from typing import TYPE_CHECKING, List, Dict
import numpy as np
import time

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter2ndGeneration
    from sequence.resource_management.memory_manager import MemoryInfo

from sequence.app.request_app import RequestApp
from sequence.resource_management.memory_manager import MemoryInfo
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.message import Message
from sequence.utils import log
from final_QEC713 import QEC713


class SyncMessage(Message):
    """Message for synchronization between Alice and Bob.
    
    Attributes:
        msg_type: Type of sync message ("READY" or "ACK")
        sender: Name of the sending node
    """
    def __init__(self, msg_type: str, sender: str):
        super().__init__(msg_type, sender)
        self.msg_type = msg_type
        self.sender = sender


class RequestLogicalPairApp(RequestApp):
    """Application for generating Bell pairs and encoding logical qubits with QEC.
    
    This app uses Barrett-Kok entanglement generation to create 7 Bell pairs,
    synchronizes between Alice and Bob, then encodes the local data qubits 
    into [[7,1,3]] logical qubits:
    - Alice (initiator): encodes to |0⟩_L
    - Bob (responder): encodes to |+⟩_L
    
    Attributes:
        node (QuantumRouter2ndGeneration): The quantum router node
        results (List[Dict]): Collected results for each Bell pair
        num_completed (int): Number of completed Bell pair measurements
        logical_state (str): '0' for |0⟩_L or '+' for |+⟩_L
        encoding_enabled (bool): Whether to perform QEC encoding
        encoding_start_time (int): When encoding begins (ps)
        encoding_completion_time (int): When encoding completes (ps)
        encoding_success (bool): Did encoding complete without errors
        qec (QEC713): QEC utility instance
        remote_node_name (str): Name of the remote node
        sync_ready (bool): Has this node finished collecting Bell pairs
        remote_ready (bool): Has remote node finished collecting Bell pairs
    """
    
    def __init__(self, node: "QuantumRouter2ndGeneration"):
        """Initialize the request logical pair application.
        
        Args:
            node: The 2nd generation quantum router node
        
        Raises:
            AssertionError: If node is not QuantumRouter2ndGeneration
        """
        super().__init__(node)
        
        # Verify this is a 2nd generation router
        assert hasattr(node, 'data_memo_arr_name'), \
            "Node must be QuantumRouter2ndGeneration with data_memo_arr_name"
        assert hasattr(node, 'ancilla_memo_arr_name'), \
            "Node must be QuantumRouter2ndGeneration with ancilla_memo_arr_name"
        
        # Bell pair generation tracking
        self.results: List[Dict] = []
        self.num_completed = 0
        self.first_pair_time = None
        self.last_pair_time = None
        self.all_pairs_processed = False       # Flag to ensure we only process completion once

        # QEC encoding state
        self.logical_state = None              # '0' for |0⟩_L, '+' for |+⟩_L
        self.encoding_enabled = True           # Flag to enable/disable QEC
        self.encoding_start_time = None        # When encoding begins (ps)
        self.encoding_completion_time = None   # When encoding completes (ps)
        self.encoding_success = False          # Did encoding complete without errors
        self.qec = QEC713()                    # QEC utility instance
        
        # Synchronization state
        self.remote_node_name = None           # Name of remote node
        self.sync_ready = False                # This node finished collecting
        self.remote_ready = False              # Remote node finished collecting
        
        log.logger.info(f"{node.name} RequestLogicalPairApp: initialized (2nd gen)")
        
        # Initialize ONLY ancilla memories (data memories initialized just before encoding)
        self._initialize_ancilla_memories()
    
    def _initialize_ancilla_memories(self):
        """Initialize only ancilla memories to |0⟩ state.
        
        Data memories are NOT initialized here to minimize decoherence.
        They will be initialized just before encoding begins.
        """
        ancilla_array = self.node.components[self.node.ancilla_memo_arr_name]
        for i in range(len(ancilla_array.memories)):
            ancilla_array[i].reset()  # Reset to |0⟩

        log.logger.info(f"{self.node.name}: initialized {len(ancilla_array.memories)} ancilla memories to |0>")
    
    def start(self, remote_node_name: str, start_time: int, end_time: int, 
              memory_size: int = 7, target_fidelity: float = 0.8,
              logical_state: str = '0', encoding_enabled: bool = True):
        """Start generation of Bell pairs with optional QEC encoding.
        
        Args:
            remote_node_name: Name of remote node (Bob)
            start_time: Simulation start time in picoseconds
            end_time: Simulation end time in picoseconds
            memory_size: Number of Bell pairs to generate (default 7)
            target_fidelity: Minimum acceptable fidelity threshold (default 0.8)
            logical_state: '0' for |0⟩_L or '+' for |+⟩_L (default '0' for initiator)
            encoding_enabled: Whether to perform QEC encoding after Bell pair generation
        """
        self.logical_state = logical_state
        self.encoding_enabled = encoding_enabled
        self.remote_node_name = remote_node_name
        
        # Validate logical_state
        assert logical_state in ['0', '+'], "logical_state must be '0' or '+'"
        
        log.logger.info(f"{self.node.name}: Starting with logical_state={logical_state}, encoding_enabled={encoding_enabled}")
        
        # Call parent RequestApp.start() to make the reservation
        super().start(remote_node_name, start_time, end_time, memory_size, target_fidelity)
    
    def get_other_reservation(self, reservation):
        """Called when this node is the responder.
        
        Bob (responder) automatically gets logical state '+'.
        
        Args:
            reservation: The reservation created by the initiator
        """
        super().get_other_reservation(reservation)
        
        # Responder always encodes to |+⟩_L
        self.logical_state = '+'
        self.encoding_enabled = True
        
        # Set remote node name (the initiator)
        self.remote_node_name = reservation.initiator
        
        log.logger.info(f"{self.node.name}: Acting as responder, will encode to |+>_L")
    
    def get_memory(self, info: "MemoryInfo"):
        """Handle incoming entangled memory.

        Called when a Bell pair is generated. Stores the memory info
        and waits until all pairs are collected before calculating fidelity.

        Args:
            info: Memory info containing entanglement details
        """
        log.logger.debug(f"{self.node.name}: get_memory called with state={info.state}, index={info.index}")
        
        if info.state != "ENTANGLED":
            log.logger.debug(f"{self.node.name}: Ignoring non-ENTANGLED memory (state={info.state})")
            return

        # Check if this memory is part of our reservation
        if info.index not in self.memo_to_reservation:
            log.logger.debug(f"{self.node.name}: Memory index {info.index} not in reservations: {list(self.memo_to_reservation.keys())}")
            return

        reservation = self.memo_to_reservation[info.index]

        # Verify this is the correct remote node (the OTHER party in the reservation)
        # If we're the initiator, remote should be responder; if responder, remote should be initiator
        other_party = reservation.responder if self.node.name == reservation.initiator else reservation.initiator
        if info.remote_node != other_party:
            log.logger.debug(f"{self.node.name}: Remote node {info.remote_node} doesn't match expected {other_party}")
            return
        
        # Record generation time
        generation_time = self.node.timeline.now()
        
        # Track first pair time
        if self.first_pair_time is None:
            self.first_pair_time = generation_time
        
        # Always update last pair time
        self.last_pair_time = generation_time
        
        log.logger.info(f"{self.node.name}: Bell pair {self.num_completed} generated "
                       f"at t={generation_time*1e-12:.6f}s on memory {info.index}")

        # Store the memory info (don't calculate fidelity yet - Barrett-Kok may not be done on remote side)
        result = {
            'pair_id': self.num_completed,
            'memory_index': info.index,
            'generation_time': generation_time * 1e-12,  # Convert to seconds
            'memory_info': info,  # Store the info object for later fidelity calculation
            'fidelity': None,  # Will be calculated later
            'remote_node': info.remote_node,
            'remote_memory': info.remote_memo
        }
        self.results.append(result)
        self.num_completed += 1
        
        # DON'T release memory yet - keep it ENTANGLED until after encoding
        # Communication memories need to remain in their entangled state so we can:
        # 1. Calculate fidelity after all pairs are collected
        # 2. Use them as reference during QEC encoding
        # Memory will be released to RAW state after encoding completes in _perform_encoding()
        # or immediately if encoding is disabled

        # Check if all pairs are complete (only trigger once)
        if self.num_completed >= self.memo_size and not self.all_pairs_processed:
            self.all_pairs_processed = True
            self._all_pairs_completed()
    
    def _calculate_fidelity(self, info: "MemoryInfo", remote_node_name: str, remote_memo_name: str) -> float:
        """Calculate Bell pair fidelity via quantum state tomography.

        Uses the density matrix computation from StabilizerState to calculate
        the fidelity with the ideal Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2.

        Formula: F = ⟨Φ⁺|ρ|Φ⁺⟩ = 0.5 × (ρ[0,0] + ρ[0,3] + ρ[3,0] + ρ[3,3])

        Args:
            info: MemoryInfo containing the entangled memory
            remote_node_name: Name of the remote node
            remote_memo_name: Full name of the remote memory entity

        Returns:
            Fidelity value between 0 and 1
            
        Raises:
            AssertionError: If fidelity calculation fails
        """
        # Get quantum manager
        qm = self.node.timeline.quantum_manager

        # Get local memory and key
        local_memory = info.memory
        local_key = local_memory.qstate_key

        # Validate remote memory info
        assert remote_memo_name is not None, \
            f"{self.node.name}: remote_memo_name is None"
        assert remote_node_name is not None, \
            f"{self.node.name}: remote_node_name is None"

        # Find remote memory using the stored name
        remote_memory = self.node.timeline.get_entity_by_name(remote_memo_name)
        assert remote_memory is not None, \
            f"{self.node.name}: could not find remote memory {remote_memo_name}"
        
        remote_key = remote_memory.qstate_key
        
        log.logger.debug(f"{self.node.name}: computing tomography for keys "
                        f"({local_key}, {remote_key})")

        # Check if both keys are in the same quantum state
        local_state = qm.states[local_key]
        remote_state = qm.states[remote_key]
        
        log.logger.debug(f"{self.node.name}: local_state has {len(local_state.keys)} qubits: {local_state.keys}")
        log.logger.debug(f"{self.node.name}: remote_state has {len(remote_state.keys)} qubits: {remote_state.keys}")
        log.logger.debug(f"{self.node.name}: local_state is remote_state: {local_state is remote_state}")
        
        assert local_state is remote_state, \
            f"{self.node.name}: local and remote qubits are in different quantum states. " \
            f"Local state: {local_state.keys}, Remote state: {remote_state.keys}. " \
            f"This likely means the remote node has already performed operations on its qubit. " \
            f"Fidelity must be calculated before any operations modify the Bell pair."
        
        state = local_state
        
        log.logger.debug(f"{self.node.name}: state contains {len(state.keys)} qubits: {state.keys}")
        
        # If there are more than 2 qubits, we need to trace out the others
        if len(state.keys) == 2:
            # Perfect case - just our Bell pair
            rho = state._compute_density_matrix()
        else:
            # There are additional qubits - compute full density matrix and trace out unwanted qubits
            log.logger.debug(f"{self.node.name}: tracing out extra qubits from {len(state.keys)}-qubit state")
            
            # Compute full density matrix
            full_rho = state._compute_density_matrix()
            
            # Get indices of our two qubits in the state
            local_idx = state.keys.index(local_key)
            remote_idx = state.keys.index(remote_key)
            
            # Trace out all qubits except our two
            rho = self._partial_trace(full_rho, len(state.keys), [local_idx, remote_idx])

        # Verify density matrix shape
        assert rho.shape == (4, 4), \
            f"{self.node.name}: expected 4x4 density matrix, got {rho.shape}"

        # Debug: Print density matrix
        log.logger.debug(f"{self.node.name}: Density matrix:\n{rho}")

        # Calculate fidelity with ideal Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        # In computational basis: |Φ⁺⟩ = [1/√2, 0, 0, 1/√2]^T
        # F = ⟨Φ⁺|ρ|Φ⁺⟩ = 0.5 * (ρ₀₀ + ρ₀₃ + ρ₃₀ + ρ₃₃)
        fidelity = 0.5 * (rho[0, 0] + rho[0, 3] + rho[3, 0] + rho[3, 3])
        fidelity = np.real(fidelity)  # Take real part

        # Debug: Print intermediate values
        log.logger.debug(f"{self.node.name}: rho[0,0]={rho[0,0]}, rho[0,3]={rho[0,3]}, rho[3,0]={rho[3,0]}, rho[3,3]={rho[3,3]}")
        log.logger.debug(f"{self.node.name}: Raw fidelity before clamping: {fidelity}")

        # Clamp to [0, 1] range to handle numerical errors
        fidelity = max(0.0, min(1.0, float(fidelity)))

        log.logger.info(f"{self.node.name}: calculated fidelity = {fidelity:.6f}")
        
        return fidelity

    def _partial_trace(self, rho: np.ndarray, n_qubits: int, keep_indices: List[int]) -> np.ndarray:
        """Compute partial trace to keep only specified qubits.
        
        Args:
            rho: Full density matrix of shape (2^n, 2^n)
            n_qubits: Total number of qubits
            keep_indices: List of qubit indices to keep (0-indexed)
            
        Returns:
            Reduced density matrix for kept qubits
        """
        dim = 2 ** n_qubits
        assert rho.shape == (dim, dim), f"Invalid density matrix shape: {rho.shape}"
        
        # Create list of all qubit indices
        all_indices = list(range(n_qubits))
        
        # Indices to trace out
        trace_out = [i for i in all_indices if i not in keep_indices]
        
        # Reshape density matrix to separate each qubit's Hilbert space
        shape = [2] * n_qubits + [2] * n_qubits
        rho_reshaped = rho.reshape(shape)
        
        # Trace out unwanted qubits (from highest index to lowest to maintain indexing)
        for idx in sorted(trace_out, reverse=True):
            # Sum over the idx-th qubit (both in ket and bra spaces)
            rho_reshaped = np.trace(rho_reshaped, axis1=idx, axis2=idx + n_qubits)
            n_qubits -= 1
        
        # Reshape back to matrix form
        final_dim = 2 ** len(keep_indices)
        result = rho_reshaped.reshape(final_dim, final_dim)
        
        return result
    
    def _all_pairs_completed(self):
        """Called when all 7 Bell pairs have been generated.
        
        Calculates fidelity for all Bell pairs (now that Barrett-Kok is complete on both sides),
        then sends a READY message to the remote node to synchronize before encoding.
        """
        log.logger.info(f"{self.node.name}: All {self.num_completed} Bell pairs generated!")
        
        # FIRST: Calculate fidelity for all Bell pairs
        # At this point, Barrett-Kok should be complete on both sides
        log.logger.info(f"{self.node.name}: Calculating fidelity for all {self.num_completed} Bell pairs...")
        
        for result in self.results:
            info = result['memory_info']
            remote_node = result['remote_node']
            remote_memory = result['remote_memory']
            
            measured_fidelity = self._calculate_fidelity(info, remote_node, remote_memory)
            result['fidelity'] = measured_fidelity
            log.logger.info(f"{self.node.name}: Bell pair {result['pair_id']} "
                           f"fidelity = {measured_fidelity:.6f}")
        
        # SECOND: Print Bell pair results (now that fidelities are calculated)
        self.print_results()
        
        # THIRD: Mark this node as ready
        self.sync_ready = True
        
        # FOURTH: Send READY message to remote node
        if self.encoding_enabled and self.remote_node_name:
            log.logger.info(f"{self.node.name}: Sending READY message to {self.remote_node_name}")
            self._send_sync_message("READY")
            
            # Check if remote is also ready
            if self.remote_ready:
                log.logger.info(f"{self.node.name}: Remote also ready, starting encoding immediately")
                self._start_encoding()
        else:
            # No encoding - release memories now
            log.logger.info(f"{self.node.name}: No encoding requested, releasing memories to RAW")
            self._release_memories()
            log.logger.info(f"{self.node.name}: Simulation continues")
    
    def _release_memories(self):
        """Release all communication memories to RAW state.
        
        Sets memories back to RAW state, making them available for future
        entanglement operations. Called after encoding completes or if 
        encoding is disabled.
        """
        for result in self.results:
            memory = result['memory_info'].memory
            self.node.resource_manager.update(None, memory, "RAW")
        log.logger.info(f"{self.node.name}: Released {len(self.results)} memories to RAW state")
    
    def _send_sync_message(self, msg_type: str):
        """Send a synchronization message to the remote node.
        
        Args:
            msg_type: Type of message ("READY" or "ACK")
        """
        msg = SyncMessage(msg_type, self.node.name)
        self.node.send_message(self.remote_node_name, msg)
    
    def received_message(self, src: str, msg: "Message"):
        """Handle received synchronization messages.
        
        Args:
            src: Source node name
            msg: The received message
        """
        if not isinstance(msg, SyncMessage):
            return
        
        log.logger.info(f"{self.node.name}: Received {msg.msg_type} from {src}")
        
        if msg.msg_type == "READY":
            # Remote node has finished collecting Bell pairs
            self.remote_ready = True
            
            # If we're also ready, send ACK and start encoding
            if self.sync_ready:
                log.logger.info(f"{self.node.name}: Both nodes ready, sending ACK and starting encoding")
                self._send_sync_message("ACK")
                self._start_encoding()
            else:
                log.logger.info(f"{self.node.name}: Waiting to finish our Bell pairs...")
        
        elif msg.msg_type == "ACK":
            # Remote node acknowledged, start encoding
            if self.sync_ready:
                log.logger.info(f"{self.node.name}: Received ACK, starting encoding")
                self._start_encoding()
            else:
                log.logger.warning(f"{self.node.name}: Received ACK but not ready yet")
    
    def _start_encoding(self):
        """Initialize data memories and schedule QEC encoding.

        Sequence:
        1. Initialize 7 data memories to |0⟩ RIGHT BEFORE encoding (minimize decoherence)
        2. For Bob (logical_state='+'): Apply H gates to all 7
        3. Schedule encoding with small delay
        """
        # Check if we've already started encoding
        if self.encoding_start_time is not None:
            log.logger.debug(f"{self.node.name}: Encoding already started, ignoring duplicate call")
            return
        
        log.logger.info(f"{self.node.name}: Starting encoding to logical |{self.logical_state}>_L")

        # TIMING: Start
        t_start = time.time()

        # Initialize data memories to |0⟩ RIGHT BEFORE encoding (minimize decoherence)
        data_array = self.node.components[self.node.data_memo_arr_name]

        # TIMING: Memory reset
        t_reset_start = time.time()
        for i in range(7):
            data_array[i].reset()  # |0⟩ state
        t_reset_end = time.time()

        # Check quantum states after reset
        qm = self.node.timeline.quantum_manager
        data_keys = [data_array[i].qstate_key for i in range(7)]
        unique_states = set(qm.states[key] for key in data_keys)

        print(f"\n{'='*70}")
        print(f"[{self.node.name}] _start_encoding() Timing Analysis")
        print(f"{'='*70}")
        print(f"Memory reset time: {t_reset_end - t_reset_start:.6f}s")
        print(f"Number of separate quantum states after reset: {len(unique_states)}")
        print(f"Data qubit keys: {data_keys}")
        log.logger.debug(f"{self.node.name}: Initialized 7 data memories to |0>")
        log.logger.info(f"{self.node.name}: After reset - {len(unique_states)} separate quantum states")

        # For Bob: apply H gates to create |+++++++⟩ before encoding
        if self.logical_state == '+':
            t_plus_start = time.time()
            self._prepare_plus_state()
            t_plus_end = time.time()
            print(f"Prepare |+> state time: {t_plus_end - t_plus_start:.6f}s")

        t_total = time.time() - t_start
        print(f"Total _start_encoding() time: {t_total:.6f}s")
        print(f"{'='*70}\n")

        # Schedule encoding with small delay (realistic timing)
        delay = 1000  # 1 nanosecond delay
        process = Process(self, '_perform_encoding', [])
        event = Event(self.node.timeline.now() + delay, process)
        self.node.timeline.schedule(event)
    
    def _prepare_plus_state(self):
        """Apply Hadamard gates to all 7 data qubits to create |+++++++⟩.

        Only called for Bob (logical_state='+').
        """
        qm = self.node.timeline.quantum_manager
        data_array = self.node.components[self.node.data_memo_arr_name]

        # Get all 7 data qubit keys
        data_keys = [data_array[i].qstate_key for i in range(7)]

        # Check states before grouping
        unique_states_before = set(qm.states[key] for key in data_keys)
        print(f"[{self.node.name}] _prepare_plus_state() - states before grouping: {len(unique_states_before)}")

        # TIMING: Group qubits together
        t_group_start = time.time()
        qm.group_qubits(data_keys)
        t_group_end = time.time()
        print(f"[{self.node.name}] _prepare_plus_state() - group_qubits() time: {t_group_end - t_group_start:.6f}s")

        state = qm.states[data_keys[0]]

        # Apply H gate to each qubit: |0⟩ → |+⟩
        t_h_start = time.time()
        for key in data_keys:
            state.circuit.append("H", [key])

        state._tableau = None  # Invalidate cached tableau
        t_h_end = time.time()
        print(f"[{self.node.name}] _prepare_plus_state() - H gates time: {t_h_end - t_h_start:.6f}s")

        log.logger.debug(f"{self.node.name}: Applied H gates to create |+++++++>")
    
    def _perform_encoding(self):
        """Perform [[7,1,3]] QEC encoding on the 7 data qubits.

        Alice: |0000000⟩ → encode → |0⟩_L
        Bob: |+++++++⟩ → encode → |+⟩_L
        """
        self.encoding_start_time = self.node.timeline.now()

        # TIMING: Real-time performance tracking
        t_start = time.time()

        qm = self.node.timeline.quantum_manager
        data_array = self.node.components[self.node.data_memo_arr_name]

        # Get the 7 data qubit keys (in order!)
        data_keys = [data_array[i].qstate_key for i in range(7)]

        print(f"\n{'='*70}")
        print(f"[{self.node.name}] _perform_encoding() Timing Analysis")
        print(f"{'='*70}")

        # Group qubits only for Alice (Bob already grouped in _prepare_plus_state)
        if self.logical_state == '0':
            # Check states before grouping
            unique_states_before = set(qm.states[key] for key in data_keys)
            print(f"Alice - states before grouping: {len(unique_states_before)}")

            # TIMING: group_qubits for Alice
            t_group_start = time.time()
            qm.group_qubits(data_keys)
            t_group_end = time.time()
            print(f"Alice - group_qubits() time: {t_group_end - t_group_start:.6f}s")
        else:
            print(f"Bob - skipping group_qubits (already grouped)")

        # TIMING: QEC encoding
        t_encode_start = time.time()
        self.qec.encode(qm, data_keys)
        t_encode_end = time.time()
        print(f"QEC713.encode() time: {t_encode_end - t_encode_start:.6f}s")

        # Record completion
        self.encoding_completion_time = self.node.timeline.now()
        self.encoding_success = True

        # NOW release all communication memories to RAW state
        # (encoding is complete, Bell pairs no longer needed)
        log.logger.info(f"{self.node.name}: Encoding complete, releasing communication memories")
        self._release_memories()

        t_total = time.time() - t_start
        print(f"Total _perform_encoding() time: {t_total:.6f}s")
        print(f"{'='*70}\n")

        encoding_time = (self.encoding_completion_time - self.encoding_start_time) * 1e-12
        log.logger.info(f"{self.node.name}: Successfully encoded to logical |{self.logical_state}>_L in {encoding_time:.6f}s")

        # Print encoding metrics
        self._print_encoding_metrics()
    
    def _print_encoding_metrics(self):
        """Print encoding performance metrics."""

        if not self.encoding_success:
            print(f"\n{self.node.name}: Encoding FAILED")
            return

        encoding_time = (self.encoding_completion_time - self.encoding_start_time) * 1e-12

        print(f"\n{'='*70}")
        print(f"QEC Encoding Results - Node: {self.node.name}")
        print(f"{'='*70}")
        print(f"Target logical state: |{self.logical_state}>_L")
        print(f"Encoding time: {encoding_time:.6f}s")
        print(f"Status: {'SUCCESS' if self.encoding_success else 'FAILED'}")
        print(f"{'='*70}\n")
    
    def get_results(self) -> Dict:
        """Get comprehensive results including Bell pairs and encoding.
        
        Returns:
            Dictionary with 'bell_pairs' and 'encoding' metrics
        """
        results = {
            'bell_pairs': sorted(self.results, key=lambda x: x['pair_id']),
            'encoding': {
                'logical_state': self.logical_state,
                'encoding_time': (self.encoding_completion_time - self.encoding_start_time) * 1e-12 if self.encoding_completion_time else None,
                'success': self.encoding_success
            }
        }
        return results
    
    def get_simulation_runtime(self) -> float:
        """Get the simulation duration in seconds (from timeline).
        
        Returns:
            Simulation duration in seconds, or None if not completed
        """
        if self.first_pair_time and self.last_pair_time:
            return (self.last_pair_time - self.first_pair_time) * 1e-12
        return None
    
    def print_results(self):
        """Print formatted results of all Bell pair generations."""
        if not self.results:
            print(f"\n{self.node.name}: No Bell pairs generated yet")
            return
        
        print(f"\n{'='*70}")
        print(f"Bell Pair Generation Results - Node: {self.node.name}")
        print(f"{'='*70}")
        
        # Calculate statistics (filter out None values)
        fidelities = [r['fidelity'] for r in self.results if r['fidelity'] is not None]
        
        if fidelities:
            avg_fidelity = np.mean(fidelities)
            min_fidelity = np.min(fidelities)
            max_fidelity = np.max(fidelities)
            
            print(f"Total pairs generated: {len(self.results)}")
            print(f"Average fidelity: {avg_fidelity:.6f}")
            print(f"Min fidelity: {min_fidelity:.6f}")
            print(f"Max fidelity: {max_fidelity:.6f}")
        else:
            print(f"Total pairs generated: {len(self.results)}")
            print(f"Fidelity: Not yet calculated")
        
        # Simulation runtime
        runtime = self.get_simulation_runtime()
        if runtime:
            print(f"Generation time: {runtime:.6f}s")
        
        # Detailed results
        print(f"\nDetailed Results:")
        print(f"{'Pair ID':<10} {'Memory':<10} {'Time (s)':<15} {'Fidelity':<12} {'Remote Node':<15}")
        print("-" * 70)
        
        for result in sorted(self.results, key=lambda x: x['pair_id']):
            fid_str = f"{result['fidelity']:.6f}" if result['fidelity'] is not None else "N/A"
            print(f"{result['pair_id']:<10} "
                  f"{result['memory_index']:<10} "
                  f"{result['generation_time']:<15.6f} "
                  f"{fid_str:<12} "
                  f"{result['remote_node']:<15}")
        
        print(f"{'='*70}")
        
        # Add encoding status at the end
        if self.encoding_enabled:
            status = 'Waiting for sync' if not self.encoding_success else f'Encoded to |{self.logical_state}>_L'
            print(f"\nQEC Encoding: {status}")
            print(f"{'='*70}\n")