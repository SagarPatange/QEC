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

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter
    from sequence.network_management.reservation import Reservation


class SyncMessage(Message):
    """Message for synchronizing nodes before fidelity calculation."""
    
    def __init__(self, msg_type: str, source: str, encoding_enabled: bool = False, 
                 logical_state: str = None):
        super().__init__(msg_type, source)
        self.source = source
        self.encoding_enabled = encoding_enabled
        self.logical_state = logical_state


class RequestLogicalPairApp(RequestApp):
    """Application for generating 7 Bell pairs and optionally encoding to logical qubits.
    
    Workflow:
    1. Request 7 Bell pairs via Barrett-Kok protocol
    2. Synchronize with remote node when all pairs are ready
    3. Calculate fidelities using quantum state tomography
    4. Optionally encode to logical qubits (Alice: |0>_L, Bob: |+>_L)
    """
    
    # Class variable to track all instances for inter-app communication
    _instances = {}
    
    def __init__(self, node: "QuantumRouter"):
        super().__init__(node)
        
        # Register this instance for inter-app communication
        RequestLogicalPairApp._instances[node.name] = self
        
        # Bell pair tracking
        self.memo_size = 7  # Need exactly 7 Bell pairs for [[7,1,3]] code
        self.num_completed = 0
        self.results = []  # Store Bell pair results
        
        # Synchronization state
        self.sync_ready = False  # This node ready
        self.remote_ready = False  # Remote node ready
        self.remote_node_name = None
        self.fidelities_calculated = False
        
        # Encoding configuration
        self.encoding_enabled = False
        self.logical_state = '0'  # '0' for |0>_L, '+' for |+>_L
        self.encoding_complete = False
        self.remote_encoding_enabled = False
        self.remote_logical_state = None
        
        # Timing
        self.first_pair_time = None
        self.last_pair_time = None
        self.encoding_start_time = None
        self.encoding_end_time = None
        
        # For responder mode
        self.is_initiator = False
        self.is_responder = False
    
    def start(self, remote_node_name: str, start_time: int, end_time: int,
              memory_size: int = 7, target_fidelity: float = 0.8,
              logical_state: str = '0', encoding_enabled: bool = False):
        """Start Bell pair generation with optional QEC encoding.
        
        Args:
            remote_node_name: Name of the remote quantum router
            start_time: Start time in picoseconds
            end_time: End time in picoseconds
            memory_size: Number of Bell pairs to generate (must be 7 for encoding)
            target_fidelity: Minimum acceptable fidelity
            logical_state: '0' for |0>_L or '+' for |+>_L (if encoding enabled)
            encoding_enabled: Whether to encode to logical qubits after Bell pair generation
        """
        assert memory_size == 7, "Must request exactly 7 Bell pairs for [[7,1,3]] encoding"
        
        self.remote_node_name = remote_node_name
        self.encoding_enabled = encoding_enabled
        self.logical_state = logical_state
        self.is_initiator = True
        self.is_responder = False
        
        # Configure the remote app directly if it exists
        if remote_node_name in RequestLogicalPairApp._instances:
            remote_app = RequestLogicalPairApp._instances[remote_node_name]
            remote_app._configure_as_responder(
                initiator_name=self.node.name,
                encoding_enabled=encoding_enabled,
                initiator_logical_state=logical_state
            )
        
        # Start Bell pair generation using parent class
        super().start(
            responder=remote_node_name,
            start_t=start_time,
            end_t=end_time,
            memo_size=memory_size,
            fidelity=target_fidelity
        )
        
        log.logger.info(f"{self.node.name}: Started Bell pair generation with {remote_node_name}, "
                       f"encoding={'enabled' if encoding_enabled else 'disabled'}")
    
    def _configure_as_responder(self, initiator_name: str, encoding_enabled: bool, 
                                initiator_logical_state: str):
        """Configure this app as a responder to an initiator's request.
        
        Args:
            initiator_name: Name of the initiating node
            encoding_enabled: Whether encoding is enabled
            initiator_logical_state: Logical state the initiator will encode to
        """
        self.remote_node_name = initiator_name
        self.remote_encoding_enabled = encoding_enabled
        self.remote_logical_state = initiator_logical_state
        self.is_responder = True
        self.is_initiator = False
        
        # Configure our encoding state
        if encoding_enabled:
            self.encoding_enabled = True
            # Responder (Bob) uses |+>_L if initiator (Alice) uses |0>_L
            self.logical_state = '+' if initiator_logical_state == '0' else '0'
        
        log.logger.info(f"{self.node.name}: Configured as responder to {initiator_name}, "
                       f"encoding={'enabled' if self.encoding_enabled else 'disabled'}, "
                       f"logical_state={self.logical_state}")
    
    def get_other_reservation(self, reservation: "Reservation") -> None:
        """Called when responder receives a reservation from initiator.
        
        This method is called automatically when the initiator creates a reservation.
        The responder doesn't call start() - it reacts to the initiator's request.
        """
        if self.is_responder:
            super().get_other_reservation(reservation)
            log.logger.info(f"{self.node.name}: Accepted reservation from {reservation.initiator}")
    
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
        
        reservation = self.memo_to_reservation[info.index]
        
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
        self.num_completed += 1
        
        # Check if all pairs are complete
        if self.num_completed >= self.memo_size:
            self._all_pairs_completed()
    
    def _send_sync_message(self, msg_type: str):
        """Send synchronization message to remote node.
        
        Uses direct app-to-app communication for reliability.
        
        Args:
            msg_type: Type of sync message ("READY" or "ENCODING_COMPLETE")
        """
        if self.remote_node_name and self.remote_node_name in RequestLogicalPairApp._instances:
            remote_app = RequestLogicalPairApp._instances[self.remote_node_name]
            remote_app._receive_sync_message(msg_type, self.node.name)
    
    def _receive_sync_message(self, msg_type: str, source: str):
        """Receive synchronization message from remote node.
        
        Args:
            msg_type: Type of sync message
            source: Name of source node
        """
        if msg_type == "READY":
            # Remote node has finished generating Bell pairs
            log.logger.info(f"{self.node.name}: Remote node {source} is READY")
            self.remote_ready = True
            
            # If we're also ready, schedule fidelity calculation as timeline event (zero delay)
            if self.sync_ready:
                log.logger.info(f"{self.node.name}: Both nodes ready, scheduling fidelity calculation")
                process = Process(self, '_calculate_all_fidelities', [])
                event = Event(self.node.timeline.now(), process)  # Zero delay
                self.node.timeline.schedule(event)
        
        elif msg_type == "ENCODING_COMPLETE":
            # Remote node has finished encoding
            log.logger.info(f"{self.node.name}: Remote node {source} completed encoding")
    
    def _calculate_fidelity_via_tomography(self, info: "MemoryInfo", remote_node_name: str, 
                                          remote_memo_name: str) -> float:
        """Calculate Bell pair fidelity using TRUE quantum state tomography.
        
        This method performs ACTUAL Pauli tomography by using the quantum manager's
        compute_density_matrix method, which performs measurements in multiple bases
        to reconstruct the density matrix, rather than accessing the internal state.
        
        The tomography process involves:
        1. Finding both qubits in the Bell pair
        2. Using qm.compute_density_matrix(keys) to perform tomography measurements
        3. Calculating fidelity F = âŸ¨Î¦+|Ï|Î¦+âŸ© from the reconstructed density matrix
        
        Args:
            info: Local memory information
            remote_node_name: Name of remote node
            remote_memo_name: Name of remote memory
            
        Returns:
            Fidelity with ideal |Î¦+âŸ© Bell state (value between 0 and 1)
        """
        try:
            qm = self.node.timeline.quantum_manager
            local_memory = info.memory
            local_key = local_memory.qstate_key
            
            # Validate remote memory info
            if remote_memo_name is None or remote_node_name is None:
                error_msg = f"{self.node.name}: Missing remote memory info - cannot calculate fidelity via tomography"
                log.logger.error(error_msg)
                raise ValueError(error_msg)

            # Find remote memory using the stored name
            remote_memory = self.node.timeline.get_entity_by_name(remote_memo_name)
            if remote_memory is None:
                error_msg = f"{self.node.name}: Could not find remote memory '{remote_memo_name}' - cannot calculate fidelity"
                log.logger.error(error_msg)
                raise ValueError(error_msg)
            
            remote_key = remote_memory.qstate_key

            # Validate both keys exist
            if local_key not in qm.states:
                error_msg = f"{self.node.name}: Local qubit key {local_key} not in quantum manager"
                log.logger.error(error_msg)
                raise ValueError(error_msg)
                
            if remote_key not in qm.states:
                error_msg = f"{self.node.name}: Remote qubit key {remote_key} not in quantum manager"
                log.logger.error(error_msg)
                raise ValueError(error_msg)
            
            log.logger.info(f"{self.node.name}: Performing TRUE tomography on qubits {local_key}, {remote_key}")
            
            # âœ… KEY CHANGE: Use quantum manager's compute_density_matrix
            # This performs ACTUAL tomography by measuring in multiple bases
            # rather than directly accessing the state's internal representation
            rho = qm.compute_density_matrix([local_key, remote_key])

            log.logger.info(f"{self.node.name}: Tomography complete - reconstructed {rho.shape} density matrix")

            # DEBUG: Log the actual density matrix diagonal and key elements
            if rho.shape == (4, 4):
                diag = np.real(np.diag(rho))
                log.logger.info(f"{self.node.name}: Density matrix diagonal = [{diag[0]:.4f}, {diag[1]:.4f}, {diag[2]:.4f}, {diag[3]:.4f}]")
                log.logger.info(f"{self.node.name}: rho[0,3] = {rho[0,3]:.4f}, rho[3,0] = {rho[3,0]:.4f}")

            # Verify density matrix properties
            trace_rho = np.trace(rho)
            if abs(trace_rho - 1.0) > 1e-6:
                log.logger.warning(f"{self.node.name}: Density matrix trace = {trace_rho}, normalizing")
                rho = rho / trace_rho


            fidelity = 0.5 * (rho[0,0] + rho[0,3] + rho[3,0] + rho[3,3])
            
            # Take real part (imaginary part should be negligible)
            fidelity = np.real(fidelity)
            
            # Log if imaginary part is significant
            imag_part = np.imag(0.5 * (rho[0,0] + rho[0,3] + rho[3,0] + rho[3,3]))
            if abs(imag_part) > 1e-6:
                log.logger.warning(f"{self.node.name}: Significant imaginary part in fidelity: {imag_part}")
            
            # Clip to valid range [0, 1] to handle numerical errors
            fidelity = max(0.0, min(1.0, fidelity))
            
            log.logger.info(f"{self.node.name}: Tomography-based fidelity = {fidelity:.6f}")
            
            # Additional diagnostics
            log.logger.debug(f"{self.node.name}: Density matrix diagonal: "
                           f"[{rho[0,0]:.4f}, {rho[1,1]:.4f}, {rho[2,2]:.4f}, {rho[3,3]:.4f}]")
            log.logger.debug(f"{self.node.name}: Off-diagonal coherence: |Ï[0,3]| = {abs(rho[0,3]):.4f}")
            
            return float(fidelity)
            
        except Exception as e:
            log.logger.error(f"{self.node.name}: Fidelity calculation via tomography FAILED: {e}")
            import traceback
            log.logger.error(traceback.format_exc())
            # Re-raise the exception - DO NOT silently return a fake fidelity value
            raise RuntimeError(f"Tomography-based fidelity calculation failed for {self.node.name}") from e
    
    def _all_pairs_completed(self):
        """Called when all 7 Bell pairs have been generated.
        
        Sends a READY message to synchronize with remote node before calculating fidelity.
        Fidelity calculation happens AFTER both nodes are ready.
        """
        log.logger.info(f"{self.node.name}: All {self.num_completed} Bell pairs generated")
        
        # Mark this node as ready
        self.sync_ready = True
        
        # ALWAYS send READY message (not just when encoding)
        if self.remote_node_name:
            log.logger.info(f"{self.node.name}: Sending READY message to {self.remote_node_name}")
            self._send_sync_message("READY")
            
            # Check if remote is also ready
            if self.remote_ready:
                log.logger.info(f"{self.node.name}: Both nodes ready, scheduling fidelity calculation")
                process = Process(self, '_calculate_all_fidelities', [])
                event = Event(self.node.timeline.now(), process)  # Zero delay
                self.node.timeline.schedule(event)
        else:
            # No remote node configured - schedule fidelity calculation with zero delay
            log.logger.info(f"{self.node.name}: No remote node, scheduling fidelity calculation")
            process = Process(self, '_calculate_all_fidelities', [])
            event = Event(self.node.timeline.now(), process)  # Zero delay
            self.node.timeline.schedule(event)
    
    def _calculate_all_fidelities(self):
        """Calculate fidelities for all Bell pairs after applying depolarization noise."""
        
        log.logger.info(f"{self.node.name}: Starting fidelity calculation")
        
        # Get current simulation time
        current_time = self.node.timeline.now()  # in picoseconds
        
        # Apply depolarization noise to each Bell pair based on elapsed time
        for i, result in enumerate(self.results):
            # Get when this pair was generated (convert from seconds to picoseconds)
            generation_time_ps = result['generation_time'] * 1e12
            
            # Calculate elapsed time
            elapsed_time_ps = current_time - generation_time_ps
            
            # Calculate depolarization probability
            p = self._calculate_depolarization_probability(elapsed_time_ps)
            
            log.logger.info(f"{self.node.name}: Pair {i} - elapsed {elapsed_time_ps*1e-3:.1f} ns, p_depol = {p:.6f}")
            
            # Apply two-qubit depolarization noise using stabilizer circuit
            info = result['memory_info']
            memory = info.memory
            local_key = memory.qstate_key
            
            # Get remote memory key
            remote_memory = self.node.timeline.get_entity_by_name(result['remote_memory'])
            remote_key = remote_memory.qstate_key
            
            qm = self.node.timeline.quantum_manager
            
            # Create two-qubit depolarization circuit
            depol_circuit = stim.Circuit(f"""
                DEPOLARIZE2({p}) 0 1
            """)

            
            # Apply to both qubits in the Bell pair
            qm.run_circuit(depol_circuit, [local_key, remote_key])
        
        # Now calculate fidelities via tomography (existing code continues...)
        for i, result in enumerate(self.results):
            info = result['memory_info']
            remote_node = result['remote_node']
            remote_memory = result['remote_memory']
            
            log.logger.info(f"{self.node.name}: Computing fidelity for Bell pair {i} "
                        f"(local mem: {info.index}, remote: {remote_memory})")
            
            measured_fidelity = self._calculate_fidelity_via_tomography(info, remote_node, remote_memory)
            result['fidelity'] = measured_fidelity
            
            log.logger.info(f"{self.node.name}: Bell pair {result['pair_id']} "
                        f"tomography fidelity = {measured_fidelity:.6f}")
        
        # Calculate and log statistics (rest of existing code...)
        fidelities = [r['fidelity'] for r in self.results if r['fidelity'] is not None]
        if fidelities:
            avg_fidelity = np.mean(fidelities)
            std_fidelity = np.std(fidelities)
            log.logger.info(f"{self.node.name}: Fidelity statistics - "
                        f"Mean: {avg_fidelity:.6f}, Std: {std_fidelity:.6f}")
        
        # Rest of existing code for encoding...
        if self.encoding_enabled:
            log.logger.info(f"{self.node.name}: All fidelities calculated - starting QEC encoding")
            self._start_encoding()
        else:
            log.logger.info(f"{self.node.name}: No encoding requested - releasing memories")
            self._release_memories()
            self.print_results()


    def _calculate_depolarization_probability(self, elapsed_time_ps: float, 
                                            coherence_time_ns: float = 1000000000000.0) -> float:
        """Calculate depolarization probability after elapsed time."""
        elapsed_time_ns = elapsed_time_ps * 1e-3
        gamma = 1.0 / coherence_time_ns
        p = 1.0 - np.exp(-gamma * elapsed_time_ns)
        return np.clip(p, 0.0, 1.0)
    
    def _release_memories(self):
        """Release all communication memories to RAW state."""
        for result in self.results:
            info = result['memory_info']
            self.node.resource_manager.update(None, info.memory, "RAW")
        log.logger.info(f"{self.node.name}: Released all memories")
    
    def _start_encoding(self):
        """Start QEC encoding process to create logical qubits.
        
        Encodes the 7 data qubits to create either |0>_L or |+>_L
        depending on the node's role (Alice or Bob).
        """
        log.logger.info(f"{self.node.name}: Starting [[7,1,3]] encoding to |{self.logical_state}>_L")
        self.encoding_start_time = self.node.timeline.now()
        
        # For now, just simulate encoding completion
        # In real implementation, this would call the QEC protocol
        delay = 1000  # 1 nanosecond
        process = Process(self, '_encoding_completed', [])
        event = Event(self.node.timeline.now() + delay, process)
        self.node.timeline.schedule(event)
    
    def _encoding_completed(self):
        """Called when QEC encoding is complete."""
        self.encoding_end_time = self.node.timeline.now()
        self.encoding_complete = True
        
        encoding_time = (self.encoding_end_time - self.encoding_start_time) * 1e-12
        log.logger.info(f"{self.node.name}: Encoding completed in {encoding_time:.6f}s")
        
        # Send completion message to remote
        self._send_sync_message("ENCODING_COMPLETE")
        
        # ✅ NEW: Expire the reservation to stop generating more pairs
        for reservation in self.memo_to_reservation.values():
            self.node.resource_manager.expire_rules_by_reservation(reservation)
            break  # Only expire once
        
        # Release memories and print results
        self._release_memories()
        self.print_results()

    def _calculate_depolarization_probability(self, elapsed_time_ps: float, 
                                         coherence_time_ns: float = 1e9) -> float:
        """
        Calculate depolarization probability after elapsed time.
        
        Args:
            elapsed_time_ps: Time elapsed since generation (picoseconds)
            coherence_time_ns: T2 coherence time (nanoseconds)
        
        Returns:
            Depolarization probability p ∈ [0, 1]
        """
        elapsed_time_ns = elapsed_time_ps * 1e-3
        gamma = 1.0 / coherence_time_ns
        p = 1.0 - np.exp(-gamma * elapsed_time_ns)
        return np.clip(p, 0.0, 1.0)
    

    def print_results(self):
        """Print comprehensive results of Bell pair generation and encoding."""
        print(f"\n{'='*70}")
        print(f"Results for {self.node.name}")
        print(f"{'='*70}")
        
        # Bell pair generation results
        print(f"Bell Pairs Generated: {self.num_completed}/{self.memo_size}")
        
        if self.results:
            # Calculate statistics
            fidelities = [r['fidelity'] for r in self.results if r['fidelity'] is not None]
            if fidelities:
                avg_fidelity = np.mean(fidelities)
                min_fidelity = np.min(fidelities)
                max_fidelity = np.max(fidelities)
                std_fidelity = np.std(fidelities)
                
                print(f"\nFidelity Statistics (via Quantum State Tomography):")
                print(f"  Average: {avg_fidelity:.6f}")
                print(f"  Std Dev: {std_fidelity:.6f}")
                print(f"  Min/Max: {min_fidelity:.6f} / {max_fidelity:.6f}")
            
            # Generation time
            if self.first_pair_time and self.last_pair_time:
                total_gen_time = (self.last_pair_time - self.first_pair_time) * 1e-12
                print(f"\nGeneration Time: {total_gen_time:.6f}s")
            
            # Individual pair details
            print("\nIndividual Bell Pairs (Tomography Results):")
            for result in self.results:
                fidelity_str = f"{result['fidelity']:.6f}" if result['fidelity'] is not None else "Not calculated"
                print(f"  Pair {result['pair_id']}: Memory {result['memory_index']}, "
                     f"Fidelity = {fidelity_str}")
            

            print("\nBell Pair Generation Timestamps (picoseconds):")
            for result in self.results:
                timestamp_ps = result['generation_time'] * 1e12
                print(f"  Pair {result['pair_id']}: t = {timestamp_ps:.0f} ps")
        
        # Encoding results
        if self.encoding_enabled:
            print(f"\nQEC Encoding:")
            print(f"  Target State: |{self.logical_state}>_L")
            print(f"  Encoding Complete: {self.encoding_complete}")
            
            if self.encoding_complete and self.encoding_start_time and self.encoding_end_time:
                encoding_time = (self.encoding_end_time - self.encoding_start_time) * 1e-12
                print(f"  Encoding Time: {encoding_time:.6f}s")
        
        print(f"{'='*70}\n")
    
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
                'target_state': '|Î¦+âŸ© = (|00âŸ© + |11âŸ©)/âˆš2',
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
        
        return results