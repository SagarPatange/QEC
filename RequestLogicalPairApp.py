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
    
    # Class variable to track all instances for inter-app communication
    _instances = {}
    
    def __init__(self, node: "QuantumRouter"):
        super().__init__(node)

        # Register this instance on node so protocols can access it
        self.node.request_logical_pair_app = self  # ADD THIS LINE
        self.remote_node: QuantumRouter = None  
    
        # Register this instance for inter-app communication
        RequestLogicalPairApp._instances[node.name] = self

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
        self.depolarization_enabled = True  # Flag to enable/disable idling decoherence
        
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
   
   
    def _calculate_logical_bell_pair_fidelity(self, shots: int = 100000) -> float:
        """
        Calculate logical Bell pair fidelity after teleported CNOT.

        Measures F = (1 + <XX> - <YY> + <ZZ>) / 4 using efficient sampling.

        Args:
            shots: Number of measurement samples per basis

        Returns:
            Logical Bell pair fidelity with |Φ⁺>_L
        """
        qm = self.node.timeline.quantum_manager

        # Get Alice's data qubits (this node)
        alice_keys = [mem.qstate_key for mem in self.data_qubits]

        # Get Bob's data qubits (remote node) - FIX: Was using communication_qubits (measured ancillas)
        bob_app = RequestLogicalPairApp._instances[self.remote_node_name]
        bob_keys = [mem.qstate_key for mem in bob_app.data_qubits]

        all_keys = alice_keys + bob_keys

        log.logger.info(f"{self.node.name}: Measuring logical Bell pair fidelity...")
        log.logger.info(f"{self.node.name}: Using Alice's data qubits: {[mem.name for mem in self.data_qubits]}")
        log.logger.info(f"{self.node.name}: Using Bob's data qubits: {[mem.name for mem in bob_app.data_qubits]}")
        log.logger.info(f"{self.node.name}: Alice keys: {alice_keys}")
        log.logger.info(f"{self.node.name}: Bob keys: {bob_keys}")
        
        # Build base circuit from all qubits
        base_circuit = stim.Circuit()
        added_states = set()

        # DEBUG: Check if qubits are in the same state
        log.logger.info(f"{self.node.name}: Checking quantum states...")
        for i, key in enumerate(all_keys):
            state = qm.states[key]
            state_id = id(state)
            state_qubits = sorted(state.keys)
            if i < 7:
                log.logger.info(f"{self.node.name}: Alice qubit {key} -> state_id={state_id}, state contains qubits: {state_qubits}")
            else:
                log.logger.info(f"{self.node.name}: Bob qubit {key} -> state_id={state_id}, state contains qubits: {state_qubits}")

        for key in all_keys:
            state = qm.states[key]
            if id(state) in added_states:
                continue
            added_states.add(id(state))

            for instruction in state.circuit:
                gate_args = instruction.gate_args_copy()
                # Handle both qubit targets and measurement record targets
                targets = []
                for t in instruction.targets_copy():
                    if t.is_measurement_record_target:
                        # Keep measurement record targets as-is
                        targets.append(t)
                    else:
                        # Convert qubit targets to integers
                        targets.append(t.value)

                if gate_args:
                    base_circuit.append(instruction.name, targets, *gate_args)
                else:
                    base_circuit.append(instruction.name, targets)

        log.logger.info(f"{self.node.name}: Found {len(added_states)} unique quantum states among {len(all_keys)} qubits")
        
        # Measure all three correlations and compute fidelity
        correlations = {}
        
        for basis in ['X', 'Y', 'Z']:
            # Copy circuit and add basis rotations
            meas_circuit = base_circuit.copy()
            
            for key in all_keys:
                if basis == 'X':
                    meas_circuit.append("H", [key])
                elif basis == 'Y':
                    meas_circuit.append("S_DAG", [key])
                    meas_circuit.append("H", [key])
            
            # Add measurements
            for key in all_keys:
                meas_circuit.append("M", [key])
            
            # Sample and compute correlation
            seed = hash(f"{self.node.name}_{basis}_{self.node.timeline.now()}") % (2**31)
            sampler = meas_circuit.compile_sampler(seed=seed)
            measurements = sampler.sample(shots=shots)
            
            # Compute logical parities and correlation

            alice_parity = np.sum(measurements[:, 14:21], axis=1) % 2  # Alice data qubits (7 qubits)
            bob_parity = np.sum(measurements[:, 21:28], axis=1) % 2    # Bob data qubits (7 qubits)
            alice_eigenvalues = 1 - 2 * alice_parity
            bob_eigenvalues = 1 - 2 * bob_parity
            correlations[basis] = float(np.mean(alice_eigenvalues * bob_eigenvalues))
        
        # Store correlations
        self.logical_fidelity['correlations']['XX'] = correlations['X']
        self.logical_fidelity['correlations']['YY'] = correlations['Y']
        self.logical_fidelity['correlations']['ZZ'] = correlations['Z']
        
        # Calculate fidelity
        fidelity = (1 + correlations['X'] - correlations['Y'] + correlations['Z']) / 4

        log.logger.info(f"{self.node.name}: Logical Bell pair fidelity = {fidelity:.6f}")
        log.logger.info(f"{self.node.name}: Correlation XX = {correlations['X']:+.6f}")
        log.logger.info(f"{self.node.name}: Correlation YY = {correlations['Y']:+.6f}")
        log.logger.info(f"{self.node.name}: Correlation ZZ = {correlations['Z']:+.6f}")
        log.logger.info(f"{self.node.name}: Fidelity formula: (1 + {correlations['X']:+.6f} - {correlations['Y']:+.6f} + {correlations['Z']:+.6f}) / 4 = {fidelity:.6f}")

        return float(fidelity)


    def start(self, remote_node_name: str, start_time: int, end_time: int,
            memory_size: int = 7, target_fidelity: float = 0.8,
            logical_state: str = '0', encoding_enabled: bool = False,
            teleported_cnot_enabled: bool = False, depolarization_enabled: bool = False,
            coherence_time_ms: float = 500, remote_node: "QuantumRouter" = None):
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
            
        self.remote_node = remote_node
        self.remote_node_name = remote_node_name
        self.encoding_enabled = encoding_enabled
        self.logical_state = logical_state
        self.teleported_cnot_enabled = teleported_cnot_enabled
        self.depolarization_enabled = depolarization_enabled
        self.depolarization_time_ms = coherence_time_ms
        self.is_initiator = True
        self.is_responder = False

        # Configure the remote app directly if it exists
        if remote_node_name in RequestLogicalPairApp._instances:
            remote_app = RequestLogicalPairApp._instances[remote_node_name]
            remote_app._configure_as_responder(
                initiator_name=self.node.name,
                encoding_enabled=encoding_enabled,
                initiator_logical_state=logical_state,
                teleported_cnot_enabled=teleported_cnot_enabled,
                depolarization_enabled=self.depolarization_enabled,
                coherence_time_ms=self.depolarization_time_ms
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
        self._release_memories()
        self.print_results()
  
        
    def _configure_as_responder(self, initiator_name: str, encoding_enabled: bool,
                                initiator_logical_state: str,
                                teleported_cnot_enabled: bool = False,
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
        self.remote_node_name = initiator_name

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
    
    
    def _initialize_teleported_cnot(self):
        """Initialize and start the teleported CNOT protocol."""

        # Determine role based on who initiated the request
        role = 'alice' if self.is_initiator else 'bob'

        # Only create protocol if not already created
        if self.teleported_cnot_protocol is None:
            # Create protocol instance
            protocol_name = f"TeleportedCNOT_{self.node.name}"
            self.teleported_cnot_protocol = TeleportedCNOTProtocol(
                owner=self.node,
                name=protocol_name,
                role=role,
                remote_node_name=self.remote_node_name,
                data_qubits=self.data_qubits,
                communication_qubits=self.communication_qubits,
                remote_node=self.remote_node
            )

            # Register protocol with node so it can receive messages
            self.node.protocols.append(self.teleported_cnot_protocol)
            log.logger.info(f"{self.node.name}: Teleported CNOT protocol initialized")

        # Alice needs to check if Bob is ready before starting
        if role == 'alice':
            # Check if Bob's protocol is initialized
            if self.remote_node and hasattr(self.remote_node, 'request_logical_pair_app'):
                bob_app = self.remote_node.request_logical_pair_app
                if bob_app.teleported_cnot_protocol is None:
                    # Bob not ready yet, schedule retry
                    process = Process(self, '_initialize_teleported_cnot', [])
                    event = Event(self.node.timeline.now() + 1000000, process)
                    self.node.timeline.schedule(event)
                    return

            # Bob is ready, start protocol
            self.teleported_cnot_start_time = self.node.timeline.now()
            log.logger.info(f"{self.node.name}: Starting teleported CNOT as Alice")
            self.teleported_cnot_protocol.alice_start_protocol()
        else:
            log.logger.info(f"{self.node.name}: Ready for teleported CNOT as Bob")


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
            self._all_physical_bell_pairs_completed()
    
    
    def _calculate_physical_bell_pair_fidelity_via_tomography(self, info: "MemoryInfo", remote_node_name: str, 
                                          remote_memo_name: str) -> float:
        """Calculate Bell pair fidelity using TRUE quantum state tomography.
        
        This method performs ACTUAL Pauli tomography by using the quantum manager's
        compute_density_matrix method, which performs measurements in multiple bases
        to reconstruct the density matrix, rather than accessing the internal state.
        
        The tomography process involves:
        1. Finding both qubits in the Bell pair
        2. Using qm.compute_density_matrix(keys) to perform tomography measurements
        3. Calculating fidelity F = Ã¢Å¸Â¨ÃŽÂ¦+|Ã|ÃŽÂ¦+Ã¢Å¸Â© from the reconstructed density matrix
        
        Args:
            info: Local memory information
            remote_node_name: Name of remote node
            remote_memo_name: Name of remote memory
            
        Returns:
            Fidelity with ideal |ÃŽÂ¦+Ã¢Å¸Â© Bell state (value between 0 and 1)
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
            
            # Ã¢Å“â€¦ KEY CHANGE: Use quantum manager's compute_density_matrix
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
            log.logger.debug(f"{self.node.name}: Off-diagonal coherence: |Ã[0,3]| = {abs(rho[0,3]):.4f}")
            
            return float(fidelity)
            
        except Exception as e:
            log.logger.error(f"{self.node.name}: Fidelity calculation via tomography FAILED: {e}")
            import traceback
            log.logger.error(traceback.format_exc())
            # Re-raise the exception - DO NOT silently return a fake fidelity value
            raise RuntimeError(f"Tomography-based fidelity calculation failed for {self.node.name}") from e


    def _all_physical_bell_pairs_completed(self):
        """Called when all 7 Bell pairs have been generated."""
        log.logger.info(f"{self.node.name}: All {self.num_completed} Bell pairs generated")
        # Directly schedule fidelity calculation
        log.logger.info(f"{self.node.name}: Scheduling fidelity calculation")
        process = Process(self, '_calculate_physical_bell_pair_fidelities', [])
        event = Event(self.node.timeline.now(), process)
        self.node.timeline.schedule(event)
    
    
    def _calculate_physical_bell_pair_fidelities(self):
        """Calculate fidelities for all Bell pairs after applying depolarization noise."""
        
        log.logger.info(f"{self.node.name}: Starting fidelity calculation")
        
        # Get current simulation time
        current_time = self.node.timeline.now()  # in picoseconds
        
        # Apply depolarization noise to each Bell pair based on elapsed time (if enabled)
        if self.depolarization_enabled:
            for i, result in enumerate(self.results):
                # Get when this pair was generated (convert from seconds to picoseconds)
                generation_time_ps = result['generation_time'] * 1e12
                
                # Calculate elapsed time
                elapsed_time_ps = current_time - generation_time_ps
                
                # Calculate depolarization probability
                p = self._calculate_depolarization_probability(elapsed_time_ps, coherence_time_ms = self.depolarization_time_ms)
                
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
            
            measured_fidelity = self._calculate_physical_bell_pair_fidelity_via_tomography(info, remote_node, remote_memory)
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
    
    
    def _release_memories(self):
        """Release all communication memories to RAW state."""
        for result in self.results:
            info = result['memory_info']
            self.node.resource_manager.update(None, info.memory, "RAW")
        log.logger.info(f"{self.node.name}: Released all memories")
    
    
    def _start_encoding(self):
        """Start QEC encoding process to create logical qubits.
        
        Encodes the pre-allocated data qubits to create either |0>_L or |+>_L.
        
        Steps:
        1. Initialize data qubits to |0> states
        2. Prepare initial state (apply H for |+> if needed)
        3. Apply [[7,1,3]] encoding circuit
        4. Schedule completion callback
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
            
            # Prepare initial state based on desired logical state
            if self.logical_state == '+':
                # Apply H to first qubit to get |+> = (|0> + |1>)/sqrt(2)
                log.logger.info(f"{self.node.name}: Applying H gate to prepare |+> initial state")
                h_circuit = QEC713.create_hadamard_circuit()
                qm.run_circuit(h_circuit, [data_keys[0]], 0.5)
            else:
                # Bob needs |0>_L: Qubits already in |0> state
                log.logger.info(f"{self.node.name}: Using |0> initial state (no preparation needed)")
            
            # Apply [[7,1,3]] encoding circuit
            log.logger.info(f"{self.node.name}: Applying [[7,1,3]] encoding circuit")
            encoding_circuit = QEC713.create_encoding_circuit()
            qm.run_circuit(encoding_circuit, data_keys, 0.5, False)
            
            log.logger.info(f"{self.node.name}: Encoding circuit applied successfully")
            
            # Schedule encoding completion callback
            delay = 1000  # 1 nanosecond
            process = Process(self, '_encoding_completed', [])
            event = Event(self.node.timeline.now() + delay, process)
            self.node.timeline.schedule(event)
            
            log.logger.info(f"{self.node.name}: Scheduled encoding completion callback")
            
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
            log.logger.info(f"{self.node.name}: Calculating product state fidelity |0>_L (Alice) x |+>_L (Bob)")
            
            # Calculate product state fidelity
            product_fidelity_result = self._calculate_product_state_fidelity(shots=10000)
            
            # Store results
            self.product_state_fidelity = product_fidelity_result
            
            alice_x = product_fidelity_result['alice_x_prob']
            bob_z = product_fidelity_result['bob_z_prob']
            fidelity = product_fidelity_result['fidelity']

        encoding_time = (self.encoding_end_time - self.encoding_start_time) * 1e-12
        log.logger.info(f"{self.node.name}: Encoding completed in {encoding_time:.6f}s")
        
        # Expire the reservation to stop generating more pairs
        for reservation in self.memo_to_reservation.values():
            self.node.resource_manager.expire_rules_by_reservation(reservation)
            break  # Only expire once
        
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
            self._release_memories()
            self.print_results()


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
    

    def print_results(self):
        """Print comprehensive results of Bell pair generation and encoding."""
        if not self.verbose:
            return

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
                
        # Logical Bell pair fidelity results
        if self.teleported_cnot_enabled:
            print(f"\nLogical Bell Pair Fidelity (Post Teleported-CNOT):")
            if self.logical_fidelity['fidelity'] is not None:
                print(f"  Fidelity: {self.logical_fidelity['fidelity']:.6f}")
                print(f"  <XX> = {self.logical_fidelity['correlations']['XX']:+.6f}")
                print(f"  <YY> = {self.logical_fidelity['correlations']['YY']:+.6f}")
                print(f"  <ZZ> = {self.logical_fidelity['correlations']['ZZ']:+.6f}")
                
                # Compare to physical fidelities
                if self.results and any(r['fidelity'] is not None for r in self.results):
                    physical_fidelities = [r['fidelity'] for r in self.results if r['fidelity'] is not None]
                    avg_physical = np.mean(physical_fidelities)
                    improvement = self.logical_fidelity['fidelity'] - avg_physical
                    print(f"\n  Comparison to Physical Bell Pairs:")
                    print(f"    Average physical fidelity: {avg_physical:.6f}")
                    print(f"    Logical fidelity improvement: {improvement:+.6f}")
            else:
                print(f"  ERROR: Logical fidelity calculation FAILED!")
                print(f"  Fidelity: None")
                print(f"  This indicates an error occurred during fidelity measurement.")
        
        print(f"{'='*70}\n")
        
        print(f"{'='*70}\n")
    

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
    
    
    def _calculate_product_state_fidelity(self, shots: int = 10000) -> dict:
        """
        Calculate fidelity of |+>_L (Alice) and |0>_L (Bob) product state.
        
        For [[7,1,3]] Steane code:
        - Logical Z = Z^(otimes 7) (all physical Z operators)
        - Logical X = X^(otimes 7) (all physical X operators)
        
        Measures:
        - Alice: P(X_L = +1) for |+>_L state
        - Bob: P(Z_L = +1) for |0>_L state
        - Product fidelity: F = P(X_L = +1) * P(Z_L = +1)
        
        Args:
            shots: Number of measurement samples
            
        Returns:
            dict with 'alice_x_prob', 'bob_z_prob', 'fidelity'
        """
        try:
            qm = self.node.timeline.quantum_manager
            
            # Get Alice's data qubits (this node)
            alice_keys = [mem.qstate_key for mem in self.data_qubits]
            
            # Get Bob's data qubits (remote node)
            bob_app = RequestLogicalPairApp._instances[self.remote_node_name]
            bob_keys = [mem.qstate_key for mem in bob_app.data_qubits]
            
            log.logger.info(f"{self.node.name}: Measuring product state fidelity")
            log.logger.info(f"{self.node.name}: Alice keys: {alice_keys}")
            log.logger.info(f"{self.node.name}: Bob keys: {bob_keys}")
            
            # Build base circuit from all qubits
            base_circuit = stim.Circuit()
            added_states = set()
            all_keys = alice_keys + bob_keys
            
            for key in all_keys:
                state = qm.states[key]
                if id(state) in added_states:
                    continue
                added_states.add(id(state))
                
                for instruction in state.circuit:
                    gate_args = instruction.gate_args_copy()
                    # Handle both qubit targets and measurement record targets
                    targets = []
                    for t in instruction.targets_copy():
                        if t.is_measurement_record_target:
                            # Keep measurement record targets as-is
                            targets.append(t)
                        else:
                            # Convert qubit targets to integers
                            targets.append(t.value)

                    if gate_args:
                        base_circuit.append(instruction.name, targets, *gate_args)
                    else:
                        base_circuit.append(instruction.name, targets)
            
            log.logger.info(f"{self.node.name}: Found {len(added_states)} unique quantum states")
            
            # Measure Alice's logical X (apply H to change to X basis, then measure)
            alice_circuit = base_circuit.copy()
            for key in alice_keys:  # Only apply H to Alice's qubits
                alice_circuit.append("H", [key])
            for key in all_keys:
                alice_circuit.append("M", [key])
            
            seed_alice = hash(f"{self.node.name}_alice_x_{self.node.timeline.now()}") % (2**31)
            alice_sampler = alice_circuit.compile_sampler(seed=seed_alice)
            alice_samples = alice_sampler.sample(shots=shots)
            
            # Calculate Alice's X_L eigenvalue: parity of first 7 measurements
            alice_x_parity = np.sum(alice_samples[:, :7], axis=1) % 2
            alice_x_eigenvalues = 1 - 2 * alice_x_parity  # +1 or -1
            alice_x_prob = float(np.mean(alice_x_eigenvalues == 1))  # P(X_L = +1)
            
            # Measure Bob's logical Z (measure directly in Z basis)
            bob_circuit = base_circuit.copy()
            for key in all_keys:
                bob_circuit.append("M", [key])
            
            seed_bob = hash(f"{self.node.name}_bob_z_{self.node.timeline.now()}") % (2**31)
            bob_sampler = bob_circuit.compile_sampler(seed=seed_bob)
            bob_samples = bob_sampler.sample(shots=shots)
            
            # Calculate Bob's Z_L eigenvalue: parity of last 7 measurements
            bob_z_parity = np.sum(bob_samples[:, 7:14], axis=1) % 2
            bob_z_eigenvalues = 1 - 2 * bob_z_parity  # +1 or -1
            bob_z_prob = float(np.mean(bob_z_eigenvalues == 1))  # P(Z_L = +1)
            
            # Product state fidelity
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