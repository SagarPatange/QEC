"""
=====================================================================
stim_node.py
---------------------------------------------------------------------
Minimal Stim-based node that coordinates all managers and components
for a two-node entanglement demo compatible with SeQUeNCe patterns.
=====================================================================
"""

from __future__ import annotations
from typing import Dict, Optional, Any
import logging

from sequence.kernel.entity import Entity
from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event

# Import all our Stim components
from stim_memory import StimMemoryArray, StimMemory, LinkLedger
from stim_memory_manager import StimMemoryManager, StimMemoryInfo, MemState
from stim_rule_manager import StimRuleManager
from stim_resource_manager import StimResourceManager
from stim_network_manager import StimNetworkManager
from stim_quantum_state import StimQuantumState
from stim_quantum_state_manager import StimQuantumStateManager

# Simple node directory for peer lookup
_node_directory: Dict[str, "StimNode"] = {}


class StimNode(Entity):
    """
    Minimal Stim-based quantum network node.
    
    Instantiates and wires together:
    - StimMemoryArray (keys/slots)
    - LinkLedger (entanglement records)
    - StimMemoryManager (state tracking)
    - StimRuleManager (rule evaluation)
    - StimResourceManager (coordinator)
    - StimNetworkManager (reservation windows)
    - StimQuantumStateManager (EG/EP/ES controller)
    - StimQuantumState (stabilizer circuit)
    """
    
    def __init__(
        self,
        name: str,
        timeline: Timeline,
        num_memories: int = 10,
        seed: Optional[int] = None
    ):
        super().__init__(name, timeline)
        self.seed = seed
        self.num_memories = num_memories
        
        # Register in directory
        _node_directory[name] = self
        
        # Create memory array with default parameters
        self.memory_array = StimMemoryArray(
            f"{name}.MemoryArray",
            timeline,
            num_memories=num_memories,
            frequency=1e6,
            efficiency=0.9,
            coherence_time=1e9,  # 1ms in ps
            wavelength=1550e-9,
            decoherence_errors={"px": 0.001, "pz": 0.002},
            cutoff_ratio=0.8
        )
        
        # Create link ledger
        self.link_ledger = LinkLedger(f"{name}.LinkLedger", timeline)
        
        # Create memory manager
        self.memory_manager = StimMemoryManager(self.memory_array, self.link_ledger)
        
        # Create rule manager
        self.rule_manager = StimRuleManager(self.memory_manager)
        
        # Create resource manager (glue layer)
        self.resource_manager = StimResourceManager(
            self, self.memory_manager, self.rule_manager
        )
        
        # Create network manager (reservation windows)
        self.network_manager = StimNetworkManager(
            f"{name}.NetworkManager",
            timeline,
            self.resource_manager
        )
        
        # Create quantum state (stabilizer circuit)
        self.quantum_state = StimQuantumState(num_qubits=num_memories * 2)  # Extra qubits for protocols
        
        # Create quantum state manager (protocol controller)
        self.quantum_state_manager = StimQuantumStateManager(
            f"{name}.QSManager",
            timeline,
            self.quantum_state,
            self.link_ledger,
            self.resource_manager
        )
        
        # Application placeholder
        self.app = None
        
        # Initialize components
        self.memory_array.init(self)
        
        logging.info(f"Created StimNode {name} with {num_memories} memories")
    
    def init(self) -> None:
        """Initialize the node (Entity abstract method implementation)."""
        pass
    
    def memory_expire(self, memory: StimMemory) -> None:
        """Handle memory expiration (TTL timeout)."""
        self.resource_manager.memory_expire(memory)
    
    def get_peer(self, name: str) -> Optional["StimNode"]:
        """Look up peer node by name."""
        return _node_directory.get(name)
    
    def set_app(self, app: Any) -> None:
        """Attach an application to this node."""
        self.app = app
        if hasattr(app, "set_node"):
            app.set_node(self)
    
    # --- Application callbacks (Chapter 5/6 compatibility) ---
    
    def get_reservation_result(self, reservation: Any, result: bool) -> None:
        """
        Callback when reservation is approved/denied.
        Forward to application if present.
        """
        if self.app and hasattr(self.app, "get_reservation_result"):
            self.app.get_reservation_result(reservation, result)
        else:
            logging.info(f"{self.name}: Reservation {'approved' if result else 'denied'}")
    
    def get_memory(self, info: StimMemoryInfo) -> None:
        """
        Callback when a memory becomes available/entangled.
        Forward to application if present.
        """
        if self.app and hasattr(self.app, "get_memory"):
            self.app.get_memory(info)
        else:
            if info.state == MemState.ENTANGLED:
                logging.info(
                    f"{self.name}: Memory {info.index} ENTANGLED with "
                    f"{info.remote_node}:{info.remote_key} at time {info.entangle_time_ps}"
                )
    
    # --- Convenience methods for testing ---
    
    def request_entanglement(
        self,
        peer_name: str,
        start_time: int,
        end_time: int,
        memory_size: int = 1,
        target_fidelity: float = 1.0,
        rules_factory: Optional[Any] = None
    ) -> None:
        """
        Request entanglement with a peer node.
        Uses the network manager to schedule a reservation window.
        """
        self.network_manager.request(
            responder=peer_name,
            start_time_ps=start_time,
            end_time_ps=end_time,
            memory_size=memory_size,
            target_fidelity=target_fidelity,
            rules_factory=rules_factory
        )
    
    def update_memory_params(self, field: str, value: Any) -> None:
        """Update a parameter on all memories (Chapter 5 compatibility)."""
        self.memory_array.update_memory_params(field, value)
    
    def get_protocol_stack(self) -> list:
        """Return protocol stack for compatibility."""
        # Index 1 should have swapping tunables
        return [None, self.network_manager.protocol_stack[1]]