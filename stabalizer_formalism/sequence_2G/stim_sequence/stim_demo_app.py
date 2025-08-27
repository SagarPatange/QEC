"""
=====================================================================
stim_demo_app.py
---------------------------------------------------------------------
Minimal two-node demo that requests entanglement and demonstrates
the full Stim-based architecture in action.
=====================================================================
"""

from __future__ import annotations
import logging
from typing import Optional, List, Any

from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event

from stim_node import StimNode
from stim_memory_manager import StimMemoryInfo, MemState
from stim_rule_manager import StimRuleManager
from stim_resource_manager import StimResourceManager


# --- Simple Rules for Demo ---

class ClaimMemoryRule:
    """Rule to claim a RAW memory and move it to OCCUPIED."""
    
    def __init__(self, priority: int = 10):
        self.priority = priority
        self.name = "ClaimMemoryRule"
    
    def is_valid(self, info: StimMemoryInfo) -> bool:
        return info.state == MemState.RAW
    
    def do(self, rm: StimResourceManager, selected: List[StimMemoryInfo]) -> None:
        if selected:
            # Claim first available memory
            info = selected[0]
            rm.mm.to_occupied(info)
            logging.info(f"Rule claimed memory {info.index} -> OCCUPIED")


class EntangleRule:
    """Rule to attempt entanglement generation on OCCUPIED memories."""
    
    def __init__(self, target_node: str, priority: int = 5):
        self.target_node = target_node
        self.priority = priority
        self.name = f"EntangleRule->{target_node}"
    
    def is_valid(self, info: StimMemoryInfo) -> bool:
        return info.state == MemState.OCCUPIED
    
    def do(self, rm: StimResourceManager, selected: List[StimMemoryInfo]) -> None:
        for info in selected:
            # Get the quantum state manager from the node
            node = rm.node
            qsm = node.quantum_state_manager
            
            # Attempt entanglement generation
            link_id = qsm.entanglement_generation(info.memory, self.target_node, herald_success=True)
            if link_id:
                logging.info(f"Rule initiated EG for memory {info.index} -> {self.target_node}, link: {link_id}")
            else:
                # Failed, return to RAW
                rm.mm.to_raw(info)
                logging.info(f"Rule EG failed for memory {info.index} -> {self.target_node}")


def rules_factory(rm: StimResourceManager, target_fidelity: float) -> List[Any]:
    """
    Factory function to create rules for a reservation window.
    
    Args:
        rm: Resource manager
        target_fidelity: Target fidelity (ignored for MVP)
        
    Returns:
        List of rule instances
    """
    # For demo, just create simple rules
    # In real system, would create more sophisticated rules based on target
    peer = "Bob" if rm.node.name == "Alice" else "Alice"
    return [
        ClaimMemoryRule(priority=10),
        EntangleRule(target_node=peer, priority=5)
    ]


# --- Application Classes (Chapter 6 style) ---

class PeriodicApp:
    """Application that periodically requests entanglement."""
    
    def __init__(
        self,
        node: StimNode,
        other: str,
        memory_size: int = 1,
        target_fidelity: float = 0.9
    ):
        self.node = node
        self.other = other
        self.memory_size = memory_size
        self.target_fidelity = target_fidelity
        self.period = 1e12  # 1 second in picoseconds
        
        # Register with node
        node.set_app(self)
    
    def start(self):
        """Start periodic entanglement requests."""
        now = self.node.timeline.now()
        
        # Request entanglement for next 0.5 seconds
        self.node.request_entanglement(
            peer_name=self.other,
            start_time=int(now + 0.1e12),  # Start 0.1s from now
            end_time=int(now + 0.6e12),    # End 0.6s from now
            memory_size=self.memory_size,
            target_fidelity=self.target_fidelity,
            rules_factory=rules_factory
        )
        
        # Schedule next request
        process = Process(self, "start", [])
        event = Event(int(now + self.period), process)
        self.node.timeline.schedule(event)
        
        logging.info(f"{self.node.name} scheduled entanglement request at {now * 1e-12}s")
    
    def get_reservation_result(self, reservation: Any, result: bool):
        """Handle reservation approval/denial."""
        time_s = self.node.timeline.now() * 1e-12
        if result:
            print(f"Reservation approved at time {time_s:.3f}s")
        else:
            print(f"Reservation failed at time {time_s:.3f}s")
    
    def get_memory(self, info: StimMemoryInfo):
        """Handle entangled memory notification."""
        if info.state == MemState.ENTANGLED and info.remote_node == self.other:
            time_s = self.node.timeline.now() * 1e-12
            print(f"\t{self.node.name} app received memory {info.index} "
                  f"ENTANGLED at time {time_s:.3f}s")
            
            # Immediately free the memory for next use (don't trigger rules)
            self.node.memory_manager.to_raw(info)


class ResetApp:
    """Application on receiving node that resets entangled memories."""
    
    def __init__(
        self,
        node: StimNode,
        other_node_name: str,
        target_fidelity: float = 0.9
    ):
        self.node = node
        self.other_node_name = other_node_name
        self.target_fidelity = target_fidelity
        
        # Register with node
        node.set_app(self)
    
    def get_reservation_result(self, reservation: Any, result: bool):
        """We're the responder, so we don't initiate reservations."""
        pass
    
    def get_memory(self, info: StimMemoryInfo):
        """Reset entangled memories back to RAW."""
        if info.state == MemState.ENTANGLED and info.remote_node == self.other_node_name:
            time_s = self.node.timeline.now() * 1e-12
            print(f"\t{self.node.name} reset memory {info.index} at time {time_s:.3f}s")
            
            # Free the memory (don't trigger rules)
            self.node.memory_manager.to_raw(info)


# --- Main Demo ---

def main():
    """
    Run a two-node entanglement demo.
    Alice periodically requests entanglement with Bob.
    """
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create timeline
    from sequence.kernel.timeline import Timeline
    
    NUM_PERIODS = 3  
    PERIOD = 1e12  # 1 second in picoseconds
    
    tl = Timeline(stop_time=PERIOD * NUM_PERIODS)
    tl.show_progress = True
    
    # Create two nodes
    alice = StimNode("Alice", tl, num_memories=10)
    bob = StimNode("Bob", tl, num_memories=10)
    
    # Update memory parameters for testing
    alice.update_memory_params("coherence_time", 1e9)  # 1ms
    alice.update_memory_params("frequency", 1e6)  # 1MHz
    bob.update_memory_params("coherence_time", 1e9)
    bob.update_memory_params("frequency", 1e6)
    
    # Create applications
    alice_app = PeriodicApp(alice, "Bob", memory_size=1, target_fidelity=0.9)
    bob_app = ResetApp(bob, "Alice", target_fidelity=0.9)
    
    # Bob will get rules during reservation windows only
    
    # Initialize timeline and start
    print("="*60)
    print("Starting Two-Node Stim Entanglement Demo")
    print("="*60)
    print(f"Running for {NUM_PERIODS} periods of {PERIOD * 1e-12}s each")
    print("="*60)
    
    tl.init()
    alice_app.start()
    tl.run()
    
    print("="*60)
    print("Demo Complete!")
    print("="*60)
    
    # Print final statistics
    alice_links = alice.link_ledger.list_links(live_only=False)
    bob_links = bob.link_ledger.list_links(live_only=False)
    
    print(f"\nAlice created {len(alice_links)} links")
    print(f"Bob created {len(bob_links)} links")
    
    # Count successful entanglements
    entangled = sum(1 for l in alice_links if l.status.name == "ENTANGLED")
    consumed = sum(1 for l in alice_links if l.status.name == "CONSUMED")
    failed = sum(1 for l in alice_links if l.status.name == "FAILED")
    
    print(f"\nLink statistics:")
    print(f"  Entangled (active): {entangled}")
    print(f"  Consumed (used): {consumed}")
    print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()