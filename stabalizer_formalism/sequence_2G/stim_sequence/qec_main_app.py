"""
=====================================================================
qec_main_application.py
---------------------------------------------------------------------
Main application for [[7,1,3]] QEC purification with proper timeline
integration and classical communication channels.
=====================================================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging

from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event
from sequence.components.optical_channel import ClassicalChannel
from sequence.constants import MILLISECOND

from stim_node import StimNode
from qec_protocol_stack import QECProtocol, pair_qec_protocols


class QECManager:
    """
    Upper layer manager for QEC protocol, similar to KeyManager in BB84.
    Collects results and provides interface to the protocol stack.
    """
    
    def __init__(self, timeline: Timeline, target_fidelity: float, num_rounds: int):
        """
        Initialize QEC manager.
        
        Args:
            timeline: Simulation timeline
            target_fidelity: Target fidelity for purification
            num_rounds: Number of purification rounds
        """
        self.timeline = timeline
        self.lower_protocols: List[QECProtocol] = []
        self.target_fidelity = target_fidelity
        self.num_rounds = num_rounds
        
        # Results storage
        self.results: List[Dict] = []
        self.completion_times: List[float] = []
        
    def send_request(self, error_prob: float = 0.1) -> None:
        """Send purification request to lower protocol."""
        for protocol in self.lower_protocols:
            protocol.push(
                rounds=self.num_rounds,
                target_fidelity=self.target_fidelity,
                error_prob=error_prob
            )
    
    def pop(self, info: Dict) -> None:
        """
        Receive results from lower protocol.
        
        Args:
            info: Dictionary with purification results
        """
        self.results.append(info)
        self.completion_times.append(self.timeline.now() / MILLISECOND)
        
        # Log results
        if 'avg_fidelity' in info:
            logging.info(f"Purification complete at {self.completion_times[-1]:.2f} ms")
            logging.info(f"  Average fidelity: {info['avg_fidelity']:.6f}")
            if 'total_corrections' in info:
                logging.info(f"  Total corrections: {info['total_corrections']}")


class QECNode(StimNode):
    """
    Extended StimNode with QEC protocol stack support.
    """
    
    def __init__(self, name: str, timeline: Timeline, 
                 stack_size: int = 1, num_memories: int = 80):
        """
        Initialize QEC-enabled node.
        
        Args:
            name: Node name
            timeline: Simulation timeline
            stack_size: Number of protocols in stack
            num_memories: Number of quantum memories (80 for full protocol)
        """
        super().__init__(name, timeline, num_memories)
        
        # Protocol stack
        self.protocol_stack: List[Optional[QECProtocol]] = [None] * stack_size
        
        # Add QEC protocol to stack
        if stack_size >= 1:
            self.protocol_stack[0] = QECProtocol(self, f"{name}.QEC")
        
        # Classical channels storage
        self.cchannels: Dict[str, ClassicalChannel] = {}
    
    def send_message(self, dst: str, msg) -> None:
        """Send classical message to destination node."""
        if dst in self.cchannels:
            delay = self.cchannels[dst].delay
            # Schedule message delivery
            process = Process(self, "_deliver_message", [dst, msg])
            event = Event(self.timeline.now() + delay, process)
            self.timeline.schedule(event)
        else:
            logging.warning(f"No channel to {dst}")
    
    def _deliver_message(self, dst: str, msg) -> None:
        """Internal method to deliver message."""
        # Find destination node
        from stim_node import _node_directory
        if dst in _node_directory:
            dst_node = _node_directory[dst]
            if hasattr(dst_node, 'protocol_stack'):
                for protocol in dst_node.protocol_stack:
                    if protocol and protocol.name == msg.receiver:
                        protocol.received_message(self.name, msg)
                        break


def run_qec_simulation(
    sim_time: int = 100,  # milliseconds
    num_rounds: int = 10,
    target_fidelity: float = 0.99,
    error_prob: float = 0.1,
    channel_distance: float = 1e3,  # meters
    show_plots: bool = True
) -> Dict:
    """
    Run QEC purification simulation.
    
    Args:
        sim_time: Simulation time in milliseconds
        num_rounds: Number of purification rounds
        target_fidelity: Target fidelity
        error_prob: Error probability per round
        channel_distance: Distance between nodes in meters
        show_plots: Whether to show result plots
        
    Returns:
        Dictionary with simulation results
    """
    print("="*70)
    print("[[7,1,3]] QEC Entanglement Purification Simulation")
    print("="*70)
    print(f"Parameters:")
    print(f"  Simulation time: {sim_time} ms")
    print(f"  Purification rounds: {num_rounds}")
    print(f"  Target fidelity: {target_fidelity}")
    print(f"  Error probability: {error_prob}")
    print(f"  Channel distance: {channel_distance/1e3:.1f} km")
    print("-"*70)
    
    # Create timeline
    tl = Timeline(sim_time * 1e9)  # Convert ms to ps
    tl.show_progress = False
    
    # Create QEC nodes
    alice = QECNode("Alice", tl, stack_size=1, num_memories=80)
    bob = QECNode("Bob", tl, stack_size=1, num_memories=80)
    
    # Set random seeds
    alice.seed = 0
    bob.seed = 1
    
    # Pair protocols
    pair_qec_protocols(alice.protocol_stack[0], bob.protocol_stack[0])
    
    # Create classical channels (bidirectional)
    cc_alice_bob = ClassicalChannel("cc_alice_bob", tl, distance=channel_distance)
    cc_bob_alice = ClassicalChannel("cc_bob_alice", tl, distance=channel_distance)
    
    # Set channel endpoints
    cc_alice_bob.set_ends(alice, bob.name)
    cc_bob_alice.set_ends(bob, alice.name)
    
    # Store channels in nodes
    alice.cchannels[bob.name] = cc_alice_bob
    bob.cchannels[alice.name] = cc_bob_alice
    
    # Create managers
    alice_manager = QECManager(tl, target_fidelity, num_rounds)
    alice_manager.lower_protocols.append(alice.protocol_stack[0])
    alice.protocol_stack[0].upper_protocols.append(alice_manager)
    
    bob_manager = QECManager(tl, target_fidelity, num_rounds)
    bob_manager.lower_protocols.append(bob.protocol_stack[0])
    bob.protocol_stack[0].upper_protocols.append(bob_manager)
    
    # Initialize timeline
    tl.init()
    
    # Start protocol
    print("\nStarting purification protocol...")
    start_time = time.time()
    alice_manager.send_request(error_prob)
    
    # Run simulation
    tl.run()
    exec_time = time.time() - start_time
    
    print(f"\nSimulation completed in {exec_time:.2f} seconds")
    print("-"*70)
    
    # Collect results
    results = {
        'alice_results': alice_manager.results,
        'bob_results': bob_manager.results,
        'alice_times': alice_manager.completion_times,
        'bob_times': bob_manager.completion_times,
        'execution_time': exec_time
    }
    
    # Analyze results
    if alice_manager.results:
        all_fidelities = []
        for res in alice_manager.results:
            if 'fidelities' in res:
                all_fidelities.extend(res['fidelities'])
        
        if all_fidelities:
            avg_fidelity = np.mean(all_fidelities)
            std_fidelity = np.std(all_fidelities)
            max_fidelity = np.max(all_fidelities)
            min_fidelity = np.min(all_fidelities)
            
            print("\nResults Summary:")
            print(f"  Total rounds completed: {len(all_fidelities)}")
            print(f"  Average fidelity: {avg_fidelity:.6f}")
            print(f"  Std deviation: {std_fidelity:.6f}")
            print(f"  Maximum fidelity: {max_fidelity:.6f}")
            print(f"  Minimum fidelity: {min_fidelity:.6f}")
            
            # Check if target achieved
            rounds_above_target = sum(f >= target_fidelity for f in all_fidelities)
            print(f"  Rounds above target ({target_fidelity}): {rounds_above_target}/{len(all_fidelities)}")
            
            results['summary'] = {
                'avg_fidelity': avg_fidelity,
                'std_fidelity': std_fidelity,
                'max_fidelity': max_fidelity,
                'min_fidelity': min_fidelity,
                'total_rounds': len(all_fidelities),
                'rounds_above_target': rounds_above_target,
                'all_fidelities': all_fidelities
            }
            
            # Plot results
            if show_plots:
                plot_results(all_fidelities, alice_manager.completion_times, target_fidelity)
    
    print("="*70)
    return results


def plot_results(fidelities: List[float], times: List[float], 
                 target: float) -> None:
    """Plot simulation results."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fidelity evolution
    rounds = range(1, len(fidelities) + 1)
    ax1.plot(rounds, fidelities, 'b-o', label='Measured Fidelity', markersize=6)
    ax1.axhline(y=target, color='g', linestyle='--', label=f'Target: {target}')
    ax1.axhline(y=np.mean(fidelities), color='r', linestyle=':', 
                label=f'Average: {np.mean(fidelities):.4f}')
    ax1.set_xlabel('Purification Round')
    ax1.set_ylabel('Fidelity')
    ax1.set_title('Fidelity Evolution During QEC Purification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(0.9, min(fidelities)*0.95), 1.01])
    
    # Timing analysis
    if len(times) > 1:
        round_times = [times[i+1] - times[i] for i in range(len(times)-1)]
        ax2.bar(range(1, len(round_times)+1), round_times)
        ax2.set_xlabel('Round Number')
        ax2.set_ylabel('Round Duration (ms)')
        ax2.set_title('Protocol Timing Analysis')
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Fidelity histogram
    plt.figure(figsize=(8, 5))
    plt.hist(fidelities, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=target, color='g', linestyle='--', 
                label=f'Target: {target}')
    plt.axvline(x=np.mean(fidelities), color='r', linestyle=':', 
                label=f'Mean: {np.mean(fidelities):.4f}')
    plt.xlabel('Fidelity')
    plt.ylabel('Count')
    plt.title('Fidelity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def interactive_demo():
    """Interactive demonstration with different parameters."""
    from ipywidgets import interact, IntSlider, FloatSlider
    
    @interact(
        sim_time=IntSlider(value=50, min=10, max=200, step=10, 
                          description='Sim Time (ms)'),
        num_rounds=IntSlider(value=5, min=1, max=20, 
                            description='Rounds'),
        target_fidelity=FloatSlider(value=0.99, min=0.9, max=1.0, step=0.01,
                                   description='Target F'),
        error_prob=FloatSlider(value=0.1, min=0.0, max=0.3, step=0.05,
                              description='Error Prob')
    )
    def run_interactive(sim_time, num_rounds, target_fidelity, error_prob):
        results = run_qec_simulation(
            sim_time=sim_time,
            num_rounds=num_rounds,
            target_fidelity=target_fidelity,
            error_prob=error_prob,
            show_plots=True
        )
        return results
    
    return run_interactive


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run default simulation
    results = run_qec_simulation(
        sim_time=100,
        num_rounds=10,
        target_fidelity=0.99,
        error_prob=0.1,
        channel_distance=1e3,
        show_plots=True
    )