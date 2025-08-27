"""
=====================================================================
qec_purification_app.py
---------------------------------------------------------------------
Complete application integrating [[7,1,3]] QEC purification protocol
with the Stim-based node architecture.
=====================================================================
"""

import numpy as np
import logging
from typing import Optional, Any
import matplotlib.pyplot as plt

from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event

from stim_node import StimNode
from stim_memory_manager import StimMemoryInfo, MemState
from qec_protocol import EntanglementPurificationProtocol
from enhanced_stabilizer_circuit import EnhancedStabilizerCircuit
from typing import Dict


class QECPurificationApp:
    """
    Application that runs [[7,1,3]] quantum error correction
    entanglement purification protocol on Stim nodes.
    """
    
    def __init__(self, node: StimNode, partner_name: str):
        """
        Initialize QEC purification application.
        
        Args:
            node: The StimNode this app is attached to
            partner_name: Name of partner node for entanglement
        """
        self.node = node
        self.partner_name = partner_name
        self.protocol = EntanglementPurificationProtocol()
        
        # Results storage
        self.fidelities = []
        self.circuits = []
        
        # Register with node
        node.set_app(self)
        
        logging.info(f"Initialized QECPurificationApp on {node.name}")
    
    def run_purification(self, 
                        num_rounds: int = 10,
                        error_probability: float = 0.1,
                        shots_per_round: int = 1000) -> None:
        """
        Run multiple rounds of purification protocol.
        
        Args:
            num_rounds: Number of purification rounds
            error_probability: Error probability per round
            shots_per_round: Shots for fidelity calculation
        """
        logging.info(f"Starting {num_rounds} rounds of QEC purification")
        
        for round_idx in range(num_rounds):
            # Create purification circuit
            circuit = self.protocol.create_purification_circuit(
                error_probability=error_probability,
                apply_errors=True
            )
            
            # Store circuit
            self.circuits.append(circuit)
            
            # Calculate fidelity
            fidelity = self.protocol.calculate_fidelity(
                circuit,
                target_qubits=(0, 49),
                shots=shots_per_round
            )
            
            self.fidelities.append(fidelity)
            
            logging.info(f"Round {round_idx + 1}: Fidelity = {fidelity:.4f}")
            
            # Schedule next round
            if round_idx < num_rounds - 1:
                now = self.node.timeline.now()
                process = Process(self, "_continue_purification", 
                                [round_idx + 1, num_rounds, error_probability, shots_per_round])
                event = Event(now + 1e9, process)  # 1ms between rounds
                self.node.timeline.schedule(event)
    
    def _continue_purification(self, round_idx, num_rounds, error_probability, shots_per_round):
        """Internal method to continue purification rounds."""
        # This would be called by the timeline scheduler
        pass
    
    def get_reservation_result(self, reservation: Any, result: bool):
        """Handle reservation results from network manager."""
        if result:
            logging.info(f"Reservation approved for QEC purification")
        else:
            logging.warning(f"Reservation failed for QEC purification")
    
    def get_memory(self, info: StimMemoryInfo):
        """Handle memory state updates."""
        if info.state == MemState.ENTANGLED:
            logging.debug(f"Memory {info.index} entangled with {info.remote_node}")
    
    def analyze_results(self) -> Dict:
        """
        Analyze purification results.
        
        Returns:
            Dictionary with statistics
        """
        if not self.fidelities:
            return {"error": "No purification rounds completed"}
        
        results = {
            "num_rounds": len(self.fidelities),
            "average_fidelity": np.mean(self.fidelities),
            "std_fidelity": np.std(self.fidelities),
            "max_fidelity": np.max(self.fidelities),
            "min_fidelity": np.min(self.fidelities),
            "fidelities": self.fidelities
        }
        
        return results
    
    def plot_fidelity_evolution(self, save_path: Optional[str] = None):
        """
        Plot fidelity evolution over purification rounds.
        
        Args:
            save_path: Optional path to save figure
        """
        if not self.fidelities:
            logging.warning("No fidelities to plot")
            return
        
        plt.figure(figsize=(10, 6))
        rounds = range(1, len(self.fidelities) + 1)
        
        # Plot fidelities
        plt.plot(rounds, self.fidelities, 'b-o', label='Measured Fidelity')
        
        # Add average line
        avg_fidelity = np.mean(self.fidelities)
        plt.axhline(y=avg_fidelity, color='r', linestyle='--', 
                   label=f'Average: {avg_fidelity:.4f}')
        
        # Add ideal Bell state fidelity
        plt.axhline(y=1.0, color='g', linestyle=':', 
                   label='Ideal |Φ+⟩')
        
        plt.xlabel('Purification Round')
        plt.ylabel('Fidelity')
        plt.title('[[7,1,3]] QEC Entanglement Purification Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function demonstrating the QEC purification protocol.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("[[7,1,3]] Quantum Error Correction Entanglement Purification Demo")
    print("="*70)
    
    # Create timeline
    tl = Timeline(stop_time=1e12)  # 1 second simulation
    tl.show_progress = False
    
    # Create nodes with sufficient qubits
    alice = StimNode("Alice", tl, num_memories=80)
    bob = StimNode("Bob", tl, num_memories=80)
    
    # Create QEC purification apps
    alice_app = QECPurificationApp(alice, "Bob")
    bob_app = QECPurificationApp(bob, "Alice")
    
    # Run standalone protocol test (without full network simulation)
    protocol = EntanglementPurificationProtocol()
    
    print("\nRunning 10 rounds of purification protocol...")
    print("-" * 40)
    
    fidelities = []
    for round_idx in range(10):
        # Create circuit with random errors
        circuit = protocol.create_purification_circuit(
            error_probability=0.1,
            apply_errors=True
        )
        
        # Calculate fidelity
        fidelity = protocol.calculate_fidelity(
            circuit,
            target_qubits=(0, 49),
            shots=1000
        )
        
        fidelities.append(fidelity)
        print(f"Round {round_idx + 1:2d}: Fidelity = {fidelity:.6f}")
    
    # Print statistics
    print("\n" + "="*40)
    print("Results Summary:")
    print("-" * 40)
    print(f"Average Fidelity: {np.mean(fidelities):.6f}")
    print(f"Std Deviation:    {np.std(fidelities):.6f}")
    print(f"Maximum Fidelity: {np.max(fidelities):.6f}")
    print(f"Minimum Fidelity: {np.min(fidelities):.6f}")
    
    # Test Bell state fidelity (without errors)
    print("\n" + "="*40)
    print("Testing ideal case (no errors):")
    print("-" * 40)
    
    ideal_circuit = protocol.create_purification_circuit(
        error_probability=0.0,
        apply_errors=False
    )
    
    ideal_fidelity = protocol.calculate_fidelity(
        ideal_circuit,
        target_qubits=(0, 49),
        shots=10000
    )
    
    print(f"Ideal Fidelity: {ideal_fidelity:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    rounds = range(1, len(fidelities) + 1)
    
    plt.plot(rounds, fidelities, 'b-o', label='With Errors (p=0.1)')
    plt.axhline(y=np.mean(fidelities), color='r', linestyle='--', 
               label=f'Average: {np.mean(fidelities):.4f}')
    plt.axhline(y=ideal_fidelity, color='g', linestyle=':', 
               label=f'Ideal: {ideal_fidelity:.4f}')
    
    plt.xlabel('Purification Round')
    plt.ylabel('Fidelity with |Φ+⟩')
    plt.title('[[7,1,3]] QEC Entanglement Purification Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.95, 1.001])
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    main()