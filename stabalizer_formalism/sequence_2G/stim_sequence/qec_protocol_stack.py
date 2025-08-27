"""
=====================================================================
qec_protocol_stack.py
---------------------------------------------------------------------
QEC [[7,1,3]] purification protocol with classical communication
between nodes, following SeQUeNCe's protocol stack architecture.
=====================================================================
"""

from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, List, Dict, Any
import numpy as np
import logging
import random
from dataclasses import dataclass

if TYPE_CHECKING:
    from stim_node import StimNode

from sequence.message import Message
from sequence.protocol import StackProtocol
from sequence.kernel.event import Event
from sequence.kernel.process import Process

from enhanced_stabilizer_circuit import EnhancedStabilizerCircuit
from qec_protocol import QEC713Protocol, EntanglementPurificationProtocol


class QECMsgType(Enum):
    """Message types for QEC protocol communication."""
    INIT_PURIFICATION = auto()
    SYNDROME_DATA = auto()
    CORRECTION_CONFIRM = auto()
    SWAPPING_RESULT = auto()
    FIDELITY_REPORT = auto()
    PROTOCOL_COMPLETE = auto()


@dataclass
class QECMessage(Message):
    """Message for QEC protocol communication."""
    
    def __init__(self, msg_type: QECMsgType, receiver: str, **kwargs):
        super().__init__(msg_type, receiver)
        self.protocol_type = QECProtocol
        self.msg_type = msg_type
        
        # Store relevant data based on message type
        if msg_type == QECMsgType.INIT_PURIFICATION:
            self.round_id = kwargs.get('round_id', 0)
            self.error_prob = kwargs.get('error_prob', 0.1)
            self.target_fidelity = kwargs.get('target_fidelity', 0.99)
            
        elif msg_type == QECMsgType.SYNDROME_DATA:
            self.syndrome = kwargs.get('syndrome', [])
            self.round_id = kwargs.get('round_id', 0)
            self.station_id = kwargs.get('station_id', 0)
            
        elif msg_type == QECMsgType.CORRECTION_CONFIRM:
            self.corrections = kwargs.get('corrections', [])
            self.round_id = kwargs.get('round_id', 0)
            
        elif msg_type == QECMsgType.SWAPPING_RESULT:
            self.measurement_outcomes = kwargs.get('outcomes', [])
            self.round_id = kwargs.get('round_id', 0)
            
        elif msg_type == QECMsgType.FIDELITY_REPORT:
            self.fidelity = kwargs.get('fidelity', 0.0)
            self.round_id = kwargs.get('round_id', 0)
            
        elif msg_type == QECMsgType.PROTOCOL_COMPLETE:
            self.final_fidelity = kwargs.get('final_fidelity', 0.0)
            self.total_rounds = kwargs.get('total_rounds', 0)


def pair_qec_protocols(alice: "QECProtocol", bob: "QECProtocol") -> None:
    """Pair two QEC protocol instances for communication."""
    alice.partner = bob
    bob.partner = alice
    alice.role = 0  # Alice
    bob.role = 1    # Bob


class QECProtocol(StackProtocol):
    """
    Stack protocol implementation for [[7,1,3]] QEC purification.
    Handles classical communication between nodes during purification.
    """
    
    def __init__(self, owner: "StimNode", name: str):
        """
        Initialize QEC protocol.
        
        Args:
            owner: Node hosting this protocol
            name: Protocol instance name
        """
        super().__init__(owner, name)
        
        self.partner: Optional[QECProtocol] = None
        self.role = -1  # 0 for Alice, 1 for Bob
        
        # Protocol state
        self.working = False
        self.current_round = 0
        self.total_rounds = 0
        
        # QEC components
        self.qec = QEC713Protocol()
        self.purification = EntanglementPurificationProtocol()
        
        # Circuit storage
        self.current_circuit: Optional[EnhancedStabilizerCircuit] = None
        self.syndrome_data: Dict[int, List] = {}
        
        # Metrics
        self.fidelities: List[float] = []
        self.error_corrections: List[int] = []
        self.latencies: List[float] = []
        self.throughput: float = 0.0
        
        # Protocol parameters
        self.error_probability = 0.1
        self.target_fidelity = 0.99
        self.shots_per_round = 1000
        
        logging.info(f"Initialized QECProtocol {name} on {owner.name}")
    
    def push(self, rounds: int, target_fidelity: float = 0.99, error_prob: float = 0.1) -> None:
        """
        Start purification protocol (called by upper layer).
        
        Args:
            rounds: Number of purification rounds
            target_fidelity: Target fidelity to achieve
            error_prob: Error probability per round
        """
        if self.role != 0:
            raise AssertionError("Purification must be initiated by Alice")
        
        self.total_rounds = rounds
        self.target_fidelity = target_fidelity
        self.error_probability = error_prob
        self.current_round = 0
        self.working = True
        
        logging.info(f"{self.name} starting {rounds} rounds of purification")
        
        # Start first round
        self.start_purification_round()
    
    def start_purification_round(self) -> None:
        """Start a new purification round."""
        if self.current_round >= self.total_rounds:
            self.complete_protocol()
            return
        
        self.current_round += 1
        start_time = self.owner.timeline.now()
        
        logging.info(f"{self.name} starting round {self.current_round}/{self.total_rounds}")
        
        # Send initialization message to Bob
        msg = QECMessage(
            QECMsgType.INIT_PURIFICATION,
            self.partner.name,
            round_id=self.current_round,
            error_prob=self.error_probability,
            target_fidelity=self.target_fidelity
        )
        self.owner.send_message(self.partner.owner.name, msg)
        
        # Schedule circuit creation
        process = Process(self, "create_purification_circuit", [])
        event = Event(self.owner.timeline.now() + 100, process)
        self.owner.timeline.schedule(event)
    
    def create_purification_circuit(self) -> None:
        """Create and execute purification circuit."""
        # Create circuit with error injection
        self.current_circuit = self.purification.create_purification_circuit(
            error_probability=self.error_probability,
            apply_errors=True
        )
        
        # Measure stabilizers for error detection
        self.measure_stabilizers()
    
    def measure_stabilizers(self) -> None:
        """Measure stabilizers and extract syndrome."""
        if not self.current_circuit:
            return
        
        # Sample the circuit to get syndrome measurements
        samples = self.current_circuit.sample(shots=100)
        
        # Extract syndrome (simplified - take majority vote)
        syndrome = []
        for i in range(6):  # 6 stabilizers in [[7,1,3]] code
            bit_count = np.sum(samples[:, i] if i < samples.shape[1] else 0)
            syndrome.append(1 if bit_count > 50 else 0)
        
        self.syndrome_data[self.current_round] = syndrome
        
        # Send syndrome to partner for verification
        msg = QECMessage(
            QECMsgType.SYNDROME_DATA,
            self.partner.name,
            syndrome=syndrome,
            round_id=self.current_round,
            station_id=1 if self.role == 0 else 2
        )
        self.owner.send_message(self.partner.owner.name, msg)
        
        # Schedule error correction
        process = Process(self, "apply_corrections", [syndrome])
        event = Event(self.owner.timeline.now() + 100, process)
        self.owner.timeline.schedule(event)
    
    def apply_corrections(self, syndrome: List[int]) -> None:
        """Apply error corrections based on syndrome."""
        corrections = self.decode_syndrome(syndrome)
        
        if corrections:
            logging.debug(f"{self.name} applying {len(corrections)} corrections")
            for qubit_idx in corrections:
                if qubit_idx < self.current_circuit.num_qubits:
                    # Apply correction (simplified - just track it)
                    self.error_corrections.append(qubit_idx)
        
        # Send confirmation to partner
        msg = QECMessage(
            QECMsgType.CORRECTION_CONFIRM,
            self.partner.name,
            corrections=corrections,
            round_id=self.current_round
        )
        self.owner.send_message(self.partner.owner.name, msg)
        
        # Proceed to entanglement swapping
        if self.role == 0:  # Alice initiates swapping
            process = Process(self, "perform_swapping", [])
            event = Event(self.owner.timeline.now() + 200, process)
            self.owner.timeline.schedule(event)
    
    def perform_swapping(self) -> None:
        """Perform entanglement swapping measurements."""
        if not self.current_circuit:
            return
        
        # Simulate swapping measurements
        measurements = []
        for i in range(7):  # 7 logical qubits
            # Simplified measurement outcomes
            m1 = random.randint(0, 1)
            m2 = random.randint(0, 1)
            measurements.append((m1, m2))
        
        # Send results to Bob
        msg = QECMessage(
            QECMsgType.SWAPPING_RESULT,
            self.partner.name,
            outcomes=measurements,
            round_id=self.current_round
        )
        self.owner.send_message(self.partner.owner.name, msg)
        
        # Calculate fidelity
        self.calculate_fidelity()
    
    def calculate_fidelity(self) -> None:
        """Calculate fidelity of current round."""
        if not self.current_circuit:
            return
        
        # Perform tomography
        fidelity = self.purification.calculate_fidelity(
            self.current_circuit,
            target_qubits=(0, 49),
            shots=self.shots_per_round
        )
        
        self.fidelities.append(fidelity)
        
        # Calculate latency
        round_time = (self.owner.timeline.now() - 
                     (self.current_round - 1) * 1e9) * 1e-12  # Convert to seconds
        self.latencies.append(round_time)
        
        logging.info(f"{self.name} Round {self.current_round}: Fidelity = {fidelity:.6f}")
        
        # Send fidelity report to partner
        msg = QECMessage(
            QECMsgType.FIDELITY_REPORT,
            self.partner.name,
            fidelity=fidelity,
            round_id=self.current_round
        )
        self.owner.send_message(self.partner.owner.name, msg)
        
        # Check if target achieved or continue
        if fidelity >= self.target_fidelity:
            logging.info(f"{self.name} achieved target fidelity!")
            self.complete_protocol()
        else:
            # Schedule next round
            process = Process(self, "start_purification_round", [])
            event = Event(self.owner.timeline.now() + 1000, process)
            self.owner.timeline.schedule(event)
    
    def complete_protocol(self) -> None:
        """Complete the protocol and report results."""
        self.working = False
        
        if self.fidelities:
            avg_fidelity = np.mean(self.fidelities)
            total_time = sum(self.latencies)
            self.throughput = len(self.fidelities) / total_time if total_time > 0 else 0
            
            logging.info(f"{self.name} Protocol complete:")
            logging.info(f"  Average fidelity: {avg_fidelity:.6f}")
            logging.info(f"  Total rounds: {len(self.fidelities)}")
            logging.info(f"  Throughput: {self.throughput:.2f} rounds/sec")
            
            # Send completion message
            msg = QECMessage(
                QECMsgType.PROTOCOL_COMPLETE,
                self.partner.name,
                final_fidelity=avg_fidelity,
                total_rounds=len(self.fidelities)
            )
            self.owner.send_message(self.partner.owner.name, msg)
            
            # Report to upper layer
            self._pop(info={
                'fidelities': self.fidelities,
                'avg_fidelity': avg_fidelity,
                'throughput': self.throughput,
                'total_corrections': len(self.error_corrections)
            })
    
    def decode_syndrome(self, syndrome: List[int]) -> List[int]:
        """
        Decode syndrome to determine error locations.
        Simplified lookup table decoder.
        """
        corrections = []
        
        # Simple syndrome decoding (real implementation would use lookup table)
        syndrome_int = int(''.join(map(str, syndrome)), 2)
        
        # Map common syndromes to corrections
        syndrome_map = {
            0b001001: [0],  # X error on qubit 0
            0b010010: [1],  # X error on qubit 1
            0b100100: [2],  # X error on qubit 2
            0b111000: [3],  # Z error on qubit 3
            # Add more syndrome patterns as needed
        }
        
        if syndrome_int in syndrome_map:
            corrections = syndrome_map[syndrome_int]
        
        return corrections
    
    def received_message(self, src: str, msg: "QECMessage") -> None:
        """Handle received messages from partner."""
        
        if msg.msg_type == QECMsgType.INIT_PURIFICATION:
            # Bob receives initialization
            self.current_round = msg.round_id
            self.error_probability = msg.error_prob
            self.target_fidelity = msg.target_fidelity
            self.working = True
            
            # Create circuit on Bob's side
            process = Process(self, "create_purification_circuit", [])
            event = Event(self.owner.timeline.now() + 50, process)
            self.owner.timeline.schedule(event)
        
        elif msg.msg_type == QECMsgType.SYNDROME_DATA:
            # Store partner's syndrome for comparison
            partner_syndrome = msg.syndrome
            round_id = msg.round_id
            
            logging.debug(f"{self.name} received syndrome from {src}: {partner_syndrome}")
            
            # Could compare with own syndrome for verification
            
        elif msg.msg_type == QECMsgType.CORRECTION_CONFIRM:
            # Partner confirmed corrections
            logging.debug(f"{self.name} partner applied {len(msg.corrections)} corrections")
            
        elif msg.msg_type == QECMsgType.SWAPPING_RESULT:
            # Bob receives swapping results
            outcomes = msg.outcomes
            
            # Apply Pauli corrections based on outcomes
            for i, (m1, m2) in enumerate(outcomes):
                if m1:  # X correction needed
                    logging.debug(f"X correction on logical qubit {i}")
                if m2:  # Z correction needed  
                    logging.debug(f"Z correction on logical qubit {i}")
            
            # Bob also calculates fidelity
            self.calculate_fidelity()
            
        elif msg.msg_type == QECMsgType.FIDELITY_REPORT:
            # Store partner's fidelity for comparison
            partner_fidelity = msg.fidelity
            logging.debug(f"{self.name} partner reported fidelity: {partner_fidelity:.6f}")
            
        elif msg.msg_type == QECMsgType.PROTOCOL_COMPLETE:
            # Protocol completed
            self.working = False
            logging.info(f"{self.name} received completion: "
                        f"final fidelity = {msg.final_fidelity:.6f}")
            
            # Report to upper layer
            if self.fidelities:
                self._pop(info={
                    'fidelities': self.fidelities,
                    'partner_fidelity': msg.final_fidelity
                })