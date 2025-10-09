"""
FIXED RequestAppThroughput - prevents runaway simulation behavior.
Critical changes:
1. Added maximum entanglement limit
2. Proper reservation termination
3. Prevents memory recycling loop
4. Sets quantum memory to RAW after use
5. Timeout mechanism to stop long-running processes
6. Per-reservation tracking of entanglements

NOTE: The quantum router should be configured with 7 quantum memories
      when setting up the network topology.
"""

import numpy as np
from typing import TYPE_CHECKING, Dict, List
from collections import defaultdict

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter
    from sequence.resource_management.memory_manager import MemoryInfo

from sequence.app.request_app import RequestApp

from sequence.constants import SECOND
import sequence.utils.log as log


class RequestAppThroughput(RequestApp):
    """Fixed request application that prevents runaway behavior.
    
    Critical fix: Limits entanglement generation to prevent infinite loops.
    
    Note: This app expects the quantum router to be configured with 7 quantum memories.
    """
    
    # Timeout in microseconds (e.g., 10 seconds)
    TIMEOUT_DURATION = 10 * SECOND
    
    def __init__(self, node: "QuantumRouter"):
        """Initialize the RequestAppThroughput instance."""
        super().__init__(node)
        # Changed to defaultdict(list) where key is reservation, value is list of timestamps
        self.entanglement_timestamps = defaultdict(list)
        self.entanglement_fidelities = defaultdict(list)
        self.max_entanglements = 100  # Reasonable limit to prevent runaway
        self.completed = False
        self.start_time = None  # Track when processing starts
        self.timeout_triggered = False
    
    def get_memory(self, info: "MemoryInfo") -> None:
        """Process entangled memory with termination logic.
        
        CRITICAL FIX: 
        - Stops processing after reaching limit to prevent runaway
        - Sets quantum memory to RAW after use
        - Implements timeout mechanism
        """
        # Initialize start time on first call
        if self.start_time is None:
            self.start_time = self.node.timeline.now()
        
        # Check for timeout
        if not self.timeout_triggered and self._check_timeout():
            self.timeout_triggered = True
            self.completed = True
            log.logger.warning(f"{self.name}: Timeout reached after {self.TIMEOUT_DURATION/SECOND} seconds, stopping")
            # Reset the quantum memory to RAW before stopping
            self.node.resource_manager.update(None, info.memory, "RAW")
            return
            
        # Stop if we've completed
        if self.completed:
            return
            
        if info.state != "ENTANGLED":
            return
            
        if info.index in self.memo_to_reservation:
            reservation = self.memo_to_reservation[info.index]
            
            if info.fidelity >= reservation.fidelity:
                # Record the entanglement per reservation
                self.entanglement_timestamps[reservation].append(self.node.timeline.now())
                self.entanglement_fidelities[reservation].append(info.fidelity)
                self.memory_counter += 1
                
                # CRITICAL: Set quantum memory to RAW after successful use
                # This frees the memory for the next entanglement attempt
                self.node.resource_manager.update(None, info.memory, "RAW")
                log.logger.debug(f"{self.name}: Reset quantum memory {info.index} to RAW state")
                
                # Check if we should stop
                if self.memory_counter >= self.max_entanglements:
                    self.completed = True
                    log.logger.info(f"{self.name}: Reached max entanglements ({self.max_entanglements}), stopping")
                    return
    
    def _check_timeout(self) -> bool:
        """Check if the process has exceeded the timeout duration."""
        if self.start_time is None:
            return False
        current_time = self.node.timeline.now()
        elapsed = current_time - self.start_time
        return elapsed >= self.TIMEOUT_DURATION
    
    def get_time_to_service(self, reservation=None) -> List[float]:
        """Calculate time to service for recorded entanglements.
        
        Args:
            reservation: Optional reservation to get times for. If None, returns all.
        """
        if reservation:
            timestamps = self.entanglement_timestamps.get(reservation, [])
        else:
            # Combine all timestamps if no specific reservation
            timestamps = []
            for res_timestamps in self.entanglement_timestamps.values():
                timestamps.extend(res_timestamps)
            timestamps.sort()
        
        if not timestamps:
            return []
        
        time_to_service = []
        start_time = self.start_time if self.start_time else timestamps[0]
        
        for i, timestamp in enumerate(timestamps):
            if i == 0:
                time_to_service.append((timestamp - start_time) / 1e3)
            else:
                time_to_service.append((timestamp - timestamps[i-1]) / 1e3)
        
        return time_to_service
    
    def get_fidelity(self, reservation=None) -> List[float]:
        """Get recorded fidelities.
        
        Args:
            reservation: Optional reservation to get fidelities for. If None, returns all.
        """
        if reservation:
            return self.entanglement_fidelities.get(reservation, [])
        else:
            # Combine all fidelities if no specific reservation
            all_fidelities = []
            for res_fidelities in self.entanglement_fidelities.values():
                all_fidelities.extend(res_fidelities)
            return all_fidelities
    
    def get_request_to_throughput(self, reservation=None) -> Dict:
        """Calculate final throughput.
        
        Args:
            reservation: Optional reservation to calculate throughput for.
        """
        if reservation:
            timestamps = self.entanglement_timestamps.get(reservation, [])
            if not timestamps:
                return {}
            
            duration = (timestamps[-1] - timestamps[0]) / SECOND
            if duration > 0:
                throughput = len(timestamps) / duration
                return {
                    "reservation": str(reservation),
                    "throughput": throughput,
                    "count": len(timestamps),
                    "duration_seconds": duration
                }
        else:
            # Calculate overall throughput
            all_timestamps = []
            for res_timestamps in self.entanglement_timestamps.values():
                all_timestamps.extend(res_timestamps)
            
            if not all_timestamps:
                return {}
            
            all_timestamps.sort()
            duration = (all_timestamps[-1] - all_timestamps[0]) / SECOND
            
            if duration > 0:
                overall_throughput = len(all_timestamps) / duration
                
                # Also calculate per-reservation throughput
                per_reservation = {}
                for res, timestamps in self.entanglement_timestamps.items():
                    if len(timestamps) > 1:
                        res_duration = (timestamps[-1] - timestamps[0]) / SECOND
                        if res_duration > 0:
                            per_reservation[str(res)] = len(timestamps) / res_duration
                
                return {
                    "summary": overall_throughput,
                    "total_entanglements": len(all_timestamps),
                    "duration_seconds": duration,
                    "per_reservation": per_reservation,
                    "timeout_triggered": self.timeout_triggered
                }
            
        return {}
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics about the entanglement generation."""
        stats = {
            "timeout_duration_seconds": self.TIMEOUT_DURATION / SECOND,
            "max_entanglements": self.max_entanglements,
            "completed": self.completed,
            "timeout_triggered": self.timeout_triggered,
            "total_entanglements": self.memory_counter,
            "reservations_processed": len(self.entanglement_timestamps),
        }
        
        # Add per-reservation statistics
        reservation_stats = {}
        for reservation in self.entanglement_timestamps:
            timestamps = self.entanglement_timestamps[reservation]
            fidelities = self.entanglement_fidelities[reservation]
            
            if timestamps:
                reservation_stats[str(reservation)] = {
                    "count": len(timestamps),
                    "avg_fidelity": np.mean(fidelities) if fidelities else 0,
                    "min_fidelity": min(fidelities) if fidelities else 0,
                    "max_fidelity": max(fidelities) if fidelities else 0,
                }
        
        stats["per_reservation_stats"] = reservation_stats
        
        return stats