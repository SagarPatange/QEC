"""
RequestAppThroughput implementation compatible with both ket state and stabilizer formalisms.
This can be integrated directly into your main_test.py functions.
"""

import numpy as np
from typing import TYPE_CHECKING, Dict, List
from collections import defaultdict

if TYPE_CHECKING:
    from sequence.topology.node import QuantumRouter
    from sequence.network_management.reservation import Reservation
    from sequence.resource_management.memory_manager import MemoryInfo

from sequence.app.request_app import RequestApp
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.kernel.quantum_manager import QuantumManager
from sequence.constants import SECOND, KET_STATE_FORMALISM, STABILIZER_FORMALISM
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
import sequence.utils.log as log


class RequestAppThroughput(RequestApp):
    """Request application for throughput measurements.
    
    Compatible with both ket state and stabilizer formalisms.
    Tracks timing, fidelity, and throughput metrics for entanglement generation.
    """
    
    def __init__(self, node: "QuantumRouter"):
        """Initialize the RequestAppThroughput instance.
        
        Args:
            node (QuantumRouter): The quantum router node this app is attached to
        """
        super().__init__(node)
        self.entanglement_timestamps = defaultdict(list)  # reservation -> list[float]
        self.entanglement_fidelities = defaultdict(list)  # reservation -> list[float]
        self.time_to_service_list = []  # Direct list for time to service
        self.fidelity_list = []  # Direct list for fidelities
        self.request_to_throughput = {}  # Maps reservations to throughput
        
    def get_memory(self, info: "MemoryInfo") -> None:
        """Process received entangled memory.
        
        Works with both ket state and stabilizer formalisms.
        
        Args:
            info (MemoryInfo): Information about the entangled memory
        """
        if info.state != "ENTANGLED":
            return
            
        if info.index in self.memo_to_reservation:
            reservation = self.memo_to_reservation[info.index]
            
            # Check if memory meets fidelity requirements
            if info.fidelity >= reservation.fidelity:
                current_time = self.node.timeline.now()
                
                # Record timestamp and fidelity for this reservation
                self.entanglement_timestamps[reservation].append(current_time)
                self.entanglement_fidelities[reservation].append(info.fidelity)
                
                # Also maintain direct lists for easy access
                self.fidelity_list.append(info.fidelity)
                
                # Calculate time to service
                if len(self.entanglement_timestamps[reservation]) == 1:
                    # First entanglement - time from reservation start
                    time_to_service = current_time - reservation.start_time
                else:
                    # Subsequent entanglements - time from previous
                    prev_time = self.entanglement_timestamps[reservation][-2]
                    time_to_service = current_time - prev_time
                self.time_to_service_list.append(time_to_service)
                
                # Log the successful entanglement
                log.logger.info(
                    f"{self.name}: Entanglement established with {info.remote_node}, "
                    f"fidelity={info.fidelity:.6f}, "
                    f"count={len(self.entanglement_timestamps[reservation])}"
                )
                
                # Update memory state to RAW for reuse
                self.node.resource_manager.update(None, info.memory, "RAW")
                
                # Increment counter for compatibility
                self.memory_counter += 1
                
                # Update throughput calculation
                self._update_throughput(reservation)
            else:
                log.logger.info(
                    f"{self.name}: Entanglement fidelity {info.fidelity:.6f} "
                    f"below threshold {reservation.fidelity}"
                )
    
    def _update_throughput(self, reservation: "Reservation") -> None:
        """Update throughput calculation for a reservation.
        
        Args:
            reservation: The reservation to update throughput for
        """
        timestamps = self.entanglement_timestamps[reservation]
        if len(timestamps) > 0:
            time_elapsed = (self.node.timeline.now() - reservation.start_time) / SECOND
            if time_elapsed > 0:
                self.request_to_throughput[reservation] = len(timestamps) / time_elapsed
    
    def get_time_to_service(self) -> List[float]:
        """Get the time to service for each entanglement.
        
        Returns:
            List of time-to-service values in nanoseconds
        """
        # Convert from picoseconds to nanoseconds
        return [t / 1e3 for t in self.time_to_service_list]
    
    def get_fidelity(self) -> List[float]:
        """Get the fidelity of each entanglement.
        
        Returns:
            List of fidelity values
        """
        return self.fidelity_list
    
    def get_request_to_throughput(self) -> Dict["Reservation", float]:
        """Get throughput for each reservation.
        
        Returns:
            Dictionary mapping reservations to throughput (entanglements/second)
        """
        # Final update of all throughputs
        for reservation in self.entanglement_timestamps:
            self._update_throughput(reservation)
        return self.request_to_throughput


