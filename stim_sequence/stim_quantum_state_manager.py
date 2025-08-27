"""
=====================================================================
stim_quantum_state_manager.py
---------------------------------------------------------------------
Controller for entanglement generation/purification/swapping protocols.
Coordinates between quantum state, link ledger, and resource manager.
=====================================================================
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union
import logging
import random

from sequence.kernel.entity import Entity
from sequence.kernel.timeline import Timeline
from sequence.kernel.process import Process
from sequence.kernel.event import Event

from stim_quantum_state import StimQuantumState
from stim_memory import LinkLedger, EntanglementLink, PauliBudget, StimMemory
from stim_resource_manager import StimResourceManager
from stim_memory_manager import MemState


class StimQuantumStateManager(Entity):
    """
    Manages entanglement protocols using stabilizer simulation.
    
    Provides:
    - Entanglement Generation (EG)
    - Entanglement Purification (EP)
    - Entanglement Swapping (ES)
    """
    
    def __init__(
        self,
        name: str,
        timeline: Timeline,
        quantum_state: StimQuantumState,
        link_ledger: LinkLedger,
        resource_manager: StimResourceManager
    ):
        super().__init__(name, timeline)
        self.qstate = quantum_state
        self.ledger = link_ledger
        self.rm = resource_manager
        
        # Protocol success rates (can be tuned)
        self.eg_success_rate = 0.9
        self.ep_success_rate = 0.8
        self.es_success_rate = 0.95
        
        logging.info(f"Created StimQuantumStateManager {name}")
    
    def init(self) -> None:
        """Initialize the quantum state manager (Entity abstract method implementation)."""
        pass
    
    def entanglement_generation(
        self,
        local_memory: StimMemory,
        remote_node: str,
        remote_key: Optional[Union[int, str]] = None,
        herald_success: bool = None
    ) -> Optional[str]:
        """
        Generate entanglement between local memory and remote node.
        
        Args:
            local_memory: Local memory to entangle
            remote_node: Name of remote node
            remote_key: Remote memory key (optional)
            herald_success: Override success probability (for testing)
            
        Returns:
            Link ID if successful, None otherwise
        """
        # Determine success
        if herald_success is None:
            herald_success = random.random() < self.eg_success_rate
        
        if not herald_success:
            # Failed - memory returns to RAW
            self.rm.update(None, local_memory, MemState.RAW)
            logging.debug(f"EG failed for memory {local_memory.index}")
            return None
        
        # Success - create Bell pair in circuit
        local_key = local_memory.index  # Use index as key
        if remote_key is None:
            remote_key = 0  # Default remote key
        
        # Bind keys to qubits  
        local_qubit_key = f"{self.name.split('.')[0]}:{local_key}"  # Use node name not full manager name
        remote_qubit_key = f"{remote_node}:{remote_key}"
        
        self.qstate.bind_key(local_qubit_key, local_key)
        self.qstate.bind_key(remote_qubit_key, self.qstate.num_qubits // 2 + remote_key)
        
        # Create Bell pair
        self.qstate.prepare_bell_pair(local_qubit_key, remote_qubit_key)
        
        # Record in ledger with TTL
        ttl = None
        if local_memory.coherence_time and local_memory.cutoff_ratio:
            ttl = int(local_memory.coherence_time * local_memory.cutoff_ratio)
        
        link = self.ledger.create(
            local_mem=local_memory,
            remote_node=remote_node,
            remote_key=remote_key,
            pauli_frame=(0, 0),
            budget=PauliBudget(),
            ttl_ps=ttl
        )
        
        # Notify resource manager
        self.rm.on_link_created(link.link_id)
        
        logging.info(f"EG success: created link {link.link_id} between {local_memory.index} and {remote_node}:{remote_key}")
        return link.link_id
    
    def entanglement_purification(
        self,
        link_ids: List[str],
        protocol: str = "simple"
    ) -> Optional[str]:
        """
        Purify multiple links into one higher-quality link.
        
        Args:
            link_ids: List of link IDs to consume
            protocol: Purification protocol type
            
        Returns:
            New link ID if successful, None otherwise
        """
        if len(link_ids) < 2:
            logging.warning("EP requires at least 2 links")
            return None
        
        # Get links
        links = [self.ledger.get(lid) for lid in link_ids]
        
        # Check all links are live and to same remote
        remote_node = links[0].remote.node
        for link in links:
            if not link.is_live():
                logging.warning(f"Link {link.link_id} not live for EP")
                return None
            if link.remote.node != remote_node:
                logging.warning("EP links must be to same remote node")
                return None
        
        # Simulate purification success
        if random.random() > self.ep_success_rate:
            # Failed - consume all links
            for lid in link_ids:
                self.ledger.mark_failed(lid)
                self.rm.on_link_failed(lid)
            logging.debug(f"EP failed, consumed links {link_ids}")
            return None
        
        # Success - consume all but first link, improve its quality
        kept_link = links[0]
        for lid in link_ids[1:]:
            self.ledger.mark_consumed(lid)
            self.rm.on_link_consumed(lid)
        
        # Reduce error budget (simplified)
        kept_link.budget.p_ix *= 0.5
        kept_link.budget.p_iz *= 0.5
        kept_link.budget.p_xi *= 0.5
        kept_link.budget.p_zi *= 0.5
        
        logging.info(f"EP success: purified {len(link_ids)} links into {kept_link.link_id}")
        return kept_link.link_id
    
    def entanglement_swapping(
        self,
        link1_id: str,
        link2_id: str
    ) -> Optional[str]:
        """
        Swap entanglement to connect outer nodes.
        
        Args:
            link1_id: First link (A-B)
            link2_id: Second link (B-C)
            
        Returns:
            New link ID (A-C) if successful, None otherwise
        """
        # Get links
        link1 = self.ledger.get(link1_id)
        link2 = self.ledger.get(link2_id)
        
        if not (link1.is_live() and link2.is_live()):
            logging.warning("Both links must be live for ES")
            return None
        
        # Perform Bell measurement on local qubits
        local_key1 = f"{self.name}:{link1.local.index}"
        local_key2 = f"{self.name}:{link2.local.index}"
        
        m1, m2 = self.qstate.apply_swap(local_key1, local_key2)
        
        # Check success
        if random.random() > self.es_success_rate:
            # Failed
            self.ledger.mark_failed(link1_id)
            self.ledger.mark_failed(link2_id)
            self.rm.on_link_failed(link1_id)
            self.rm.on_link_failed(link2_id)
            logging.debug(f"ES failed for links {link1_id}, {link2_id}")
            return None
        
        # Success - consume old links
        self.ledger.mark_consumed(link1_id)
        self.ledger.mark_consumed(link2_id)
        self.rm.on_link_consumed(link1_id)
        self.rm.on_link_consumed(link2_id)
        
        # Create new link between outer nodes
        # For MVP, just pick first available memory
        new_memory = self.rm.mm.claim_one()
        if not new_memory:
            logging.warning("No memory available for swapped link")
            return None
        
        # Combine Pauli frames based on measurement outcomes
        new_frame = (
            (link1.pauli_frame[0] + link2.pauli_frame[0] + m1) & 1,
            (link1.pauli_frame[1] + link2.pauli_frame[1] + m2) & 1
        )
        
        # Combine error budgets
        new_budget = PauliBudget()
        new_budget.p_ix = min(1.0, link1.budget.p_ix + link2.budget.p_xi)
        new_budget.p_iz = min(1.0, link1.budget.p_iz + link2.budget.p_zi)
        new_budget.p_xi = min(1.0, link1.budget.p_xi + link2.budget.p_ix)
        new_budget.p_zi = min(1.0, link1.budget.p_zi + link2.budget.p_iz)
        new_budget.p_xx = min(1.0, link1.budget.p_xx + link2.budget.p_xx)
        new_budget.p_zz = min(1.0, link1.budget.p_zz + link2.budget.p_zz)
        
        # Determine remote node (the one that's not us)
        if link1.remote.node == self.name:
            remote_node = link2.remote.node
            remote_key = link2.remote.key
        else:
            remote_node = link1.remote.node
            remote_key = link1.remote.key
        
        # Create new link
        new_link = self.ledger.create(
            local_mem=new_memory.memory,
            remote_node=remote_node,
            remote_key=remote_key,
            pauli_frame=new_frame,
            budget=new_budget,
            ttl_ps=min(
                link1.deadline_ps - self.timeline.now() if link1.deadline_ps else float('inf'),
                link2.deadline_ps - self.timeline.now() if link2.deadline_ps else float('inf')
            ) if link1.deadline_ps or link2.deadline_ps else None
        )
        
        self.rm.on_link_created(new_link.link_id)
        
        logging.info(
            f"ES success: swapped {link1_id} + {link2_id} -> {new_link.link_id} "
            f"connecting to {remote_node}:{remote_key}"
        )
        return new_link.link_id
    
    # --- Helper methods for rules to call ---
    
    def try_entangle_memory(self, memory: StimMemory, target_node: str) -> bool:
        """
        Attempt to entangle a memory with a target node.
        Used by rules during evaluation.
        """
        link_id = self.entanglement_generation(memory, target_node)
        return link_id is not None
    
    def try_purify_links(self, target_node: str, num_links: int = 2) -> bool:
        """
        Try to purify multiple links to a target node.
        Used by rules during evaluation.
        """
        links = self.ledger.links_by_remote(target_node, live_only=True)
        if len(links) >= num_links:
            link_ids = [l.link_id for l in links[:num_links]]
            result = self.entanglement_purification(link_ids)
            return result is not None
        return False
    
    def try_swap_links(self, node1: str, node2: str) -> bool:
        """
        Try to swap links to connect two remote nodes.
        Used by rules during evaluation.
        """
        links1 = self.ledger.links_by_remote(node1, live_only=True)
        links2 = self.ledger.links_by_remote(node2, live_only=True)
        
        if links1 and links2:
            result = self.entanglement_swapping(links1[0].link_id, links2[0].link_id)
            return result is not None
        return False