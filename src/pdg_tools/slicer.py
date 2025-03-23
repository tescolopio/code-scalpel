import networkx as nx
from typing import Dict, List, Set, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import ast
from collections import defaultdict

class SliceType(Enum):
    """Types of program slices."""
    BACKWARD = 'backward'
    FORWARD = 'forward'
    CONTROL = 'control'
    DATA = 'data'
    THIN = 'thin'
    UNION = 'union'
    INTERSECTION = 'intersection'

@dataclass
class SlicingCriteria:
    """Criteria for program slicing."""
    nodes: Set[str]
    variables: Set[str]
    line_range: Optional[tuple[int, int]] = None
    dependency_types: Set[str] = None
    include_control: bool = True
    include_data: bool = True

@dataclass
class SliceInfo:
    """Information about a program slice."""
    nodes: Set[str]
    edges: Set[tuple[str, str]]
    variables: Set[str]
    line_range: tuple[int, int]
    size: int
    complexity: int

class ProgramSlicer:
    """Advanced program slicer with multiple slicing strategies."""
    
    def __init__(self, pdg: nx.DiGraph):
        self.pdg = pdg
        self.cache = {}
        self._initialize_indices()

    def _initialize_indices(self):
        """Initialize indices for faster slicing."""
        self.var_def_sites = defaultdict(set)
        self.var_use_sites = defaultdict(set)
        self.line_to_nodes = defaultdict(set)
        
        for node, data in self.pdg.nodes(data=True):
            # Index variable definitions
            if 'defines' in data:
                for var in data['defines']:
                    self.var_def_sites[var].add(node)
            
            # Index variable uses
            if 'uses' in data:
                for var in data['uses']:
                    self.var_use_sites[var].add(node)
            
            # Index line numbers
            if 'lineno' in data:
                self.line_to_nodes[data['lineno']].add(node)

    def compute_slice(self, criteria: Union[SlicingCriteria, str],
                     slice_type: SliceType = SliceType.BACKWARD) -> nx.DiGraph:
        """
        Compute a program slice based on given criteria.
        
        Args:
            criteria: Slicing criteria (node ID or SlicingCriteria object)
            slice_type: Type of slice to compute
        
        Returns:
            Sliced PDG
        """
        if isinstance(criteria, str):
            criteria = SlicingCriteria(
                nodes={criteria},
                variables=set(),
                include_control=True,
                include_data=True
            )
        
        cache_key = self._make_cache_key(criteria, slice_type)
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        if slice_type == SliceType.BACKWARD:
            sliced_pdg = self._compute_backward_slice(criteria)
        elif slice_type == SliceType.FORWARD:
            sliced_pdg = self._compute_forward_slice(criteria)
        elif slice_type == SliceType.THIN:
            sliced_pdg = self._compute_thin_slice(criteria)
        elif slice_type in (SliceType.UNION, SliceType.INTERSECTION):
            sliced_pdg = self._compute_composite_slice(criteria, slice_type)
        else:
            sliced_pdg = self._compute_specialized_slice(criteria, slice_type)
        
        self.cache[cache_key] = sliced_pdg
        return sliced_pdg.copy()

    def get_slice_info(self, sliced_pdg: nx.DiGraph) -> SliceInfo:
        """Get information about a program slice."""
        nodes = set(sliced_pdg.nodes())
        edges = set(sliced_pdg.edges())
        
        variables = set()
        line_numbers = []
        
        for node, data in sliced_pdg.nodes(data=True):
            if 'defines' in data:
                variables.update(data['defines'])
            if 'uses' in data:
                variables.update(data['uses'])
            if 'lineno' in data:
                line_numbers.append(data['lineno'])
        
        return SliceInfo(
            nodes=nodes,
            edges=edges,
            variables=variables,
            line_range=(min(line_numbers), max(line_numbers)) if line_numbers else (0, 0),
            size=len(nodes),
            complexity=self._calculate_slice_complexity(sliced_pdg)
        )

    def compute_chop(self, source_criteria: SlicingCriteria,
                    target_criteria: SlicingCriteria) -> nx.DiGraph:
        """
        Compute a program chop between source and target criteria.
        
        A chop is the intersection of a forward slice from the source
        and a backward slice from the target.
        """
        forward_slice = self.compute_slice(source_criteria, SliceType.FORWARD)
        backward_slice = self.compute_slice(target_criteria, SliceType.BACKWARD)
        
        # Compute intersection
        chop_nodes = set(forward_slice.nodes()) & set(backward_slice.nodes())
        return self._induce_subgraph(chop_nodes)

    def decompose_slice(self, criteria: SlicingCriteria) -> Dict[str, nx.DiGraph]:
        """
        Decompose a slice into meaningful components.
        
        Returns:
            Dictionary mapping component types to subgraphs
        """
        full_slice = self.compute_slice(criteria)
        
        return {
            'data_only': self._extract_data_component(full_slice),
            'control_only': self._extract_control_component(full_slice),
            'core': self._extract_core_component(full_slice),
            'auxiliary': self._extract_auxiliary_component(full_slice)
        }

    def _compute_backward_slice(self, criteria: SlicingCriteria) -> nx.DiGraph:
        """Compute a backward slice."""
        sliced_nodes = set()
        worklist = set()
        
        # Initialize worklist
        worklist.update(criteria.nodes)
        for var in criteria.variables:
            worklist.update(self.var_def_sites[var])
        
        while worklist:
            node = worklist.pop()
            if node in sliced_nodes:
                continue
                
            sliced_nodes.add(node)
            
            # Add dependencies based on criteria
            if criteria.include_data:
                worklist.update(self._get_data_dependencies(node))
            if criteria.include_control:
                worklist.update(self._get_control_dependencies(node))
                
        return self._induce_subgraph(sliced_nodes)

    def _compute_forward_slice(self, criteria: SlicingCriteria) -> nx.DiGraph:
        """Compute a forward slice."""
        sliced_nodes = set()
        worklist = set()
        
        # Initialize worklist
        worklist.update(criteria.nodes)
        for var in criteria.variables:
            worklist.update(self.var_use_sites[var])
        
        while worklist:
            node = worklist.pop()
            if node in sliced_nodes:
                continue
                
            sliced_nodes.add(node)
            
            # Add dependents based on criteria
            if criteria.include_data:
                worklist.update(self._get_data_dependents(node))
            if criteria.include_control:
                worklist.update(self._get_control_dependents(node))
                
        return self._induce_subgraph(sliced_nodes)

    def _compute_thin_slice(self, criteria: SlicingCriteria) -> nx.DiGraph:
        """
        Compute a thin slice (data dependencies only, no transitive flows).
        """
        sliced_nodes = set()
        
        # Get direct data dependencies
        for node in criteria.nodes:
            sliced_nodes.add(node)
            sliced_nodes.update(self._get_direct_data_dependencies(node))
        
        for var in criteria.variables:
            def_sites = self.var_def_sites[var]
            use_sites = self.var_use_sites[var]
            sliced_nodes.update(def_sites)
            sliced_nodes.update(use_sites)
        
        return self._induce_subgraph(sliced_nodes)

    def _compute_composite_slice(self, criteria: SlicingCriteria,
                               slice_type: SliceType) -> nx.DiGraph:
        """Compute union or intersection of multiple slices."""
        node_sets = []
        
        # Compute individual slices for each node
        for node in criteria.nodes:
            single_criteria = SlicingCriteria(
                nodes={node},
                variables=set(),
                include_control=criteria.include_control,
                include_data=criteria.include_data
            )
            sliced_pdg = self.compute_slice(single_criteria, SliceType.BACKWARD)
            node_sets.append(set(sliced_pdg.nodes()))
        
        # Combine results based on slice type
        if slice_type == SliceType.UNION:
            combined_nodes = set().union(*node_sets)
        else:  # INTERSECTION
            combined_nodes = set.intersection(*node_sets)
            
        return self._induce_subgraph(combined_nodes)

    def _get_data_dependencies(self, node: str) -> Set[str]:
        """Get all nodes that the given node data-depends on."""
        deps = set()
        for pred, _, data in self.pdg.in_edges(node, data=True):
            if data.get('type') == 'data_dependency':
                deps.add(pred)
        return deps

    def _get_control_dependencies(self, node: str) -> Set[str]:
        """Get all nodes that the given node control-depends on."""
        deps = set()
        for pred, _, data in self.pdg.in_edges(node, data=True):
            if data.get('type') == 'control_dependency':
                deps.add(pred)
        return deps

    def _get_data_dependents(self, node: str) -> Set[str]:
        """Get all nodes that data-depend on the given node."""
        deps = set()
        for _, succ, data in self.pdg.out_edges(node, data=True):
            if data.get('type') == 'data_dependency':
                deps.add(succ)
        return deps

    def _get_control_dependents(self, node: str) -> Set[str]:
        """Get all nodes that control-depend on the given node."""
        deps = set()
        for _, succ, data in self.pdg.out_edges(node, data=True):
            if data.get('type') == 'control_dependency':
                deps.add(succ)
        return deps

    def _get_direct_data_dependencies(self, node: str) -> Set[str]:
        """Get only direct data dependencies (no transitive closure)."""
        return {pred for pred, _, data in self.pdg.in_edges(node, data=True)
                if data.get('type') == 'data_dependency'}

    def _calculate_slice_complexity(self, sliced_pdg: nx.DiGraph) -> int:
        """Calculate complexity of a slice."""
        complexity = 0
        for node, data in sliced_pdg.nodes(data=True):
            if data.get('type') in ('if', 'while', 'for'):
                complexity += 1
            elif data.get('type') == 'call':
                complexity += 2
        return complexity

    def _induce_subgraph(self, nodes: Set[str]) -> nx.DiGraph:
        """Create a subgraph from the given nodes, preserving edge attributes."""
        return self.pdg.subgraph(nodes).copy()

    def _make_cache_key(self, criteria: SlicingCriteria,
                       slice_type: SliceType) -> tuple:
        """Create a cache key for the given criteria and slice type."""
        return (
            frozenset(criteria.nodes),
            frozenset(criteria.variables),
            criteria.line_range,
            criteria.include_control,
            criteria.include_data,
            slice_type
        )

# Utility functions
def compute_slice(pdg: nx.DiGraph, node: str,
                 backward: bool = True,
                 criteria: Optional[SlicingCriteria] = None) -> nx.DiGraph:
    """
    Convenience function to compute a program slice.
    """
    slicer = ProgramSlicer(pdg)
    if criteria is None:
        criteria = SlicingCriteria(
            nodes={node},
            variables=set()
        )
    return slicer.compute_slice(
        criteria,
        SliceType.BACKWARD if backward else SliceType.FORWARD
    )