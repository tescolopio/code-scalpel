import ast
import copy
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import networkx as nx


class TransformationType(Enum):
    """Types of PDG transformations."""

    REMOVE = "remove"
    INSERT = "insert"
    REPLACE = "replace"
    MERGE = "merge"
    SPLIT = "split"
    OPTIMIZE = "optimize"
    REFACTOR = "refactor"


@dataclass
class TransformationResult:
    """Result of a PDG transformation."""

    success: bool
    modified_nodes: set[str]
    added_nodes: set[str]
    removed_nodes: set[str]
    description: str
    metrics: dict[str, Any]


class PDGTransformer:
    """Advanced PDG transformer with optimization and refactoring capabilities."""

    def __init__(self, pdg: nx.DiGraph):
        self.pdg = pdg
        self.history: list[tuple[TransformationType, TransformationResult]] = []
        self.node_counter = defaultdict(int)

    def transform(
        self, transformation_type: TransformationType, **kwargs
    ) -> TransformationResult:
        """
        Apply a transformation to the PDG.

        Args:
            transformation_type: Type of transformation to apply
            **kwargs: Transformation-specific parameters

        Returns:
            TransformationResult object
        """
        result = None

        if transformation_type == TransformationType.OPTIMIZE:
            result = self.optimize_pdg(**kwargs)
        elif transformation_type == TransformationType.REFACTOR:
            result = self.refactor_pdg(**kwargs)
        else:
            result = self._apply_basic_transformation(transformation_type, **kwargs)

        if result.success:
            self.history.append((transformation_type, result))

        return result

    def optimize_pdg(
        self,
        optimize_dead_code: bool = True,
        optimize_constants: bool = True,
        optimize_loops: bool = True,
    ) -> TransformationResult:
        """Perform various PDG optimizations."""
        modified_nodes = set()
        added_nodes = set()
        removed_nodes = set()
        metrics = defaultdict(int)

        # Dead code elimination
        if optimize_dead_code:
            dead_code_result = self._eliminate_dead_code()
            removed_nodes.update(dead_code_result.removed_nodes)
            metrics["dead_code_removed"] = len(dead_code_result.removed_nodes)

        # Constant propagation
        if optimize_constants:
            const_prop_result = self._propagate_constants()
            modified_nodes.update(const_prop_result.modified_nodes)
            metrics["constants_propagated"] = len(const_prop_result.modified_nodes)

        # Loop optimization
        if optimize_loops:
            loop_opt_result = self._optimize_loops()
            modified_nodes.update(loop_opt_result.modified_nodes)
            added_nodes.update(loop_opt_result.added_nodes)
            metrics["loops_optimized"] = len(loop_opt_result.modified_nodes)

        return TransformationResult(
            success=True,
            modified_nodes=modified_nodes,
            added_nodes=added_nodes,
            removed_nodes=removed_nodes,
            description="Applied PDG optimizations",
            metrics=dict(metrics),
        )

    def refactor_pdg(self, refactoring_type: str, **kwargs) -> TransformationResult:
        """Apply refactoring transformations to the PDG."""
        if refactoring_type == "extract_method":
            return self._extract_method(**kwargs)
        elif refactoring_type == "inline_method":
            return self._inline_method(**kwargs)
        elif refactoring_type == "move_node":
            return self._move_node(**kwargs)
        else:
            raise ValueError(f"Unknown refactoring type: {refactoring_type}")

    def merge_nodes(
        self, nodes: list[str], new_node_id: str, new_data: dict
    ) -> TransformationResult:
        """Merge multiple nodes into a single node."""
        if not all(node in self.pdg for node in nodes):
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description="Some nodes to merge do not exist",
                metrics={},
            )

        # Collect all incoming and outgoing edges
        incoming_edges = []
        outgoing_edges = []
        for node in nodes:
            incoming_edges.extend(self.pdg.in_edges(node, data=True))
            outgoing_edges.extend(self.pdg.out_edges(node, data=True))

        # Add new node
        self.pdg.add_node(new_node_id, **new_data)

        # Reconnect edges
        for pred, _, data in incoming_edges:
            if pred not in nodes:
                self.pdg.add_edge(pred, new_node_id, **data)

        for _, succ, data in outgoing_edges:
            if succ not in nodes:
                self.pdg.add_edge(new_node_id, succ, **data)

        # Remove old nodes
        self.pdg.remove_nodes_from(nodes)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes={new_node_id},
            removed_nodes=set(nodes),
            description=f"Merged {len(nodes)} nodes into {new_node_id}",
            metrics={"nodes_merged": len(nodes)},
        )

    def split_node(self, node: str, split_data: list[dict]) -> TransformationResult:
        """Split a node into multiple nodes."""
        if node not in self.pdg:
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Node {node} does not exist",
                metrics={},
            )

        new_nodes = []
        for idx, data in enumerate(split_data):
            new_node_id = f"{node}_split_{idx}"
            self.pdg.add_node(new_node_id, **data)
            new_nodes.append(new_node_id)

        # Connect split nodes sequentially
        for idx in range(len(new_nodes) - 1):
            self.pdg.add_edge(
                new_nodes[idx], new_nodes[idx + 1], type="control_dependency"
            )

        # Reconnect dependencies
        self._reconnect_split_dependencies(node, new_nodes)

        # Remove original node
        self.pdg.remove_node(node)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes=set(new_nodes),
            removed_nodes={node},
            description=f"Split node {node} into {len(new_nodes)} nodes",
            metrics={"nodes_created": len(new_nodes)},
        )

    def insert_node(
        self, node: str, data: dict, dependencies: list = None
    ) -> TransformationResult:
        """Insert a new node into the PDG."""
        if node in self.pdg:
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Node {node} already exists",
                metrics={},
            )

        self.pdg.add_node(node, **data)
        if dependencies:
            for predecessor, dep_type in dependencies:
                self.pdg.add_edge(predecessor, node, type=dep_type)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes={node},
            removed_nodes=set(),
            description=f"Inserted node {node}",
            metrics={"nodes_inserted": 1},
        )

    def remove_node(self, node: str) -> TransformationResult:
        """Remove a node from the PDG."""
        if node not in self.pdg:
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Node {node} does not exist",
                metrics={},
            )

        self.pdg.remove_node(node)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes=set(),
            removed_nodes={node},
            description=f"Removed node {node}",
            metrics={"nodes_removed": 1},
        )

    def replace_node(
        self, old_node: str, new_node: str, data: dict
    ) -> TransformationResult:
        """Replace an existing node with a new node in the PDG."""
        if old_node not in self.pdg:
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Node {old_node} does not exist",
                metrics={},
            )

        predecessors = list(self.pdg.predecessors(old_node))
        successors = list(self.pdg.successors(old_node))
        self.pdg.remove_node(old_node)
        self.pdg.add_node(new_node, **data)
        for predecessor in predecessors:
            edge_data = self.pdg.edges[predecessor, old_node]
            self.pdg.add_edge(predecessor, new_node, **edge_data)
        for successor in successors:
            edge_data = self.pdg.edges[old_node, successor]
            self.pdg.add_edge(new_node, successor, **edge_data)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes={new_node},
            removed_nodes={old_node},
            description=f"Replaced node {old_node} with {new_node}",
            metrics={"nodes_replaced": 1},
        )

    def _eliminate_dead_code(self) -> TransformationResult:
        """Eliminate dead code from the PDG."""
        removed_nodes = set()

        # Find nodes with no outgoing data dependencies
        for node in list(self.pdg.nodes()):
            if not self._has_effect(node):
                removed_nodes.add(node)
                self.pdg.remove_node(node)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes=set(),
            removed_nodes=removed_nodes,
            description=f"Removed {len(removed_nodes)} dead code nodes",
            metrics={"nodes_removed": len(removed_nodes)},
        )

    def _propagate_constants(self) -> TransformationResult:
        """Propagate constant values through the PDG."""
        modified_nodes = set()
        constant_values = {}

        # Find constant assignments
        for node, data in self.pdg.nodes(data=True):
            if data.get("type") == "assign" and self._is_constant_value(
                data.get("value")
            ):
                constant_values[data["target"]] = data["value"]

        # Propagate constants
        for node, data in self.pdg.nodes(data=True):
            if self._uses_constants(data, constant_values):
                new_data = self._replace_constants(data, constant_values)
                self.pdg.nodes[node].update(new_data)
                modified_nodes.add(node)

        return TransformationResult(
            success=True,
            modified_nodes=modified_nodes,
            added_nodes=set(),
            removed_nodes=set(),
            description=f"Propagated {len(constant_values)} constants",
            metrics={"constants_propagated": len(constant_values)},
        )

    def _optimize_loops(self) -> TransformationResult:
        """Optimize loop structures in the PDG."""
        modified_nodes = set()
        added_nodes = set()

        # Find loop nodes
        loop_nodes = [
            node
            for node, data in self.pdg.nodes(data=True)
            if data.get("type") in ("for", "while")
        ]

        for loop_node in loop_nodes:
            # Find loop-invariant code
            invariant_nodes = self._find_loop_invariant_nodes(loop_node)

            if invariant_nodes:
                # Move invariant nodes outside the loop
                new_nodes = self._move_nodes_before_loop(loop_node, invariant_nodes)
                modified_nodes.add(loop_node)
                added_nodes.update(new_nodes)

        return TransformationResult(
            success=True,
            modified_nodes=modified_nodes,
            added_nodes=added_nodes,
            removed_nodes=set(),
            description=f"Optimized {len(loop_nodes)} loops",
            metrics={"loops_optimized": len(loop_nodes)},
        )

    def _extract_method(
        self, nodes: list[str], method_name: str, parameters: list[str] = None
    ) -> TransformationResult:
        """Extract a set of nodes into a new method."""
        if not all(node in self.pdg for node in nodes):
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description="Some nodes to extract do not exist",
                metrics={},
            )

        # Create new method node
        method_node = self._create_method_node(method_name, nodes, parameters)

        # Update dependencies
        self._update_method_dependencies(method_node, nodes)

        # Remove original nodes
        self.pdg.remove_nodes_from(nodes)

        return TransformationResult(
            success=True,
            modified_nodes=set(),
            added_nodes={method_node},
            removed_nodes=set(nodes),
            description=f"Extracted method {method_name}",
            metrics={"nodes_extracted": len(nodes)},
        )

    def _inline_method(self, method_node: str) -> TransformationResult:
        """Inline a method by replacing a function call with the body of the function."""
        if method_node not in self.pdg:
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Method node {method_node} does not exist",
                metrics={},
            )

        method_data = self.pdg.nodes[method_node]
        if method_data.get("type") != "function":
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Node {method_node} is not a function",
                metrics={},
            )

        body_nodes = method_data.get("body_nodes", [])
        call_sites = [
            node
            for node, data in self.pdg.nodes(data=True)
            if data.get("type") == "call" and data.get("function") == method_node
        ]

        for call_site in call_sites:
            for body_node in body_nodes:
                new_node_id = f"{call_site}_inlined_{body_node}"
                self.pdg.add_node(new_node_id, **self.pdg.nodes[body_node])
                self.pdg.add_edge(call_site, new_node_id, type="control_dependency")

        self.pdg.remove_node(method_node)

        return TransformationResult(
            success=True,
            modified_nodes=set(call_sites),
            added_nodes=set(body_nodes),
            removed_nodes={method_node},
            description=f"Inlined method {method_node}",
            metrics={"nodes_inlined": len(body_nodes)},
        )

    def _move_node(
        self, node: str, new_predecessors: list[str], new_successors: list[str]
    ) -> TransformationResult:
        """Move a node to a different location in the PDG."""
        if node not in self.pdg:
            return TransformationResult(
                success=False,
                modified_nodes=set(),
                added_nodes=set(),
                removed_nodes=set(),
                description=f"Node {node} does not exist",
                metrics={},
            )

        old_predecessors = list(self.pdg.predecessors(node))
        old_successors = list(self.pdg.successors(node))

        for predecessor in old_predecessors:
            self.pdg.remove_edge(predecessor, node)
        for successor in old_successors:
            self.pdg.remove_edge(node, successor)

        for predecessor in new_predecessors:
            self.pdg.add_edge(predecessor, node, type="control_dependency")
        for successor in new_successors:
            self.pdg.add_edge(node, successor, type="control_dependency")

        return TransformationResult(
            success=True,
            modified_nodes={node},
            added_nodes=set(),
            removed_nodes=set(),
            description=f"Moved node {node}",
            metrics={"nodes_moved": 1},
        )

    def _has_effect(self, node: str) -> bool:
        """Check if a node has any effect on the program output."""
        return any(
            edge[2].get("type") == "data_dependency"
            for edge in self.pdg.out_edges(node, data=True)
        )

    def _is_constant_value(self, value: Any) -> bool:
        """Check if a value is a constant."""
        try:
            ast.literal_eval(str(value))
            return True
        except:
            return False

    def _uses_constants(self, data: dict, constants: dict[str, Any]) -> bool:
        """Check if node data uses any known constants."""
        if "uses" in data:
            return any(var in constants for var in data["uses"])
        return False

    def _replace_constants(self, data: dict, constants: dict[str, Any]) -> dict:
        """Replace constant references in node data."""
        new_data = copy.deepcopy(data)
        if "value" in new_data:
            for const_name, const_value in constants.items():
                new_data["value"] = new_data["value"].replace(
                    const_name, str(const_value)
                )
        return new_data

    def _find_loop_invariant_nodes(self, loop_node: str) -> set[str]:
        """Find nodes that are invariant within a loop."""
        loop_body = self._get_loop_body(loop_node)
        invariant_nodes = set()

        for node in loop_body:
            if self._is_loop_invariant(node, loop_node):
                invariant_nodes.add(node)

        return invariant_nodes

    def _is_loop_invariant(self, node: str, loop_node: str) -> bool:
        """Check if a node is invariant within a loop."""
        node_data = self.pdg.nodes[node]

        # Node must not depend on loop variables
        if self._depends_on_loop_variables(node, loop_node):
            return False

        # Node must not have side effects
        return not node_data.get("has_side_effects", False)

    def _move_nodes_before_loop(self, loop_node: str, nodes: set[str]) -> set[str]:
        """Move nodes to execute before a loop."""
        new_nodes = set()

        for node in nodes:
            new_node_id = f"{node}_hoisted"
            new_nodes.add(new_node_id)

            # Copy node with dependencies
            self.pdg.add_node(new_node_id, **self.pdg.nodes[node])
            for pred, _, data in self.pdg.in_edges(node, data=True):
                if pred not in nodes:
                    self.pdg.add_edge(pred, new_node_id, **data)

        return new_nodes

    def _get_loop_body(self, loop_node: str) -> set[str]:
        """
        Get the set of nodes that form the body of a loop.

        Parameters:
        loop_node (str): The identifier of the loop node in the PDG.

        Returns:
        Set[str]: A set of node identifiers representing the body nodes of the loop.
        """
        loop_body = set()
        if loop_node in self.pdg:
            loop_data = self.pdg.nodes[loop_node]
            if "body_nodes" in loop_data:
                loop_body.update(loop_data["body_nodes"])
        return loop_body
