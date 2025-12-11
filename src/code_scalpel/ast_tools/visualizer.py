import ast
import html
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import zip_longest
from typing import Optional

import networkx as nx
from graphviz import Digraph

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for AST visualization."""

    node_colors: dict[str, str] = None
    edge_colors: dict[str, str] = None
    node_shapes: dict[str, str] = None
    highlight_nodes: set[int] = None
    show_attributes: bool = True
    show_line_numbers: bool = True
    show_source_code: bool = True
    max_label_length: int = 50


class ASTVisualizer:
    """Advanced AST visualization with customizable styling and multiple output formats."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._default_colors = {
            "ast.FunctionDef": "#a8d5e5",
            "ast.ClassDef": "#95c8d8",
            "ast.Call": "#d1e8ef",
            "ast.Name": "#e8f4f8",
            "ast.Constant": "#f0f9fc",
        }
        self._default_shapes = {
            "ast.FunctionDef": "box",
            "ast.ClassDef": "box3d",
            "ast.Call": "ellipse",
            "ast.Name": "oval",
            "ast.Constant": "diamond",
        }

    def visualize(
        self,
        tree: ast.AST,
        output_file: str = "ast_visualization",
        format: str = "png",
        view: bool = True,
    ) -> None:
        """
        Create a visualization of the AST with advanced formatting.

        Args:
            tree: The AST to visualize
            output_file: Base name for the output file
            format: Output format (png, svg, pdf, etc.)
            view: Whether to open the visualization
        """
        try:
            dot = self._create_digraph()

            # Track node relationships for layout optimization
            self.node_relationships = defaultdict(set)

            # Create the visualization
            self._build_visualization(tree, dot)

            # Optimize layout
            self._optimize_layout(dot)

            # Save in requested format
            dot.render(output_file, format=format, view=view, cleanup=True)

            # Generate additional outputs if requested
            if self.config.show_source_code:
                self._generate_html_view(tree, output_file)
        except Exception as e:
            logger.error(f"Error visualizing AST: {str(e)}")
            raise

    def visualize_diff(
        self, tree1: ast.AST, tree2: ast.AST, output_file: str = "ast_diff"
    ) -> None:
        """Visualize differences between two ASTs."""
        try:
            dot = self._create_digraph()

            # Analyze differences
            changes = self._analyze_differences(tree1, tree2)

            # Create visualization with highlighted differences
            self._build_diff_visualization(tree1, tree2, changes, dot)

            dot.render(output_file, view=True)
        except Exception as e:
            logger.error(f"Error visualizing AST differences: {str(e)}")
            raise

    def create_interactive_visualization(
        self, tree: ast.AST, output_file: str = "interactive_ast.html"
    ) -> None:
        """Create an interactive HTML visualization using d3.js."""
        try:
            # Convert AST to networkx graph
            G = self._ast_to_networkx(tree)

            # Generate JSON data for d3.js
            json_data = self._generate_d3_json(G)

            # Create interactive HTML
            html_content = self._generate_interactive_html(json_data)

            # Save to file
            with open(output_file, "w") as f:
                f.write(html_content)
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}")
            raise

    def _create_digraph(self) -> Digraph:
        """Create and configure the Digraph object."""
        dot = Digraph(comment="Abstract Syntax Tree Visualization")
        dot.attr(rankdir="TB")
        dot.attr(splines="ortho")
        return dot

    def _build_visualization(
        self,
        node: ast.AST,
        dot: Digraph,
        parent_id: Optional[str] = None,
        edge_label: Optional[str] = None,
    ) -> str:
        """Recursively build the visualization."""
        node_id = str(id(node))

        # Create node label
        label = self._create_node_label(node)

        # Get node styling
        style = self._get_node_style(node)

        # Add node to graph
        dot.node(node_id, label, **style)

        # Add edge if there's a parent
        if parent_id:
            edge_style = self._get_edge_style(node)
            if edge_label:
                edge_style["label"] = edge_label
            dot.edge(parent_id, node_id, **edge_style)

        # Process child nodes
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for _idx, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        self._build_visualization(
                            item,
                            dot,
                            node_id,
                            edge_label=field if self.config.show_attributes else None,
                        )
            elif isinstance(value, ast.AST):
                self._build_visualization(
                    value,
                    dot,
                    node_id,
                    edge_label=field if self.config.show_attributes else None,
                )

        return node_id

    def _create_node_label(self, node: ast.AST) -> str:
        """Create a detailed node label."""
        label_parts = [node.__class__.__name__]

        if self.config.show_line_numbers and hasattr(node, "lineno"):
            label_parts.append(f"Line: {node.lineno}")

        # Add node-specific information
        if isinstance(node, ast.FunctionDef):
            label_parts.append(f"Function: {node.name}")
            if node.args.args:
                args = [arg.arg for arg in node.args.args]
                label_parts.append(f"Args: {', '.join(args)}")
        elif isinstance(node, ast.ClassDef):
            label_parts.append(f"Class: {node.name}")
            if node.bases:
                bases = [ast.unparse(base) for base in node.bases]
                label_parts.append(f"Bases: {', '.join(bases)}")
        elif isinstance(node, ast.Name):
            label_parts.append(f"Name: {node.id}")
        elif isinstance(node, ast.Constant):
            value = str(node.value)
            if len(value) > self.config.max_label_length:
                value = value[: self.config.max_label_length] + "..."
            label_parts.append(f"Value: {value}")

        return "\n".join(label_parts)

    def _get_node_style(self, node: ast.AST) -> dict[str, str]:
        """Get styling for a node."""
        node_type = node.__class__.__name__
        style = {
            "shape": self._default_shapes.get(f"ast.{node_type}", "box"),
            "style": "filled",
            "fillcolor": self._default_colors.get(f"ast.{node_type}", "#ffffff"),
        }

        # Apply custom styling if configured
        if self.config.node_colors and f"ast.{node_type}" in self.config.node_colors:
            style["fillcolor"] = self.config.node_colors[f"ast.{node_type}"]
        if self.config.node_shapes and f"ast.{node_type}" in self.config.node_shapes:
            style["shape"] = self.config.node_shapes[f"ast.{node_type}"]

        # Highlight node if configured
        if self.config.highlight_nodes and id(node) in self.config.highlight_nodes:
            style["penwidth"] = "3.0"
            style["color"] = "red"

        return style

    def _get_edge_style(self, node: ast.AST) -> dict[str, str]:
        """Get styling for an edge."""
        edge_type = node.__class__.__name__
        style = {"arrowhead": "normal", "color": "#666666"}

        if self.config.edge_colors and f"ast.{edge_type}" in self.config.edge_colors:
            style["color"] = self.config.edge_colors[f"ast.{edge_type}"]

        return style

    def _analyze_differences(
        self, tree1: ast.AST, tree2: ast.AST
    ) -> dict[str, set[int]]:
        """Analyze differences between two ASTs."""
        changes = {"added": set(), "removed": set(), "modified": set()}

        def compare_nodes(node1: Optional[ast.AST], node2: Optional[ast.AST]) -> None:
            if node1 is None and node2 is not None:
                changes["added"].add(id(node2))
            elif node1 is not None and node2 is None:
                changes["removed"].add(id(node1))
            elif not self._nodes_equal(node1, node2):
                changes["modified"].add(id(node1))
                changes["modified"].add(id(node2))

            # Compare children
            children1 = list(ast.iter_child_nodes(node1)) if node1 else []
            children2 = list(ast.iter_child_nodes(node2)) if node2 else []

            for child1, child2 in zip_longest(children1, children2):
                compare_nodes(child1, child2)

        compare_nodes(tree1, tree2)
        return changes

    def _generate_html_view(self, tree: ast.AST, base_filename: str) -> None:
        """Generate an HTML view with source code and AST side by side."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AST Visualization</title>
            <style>
                .container {{ display: flex; }}
                .source, .ast {{ flex: 1; padding: 20px; }}
                pre {{ background: #f5f5f5; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="source">
                    <h2>Source Code</h2>
                    <pre>{html.escape(ast.unparse(tree))}</pre>
                </div>
                <div class="ast">
                    <h2>AST Visualization</h2>
                    <img src="{base_filename}.png" alt="AST visualization">
                </div>
            </div>
        </body>
        </html>
        """

        with open(f"{base_filename}.html", "w") as f:
            f.write(html_content)

    def _optimize_layout(self, dot: Digraph) -> None:
        """Optimize the graph layout."""
        # Add invisible edges for better layout
        for _parent, children in self.node_relationships.items():
            if len(children) > 1:
                sorted_children = sorted(children)
                for i in range(len(sorted_children) - 1):
                    dot.edge(
                        str(sorted_children[i]),
                        str(sorted_children[i + 1]),
                        style="invis",
                    )

    def _nodes_equal(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two AST nodes are equal."""
        if type(node1) is not type(node2):
            return False
        for field in node1._fields:
            if getattr(node1, field) != getattr(node2, field):
                return False
        return True

    def _ast_to_networkx(self, tree: ast.AST) -> nx.DiGraph:
        """Convert AST to networkx graph."""
        G = nx.DiGraph()

        def add_edges(node: ast.AST, parent: Optional[ast.AST] = None):
            node_id = id(node)
            G.add_node(node_id, label=self._create_node_label(node))
            if parent:
                G.add_edge(id(parent), node_id)
            for child in ast.iter_child_nodes(node):
                add_edges(child, node)

        add_edges(tree)
        return G

    def _generate_d3_json(self, G: nx.DiGraph) -> str:
        """Generate JSON data for d3.js visualization."""
        data = {
            "nodes": [{"id": n, "label": G.nodes[n]["label"]} for n in G.nodes],
            "edges": [{"source": u, "target": v} for u, v in G.edges],
        }
        return json.dumps(data, indent=2)

    def _generate_interactive_html(self, json_data: str) -> str:
        """Generate interactive HTML using d3.js."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive AST Visualization</title>
            <script src="https://d3js.org/d3.v5.min.js"></script>
            <style>
                .node {{ stroke: #fff; stroke-width: 1.5px; }}
                .link {{ stroke: #999; stroke-opacity: 0.6; }}
            </style>
        </head>
        <body>
            <script>
                var graph = {json_data};
                var width = 960, height = 600;
                var svg = d3.select("body").append("svg")
                    .attr("width", width)
                    .attr("height", height);
                var simulation = d3.forceSimulation()
                    .force("link", d3.forceLink().id(function(d) {{ return d.id; }}))
                    .force("charge", d3.forceManyBody())
                    .force("center", d3.forceCenter(width / 2, height / 2));
                var link = svg.append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(graph.edges)
                    .enter().append("line")
                    .attr("class", "link");
                var node = svg.append("g")
                    .attr("class", "nodes")
                    .selectAll("circle")
                    .data(graph.nodes)
                    .enter().append("circle")
                    .attr("class", "node")
                    .attr("r", 5)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));
                node.append("title")
                    .text(function(d) {{ return d.label; }});
                simulation
                    .nodes(graph.nodes)
                    .on("tick", ticked);
                simulation.force("link")
                    .links(graph.edges);
                function ticked() {{
                    link
                        .attr("x1", function(d) {{ return d.source.x; }})
                        .attr("y1", function(d) {{ return d.source.y; }})
                        .attr("x2", function(d) {{ return d.target.x; }})
                        .attr("y2", function(d) {{ return d.target.y; }});
                    node
                        .attr("cx", function(d) {{ return d.x; }})
                        .attr("cy", function(d) {{ return d.y; }});
                }}
                function dragstarted(d) {{
                    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
                    d.fx = d.x;
                    d.fy = d.y;
                }}
                function dragged(d) {{
                    d.fx = d3.event.x;
                    d.fy = d3.event.y;
                }}
                function dragended(d) {{
                    if (!d3.event.active) simulation.alphaTarget(0);
                    d.fx = null;
                    d.fy = null;
                }}
            </script>
        </body>
        </html>
        """
