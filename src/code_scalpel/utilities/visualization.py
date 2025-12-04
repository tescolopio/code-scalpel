import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import networkx as nx
import plotly.graph_objects as go
import pyvis.network as net
from graphviz import Digraph


class VisualizationType(Enum):
    """Types of visualization outputs."""

    STATIC = "static"
    INTERACTIVE = "interactive"
    NOTEBOOK = "notebook"


class GraphType(Enum):
    """Types of graphs to visualize."""

    PDG = "pdg"
    CFG = "cfg"
    CALL_GRAPH = "call_graph"
    AST = "ast"
    CUSTOM = "custom"


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""

    type: VisualizationType = VisualizationType.STATIC
    output_format: str = "png"
    node_colors: dict[str, str] = None
    edge_colors: dict[str, str] = None
    node_shapes: dict[str, str] = None
    edge_styles: dict[str, str] = None
    font_sizes: dict[str, int] = None
    layout: str = "dot"
    highlight_nodes: set[str] = None
    highlight_edges: set[tuple] = None
    show_labels: bool = True
    label_wrap_length: int = 20
    physics_enabled: bool = True
    enable_clustering: bool = False
    dark_mode: bool = False


class GraphVisualizer:
    """Advanced graph visualizer with multiple output formats."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._init_styles()
        self._setup_logging()

    def visualize(
        self,
        graph: nx.DiGraph,
        output_file: str,
        graph_type: GraphType = GraphType.CUSTOM,
    ) -> None:
        """
        Create graph visualization.

        Args:
            graph: NetworkX graph to visualize
            output_file: Output file path
            graph_type: Type of graph
        """
        if self.config.type == VisualizationType.STATIC:
            self._create_static_visualization(graph, output_file, graph_type)
        elif self.config.type == VisualizationType.INTERACTIVE:
            self._create_interactive_visualization(graph, output_file, graph_type)
        else:
            self._create_notebook_visualization(graph, output_file, graph_type)

    def create_subgraph_view(
        self, graph: nx.DiGraph, nodes: set[str], context_depth: int = 1
    ) -> nx.DiGraph:
        """Create view of subgraph with context."""
        # Get neighborhood of selected nodes
        context_nodes = set(nodes)
        for _ in range(context_depth):
            for node in list(context_nodes):
                context_nodes.update(graph.predecessors(node))
                context_nodes.update(graph.successors(node))

        return graph.subgraph(context_nodes)

    def highlight_path(
        self, graph: nx.DiGraph, path: list[str], color: str = "#ff0000"
    ) -> None:
        """Highlight a path in the graph."""
        if not self.config.highlight_nodes:
            self.config.highlight_nodes = set()
        if not self.config.highlight_edges:
            self.config.highlight_edges = set()

        self.config.highlight_nodes.update(path)

        # Add edges along path
        for i in range(len(path) - 1):
            self.config.highlight_edges.add((path[i], path[i + 1]))

    def apply_layout(self, graph: nx.DiGraph, layout_type: str) -> dict[str, tuple]:
        """Apply layout algorithm to graph."""
        if layout_type == "force":
            return nx.spring_layout(graph)
        elif layout_type == "circular":
            return nx.circular_layout(graph)
        elif layout_type == "spectral":
            return nx.spectral_layout(graph)
        elif layout_type == "spiral":
            return self._create_spiral_layout(graph)
        else:
            return nx.kamada_kawai_layout(graph)

    def _create_static_visualization(
        self, graph: nx.DiGraph, output_file: str, graph_type: GraphType
    ) -> None:
        """Create static visualization using Graphviz."""
        dot = Digraph(comment=f"{graph_type.value.upper()} Visualization")
        dot.attr(rankdir="TB")

        # Add nodes
        for node in graph.nodes:
            attrs = graph.nodes[node]
            label = self._create_node_label(node, attrs)
            style = self._get_node_style(node, attrs, graph_type)
            dot.node(str(node), label, **style)

        # Add edges
        for src, dst, data in graph.edges(data=True):
            style = self._get_edge_style(data, graph_type)
            dot.edge(str(src), str(dst), **style)

        # Set layout
        dot.attr(layout=self.config.layout)

        # Render
        dot.render(output_file, format=self.config.output_format, cleanup=True)

    def _create_interactive_visualization(
        self, graph: nx.DiGraph, output_file: str, graph_type: GraphType
    ) -> None:
        """Create interactive visualization using Pyvis."""
        network = net.Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff" if not self.config.dark_mode else "#000000",
            font_color="#000000" if not self.config.dark_mode else "#ffffff",
        )

        # Configure physics
        network.force_atlas_2based()
        network.show_buttons(filter_=["physics"])

        # Add nodes
        for node in graph.nodes:
            attrs = graph.nodes[node]
            network.add_node(
                node,
                label=self._create_node_label(node, attrs),
                **self._get_pyvis_node_style(node, attrs, graph_type),
            )

        # Add edges
        for src, dst, data in graph.edges(data=True):
            network.add_edge(src, dst, **self._get_pyvis_edge_style(data, graph_type))

        # Enable clustering if configured
        if self.config.enable_clustering:
            network.set_options(
                """
                var options = {
                    "nodes": {
                        "scaling": {
                            "min": 10,
                            "max": 30
                        }
                    },
                    "edges": {
                        "smooth": {
                            "type": "continuous"
                        }
                    },
                    "physics": {
                        "barnesHut": {
                            "gravitationalConstant": -80000,
                            "springLength": 250,
                            "springConstant": 0.001
                        }
                    },
                    "interaction": {
                        "hover": true
                    }
                }
            """
            )

        # Save to HTML
        network.save_graph(f"{output_file}.html")

    def _create_notebook_visualization(
        self, graph: nx.DiGraph, output_file: str, graph_type: GraphType
    ) -> None:
        """Create visualization for Jupyter notebooks using Plotly."""
        # Get layout
        layout = self.apply_layout(graph, self.config.layout)

        # Create edge trace
        edge_x = []
        edge_y = []
        for src, dst in graph.edges():
            x0, y0 = layout[src]
            x1, y1 = layout[dst]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line={"width": 0.5, "color": "#888"},
            hoverinfo="none",
            mode="lines",
        )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        for node in graph.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            attrs = graph.nodes[node]
            node_text.append(self._create_node_label(node, attrs))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_text,
            textposition="bottom center",
            marker={"size": 10, "line_width": 2},
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin={"b": 0, "l": 0, "r": 0, "t": 0},
                xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
                yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            ),
        )

        # Save
        fig.write_html(f"{output_file}.html")

    def _create_node_label(self, node: str, attrs: dict) -> str:
        """Create formatted node label."""
        parts = [str(node)]

        if self.config.show_labels:
            if "type" in attrs:
                parts.append(f"Type: {attrs['type']}")
            if "value" in attrs:
                value = str(attrs["value"])
                if len(value) > self.config.label_wrap_length:
                    value = value[: self.config.label_wrap_length] + "..."
                parts.append(f"Value: {value}")
            if "condition" in attrs:
                condition = str(attrs["condition"])
                if len(condition) > self.config.label_wrap_length:
                    condition = condition[: self.config.label_wrap_length] + "..."
                parts.append(f"Condition: {condition}")

        return "\n".join(parts)

    def _get_node_style(self, node: str, attrs: dict, graph_type: GraphType) -> dict:
        """Get node style attributes."""
        node_type = attrs.get("type", "default")

        style = {
            "shape": self._get_node_shape(node_type),
            "style": "filled",
            "fillcolor": self._get_node_color(node_type),
            "fontsize": str(self._get_font_size(node_type)),
        }

        # Apply highlighting
        if self.config.highlight_nodes and node in self.config.highlight_nodes:
            style.update({"penwidth": "3.0", "color": "#ff0000"})

        return style

    def _get_edge_style(self, data: dict, graph_type: GraphType) -> dict:
        """Get edge style attributes."""
        edge_type = data.get("type", "default")

        style = {
            "label": edge_type if self.config.show_labels else "",
            "color": self._get_edge_color(edge_type, graph_type),
            "style": self._get_edge_line_style(edge_type),
        }

        return style

    def _init_styles(self):
        """Initialize default styles."""
        if not self.config.node_colors:
            self.config.node_colors = {
                "variable": "#a8d5e5",
                "function": "#95c8d8",
                "condition": "#d1e8ef",
                "assignment": "#e8f4f8",
                "default": "#ffffff",
            }

        if not self.config.node_shapes:
            self.config.node_shapes = {
                "variable": "ellipse",
                "function": "box",
                "condition": "diamond",
                "assignment": "box",
                "default": "ellipse",
            }

    def _get_node_shape(self, node_type: str) -> str:
        """Get shape for node type."""
        return self.config.node_shapes.get(
            node_type, self.config.node_shapes["default"]
        )

    def _get_node_color(self, node_type: str) -> str:
        """Get color for node type."""
        return self.config.node_colors.get(
            node_type, self.config.node_colors["default"]
        )

    def _get_font_size(self, node_type: str) -> int:
        """Get font size for node type."""
        return self.config.font_sizes.get(node_type, 12)

    def _get_edge_color(self, edge_type: str, graph_type: GraphType) -> str:
        """Get color for edge type."""
        if graph_type == GraphType.PDG:
            return "blue" if edge_type == "data_dependency" else "red"
        return self.config.edge_colors.get(edge_type, "#888888")

    def _get_edge_line_style(self, edge_type: str) -> str:
        """Get line style for edge type."""
        return self.config.edge_styles.get(edge_type, "solid")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("GraphVisualizer")


def create_visualizer(config: Optional[VisualizationConfig] = None) -> GraphVisualizer:
    """Create a new graph visualizer instance."""
    return GraphVisualizer(config)
