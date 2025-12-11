import json
import base64
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import networkx as nx
from graphviz import Digraph


class VisualizationFormat(Enum):
    """Supported visualization output formats."""

    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    DOT = "dot"


@dataclass
class VisualizationConfig:
    """Configuration for PDG visualization."""

    node_colors: dict[str, str] = None
    edge_colors: dict[str, str] = None
    node_shapes: dict[str, str] = None
    edge_styles: dict[str, str] = None
    font_sizes: dict[str, int] = None
    layout: str = "dot"
    dpi: int = 300
    highlight_nodes: set[str] = None
    highlight_edges: set[tuple[str, str]] = None
    cluster_groups: dict[str, list[str]] = None
    show_attributes: bool = True
    label_wrapping: int = 30


class PDGVisualizer:
    """Advanced PDG visualizer with customization and multiple output formats."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self._init_default_styles()

    def _init_default_styles(self):
        """Initialize default visualization styles."""
        self._default_node_colors = {
            "assign": "#a8d5e5",
            "if": "#95c8d8",
            "while": "#95c8d8",
            "for": "#95c8d8",
            "call": "#d1e8ef",
            "return": "#e8f4f8",
            "function": "#89cff0",
            "class": "#7ab8d3",
            "module": "#6ca6c1",
        }

        self._default_node_shapes = {
            "assign": "box",
            "if": "diamond",
            "while": "diamond",
            "for": "diamond",
            "call": "ellipse",
            "return": "box",
            "function": "folder",
            "class": "tab",
            "module": "component",
        }

        self._default_edge_styles = {
            "data_dependency": {"color": "blue", "style": "solid", "penwidth": "1.0"},
            "control_dependency": {"color": "red", "style": "solid", "penwidth": "1.0"},
            "call_dependency": {"color": "green", "style": "dashed", "penwidth": "1.0"},
            "parameter_dependency": {
                "color": "purple",
                "style": "dotted",
                "penwidth": "1.0",
            },
        }

    def visualize(
        self,
        pdg: nx.DiGraph,
        output_file: str,
        format: VisualizationFormat = VisualizationFormat.PNG,
        view: bool = True,
    ) -> None:
        """
        Create a visualization of the PDG.

        Args:
            pdg: The Program Dependence Graph
            output_file: Base name for output file
            format: Output format
            view: Whether to display the visualization
        """
        dot = self._create_digraph()

        # Add clusters if specified
        if self.config.cluster_groups:
            self._add_clusters(dot, pdg)
        else:
            self._add_nodes(dot, pdg)

        self._add_edges(dot, pdg)

        # Apply layout settings
        self._apply_layout_settings(dot)

        # Render the visualization
        dot.render(output_file, format=format.value, view=view, cleanup=True)

    def create_interactive_html(
        self, pdg: nx.DiGraph, output_file: str, include_details: bool = True
    ) -> None:
        """Create an interactive HTML visualization using d3.js."""
        # Convert PDG to D3 compatible format
        graph_data = self._convert_to_d3_format(pdg)

        # Generate HTML with embedded visualization
        html_content = self._generate_interactive_html(graph_data, include_details)

        # Write to file
        with open(output_file, "w") as f:
            f.write(html_content)

    def highlight_subgraph(
        self, nodes: set[str], color: str = "#ff0000", temporary: bool = True
    ) -> None:
        """Highlight a subgraph of nodes."""
        if temporary:
            self._temp_highlights = nodes
        else:
            if self.config.highlight_nodes is None:
                self.config.highlight_nodes = set()
            self.config.highlight_nodes.update(nodes)

    def create_comparison_view(
        self,
        pdg1: nx.DiGraph,
        pdg2: nx.DiGraph,
        output_file: str,
        highlight_differences: bool = True,
    ) -> None:
        """Create a side-by-side comparison visualization."""
        # Create HTML with two visualizations
        html_content = self._generate_comparison_html(pdg1, pdg2, highlight_differences)

        with open(output_file, "w") as f:
            f.write(html_content)

    def _create_digraph(self) -> Digraph:
        """Create and configure the Digraph object."""
        dot = Digraph(comment="Program Dependence Graph")
        dot.attr(rankdir="TB")
        dot.attr(splines="ortho")
        dot.attr("node", fontname="Arial")
        dot.attr("edge", fontname="Arial")
        return dot

    def _add_nodes(self, dot: Digraph, pdg: nx.DiGraph) -> None:
        """Add nodes to the visualization."""
        for node in pdg.nodes:
            attrs = pdg.nodes[node]

            # Create node label
            label = self._create_node_label(node, attrs)

            # Get node style
            style = self._get_node_style(node, attrs)

            # Add node
            dot.node(str(node), label, **style)

    def _add_clusters(self, dot: Digraph, pdg: nx.DiGraph) -> None:
        """Add clustered nodes to the visualization."""
        for cluster_name, nodes in self.config.cluster_groups.items():
            with dot.subgraph(name=f"cluster_{cluster_name}") as c:
                c.attr(label=cluster_name)
                c.attr(style="rounded")
                c.attr(color="gray")

                for node in nodes:
                    if node in pdg.nodes:
                        attrs = pdg.nodes[node]
                        label = self._create_node_label(node, attrs)
                        style = self._get_node_style(node, attrs)
                        c.node(str(node), label, **style)

    def _add_edges(self, dot: Digraph, pdg: nx.DiGraph) -> None:
        """Add edges to the visualization."""
        for edge in pdg.edges:
            # Get edge style
            style = self._get_edge_style(pdg.edges[edge])

            # Add edge
            dot.edge(str(edge[0]), str(edge[1]), **style)

    def _create_node_label(self, node: str, attrs: dict) -> str:
        """Create a formatted node label."""
        label_parts = [str(node)]

        if self.config.show_attributes:
            node_type = attrs.get("type", "")
            label_parts.append(f"Type: {node_type}")

            # Add relevant attributes based on node type
            if "condition" in attrs:
                label_parts.append(
                    self._wrap_text(
                        f"Condition: {attrs['condition']}", self.config.label_wrapping
                    )
                )
            elif "value" in attrs:
                label_parts.append(
                    self._wrap_text(
                        f"Value: {attrs['value']}", self.config.label_wrapping
                    )
                )

            # Add line numbers if available
            if "lineno" in attrs:
                label_parts.append(f"Line: {attrs['lineno']}")

        return "\n".join(label_parts)

    def _get_node_style(self, node: str, attrs: dict) -> dict[str, str]:
        """Get styling for a node."""
        node_type = attrs.get("type", "default")

        style = {
            "shape": self._get_node_shape(node_type),
            "style": "filled",
            "fillcolor": self._get_node_color(node_type),
            "fontsize": str(self._get_font_size(node_type)),
        }

        # Apply highlighting if configured
        if self.config.highlight_nodes and node in self.config.highlight_nodes:
            style.update({"penwidth": "3.0", "color": "red"})

        return style

    def _get_edge_style(self, edge_attrs: dict) -> dict[str, str]:
        """Get styling for an edge."""
        edge_type = edge_attrs.get("type", "default")

        # Get base style
        style = self._default_edge_styles.get(edge_type, {}).copy()

        # Apply custom styling if configured
        if self.config.edge_styles and edge_type in self.config.edge_styles:
            style.update(self.config.edge_styles[edge_type])

        return style

    def _get_node_color(self, node_type: str) -> str:
        """Get color for a node type."""
        if self.config.node_colors and node_type in self.config.node_colors:
            return self.config.node_colors[node_type]
        return self._default_node_colors.get(node_type, "#ffffff")

    def _get_node_shape(self, node_type: str) -> str:
        """Get shape for a node type."""
        if self.config.node_shapes and node_type in self.config.node_shapes:
            return self.config.node_shapes[node_type]
        return self._default_node_shapes.get(node_type, "box")

    def _get_font_size(self, node_type: str) -> int:
        """Get font size for a node type."""
        if self.config.font_sizes and node_type in self.config.font_sizes:
            return self.config.font_sizes[node_type]
        return 10

    def _apply_layout_settings(self, dot: Digraph) -> None:
        """Apply layout settings to the visualization."""
        dot.attr(layout=self.config.layout)
        dot.attr(dpi=str(self.config.dpi))

        # Additional layout settings based on the chosen layout
        if self.config.layout == "dot":
            dot.attr(rankdir="TB")
            dot.attr(ranksep="0.5")
            dot.attr(nodesep="0.5")
        elif self.config.layout == "neato":
            dot.attr(overlap="false")
            dot.attr(splines="true")

    def _convert_to_d3_format(self, pdg: nx.DiGraph) -> dict:
        """Convert PDG to D3.js compatible format."""
        return {
            "nodes": [
                {
                    "id": node,
                    "type": pdg.nodes[node].get("type", "unknown"),
                    "data": pdg.nodes[node],
                }
                for node in pdg.nodes
            ],
            "links": [
                {
                    "source": source,
                    "target": target,
                    "type": pdg.edges[source, target].get("type", "unknown"),
                }
                for source, target in pdg.edges
            ],
        }

    def _generate_interactive_html(
        self, graph_data: dict, include_details: bool
    ) -> str:
        """Generate HTML content for interactive visualization."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Interactive PDG Visualization</title>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <style>
                .node {{
                    cursor: pointer;
                }}
                .link {{
                    stroke-width: 1.5px;
                }}
                .node-details {{
                    position: fixed;
                    right: 10px;
                    top: 10px;
                    background: white;
                    padding: 10px;
                    border: 1px solid #ccc;
                    display: none;
                }}
            </style>
        </head>
        <body>
            <div id="graph"></div>
            {self._generate_details_panel() if include_details else ''}
            <script>
                const graphData = {json.dumps(graph_data)};
                {self._get_d3_script()}
            </script>
        </body>
        </html>
        """

    def _generate_comparison_html(
        self, pdg1: nx.DiGraph, pdg2: nx.DiGraph, highlight_differences: bool
    ) -> str:
        """Generate HTML for comparison visualization."""
        graph1_data = self._convert_to_d3_format(pdg1)
        graph2_data = self._convert_to_d3_format(pdg2)

        if highlight_differences:
            self._highlight_graph_differences(graph1_data, graph2_data)

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PDG Comparison</title>
            <style>
                .container {{
                    display: flex;
                    justify-content: space-between;
                }}
                .graph {{
                    flex: 1;
                    margin: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="graph">
                    <h2>Original PDG</h2>
                    <img src="data:image/png;base64,{self._generate_base64_image(pdg1)}">
                </div>
                <div class="graph">
                    <h2>Modified PDG</h2>
                    <img src="data:image/png;base64,{self._generate_base64_image(pdg2)}">
                </div>
            </div>
        </body>
        </html>
        """

    @staticmethod
    def _wrap_text(text: str, width: int) -> str:
        """Wrap text to specified width."""
        if len(text) <= width:
            return text

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        return "\\n".join(lines)

    def _generate_details_panel(self) -> str:
        """Generate HTML for the details panel."""
        return """
        <div id="details" class="node-details">
            <h3>Node Details</h3>
            <div id="node-content">Select a node to view details</div>
        </div>
        """

    def _get_d3_script(self) -> str:
        """Generate D3.js script for interactive visualization."""
        return """
        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        // Arrow marker
        svg.append("defs").selectAll("marker")
            .data(["end"])
            .enter().append("marker")
            .attr("id", "end")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 15)
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("fill", "#999");

        const link = svg.append("g")
            .selectAll("line")
            .data(graphData.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke", "#999")
            .attr("marker-end", "url(#end)");

        const node = svg.append("g")
            .selectAll("g")
            .data(graphData.nodes)
            .enter().append("g")
            .attr("class", "node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        node.append("circle")
            .attr("r", 10)
            .attr("fill", d => getNodeColor(d.type));

        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.id);

        node.on("click", (event, d) => {
            const details = document.getElementById("details");
            const content = document.getElementById("node-content");
            details.style.display = "block";
            
            let html = `<strong>ID:</strong> ${d.id}<br>`;
            html += `<strong>Type:</strong> ${d.type}<br>`;
            
            for (const [key, value] of Object.entries(d.data)) {
                if (key !== "type") {
                    html += `<strong>${key}:</strong> ${value}<br>`;
                }
            }
            content.innerHTML = html;
        });

        simulation.on("tick", () => {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("transform", d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

        function getNodeColor(type) {
            const colors = {
                "assign": "#a8d5e5",
                "if": "#95c8d8",
                "while": "#95c8d8",
                "function": "#89cff0",
                "call": "#d1e8ef"
            };
            return colors[type] || "#cccccc";
        }
        """

    def _highlight_graph_differences(self, graph1_data: dict, graph2_data: dict) -> None:
        """Highlight differences between two graphs."""
        nodes1 = {n["id"] for n in graph1_data["nodes"]}
        nodes2 = {n["id"] for n in graph2_data["nodes"]}
        
        # Mark added/removed nodes
        for node in graph1_data["nodes"]:
            if node["id"] not in nodes2:
                node["data"]["_status"] = "removed"
                
        for node in graph2_data["nodes"]:
            if node["id"] not in nodes1:
                node["data"]["_status"] = "added"

    def _generate_base64_image(self, pdg: nx.DiGraph) -> str:
        """Generate base64 encoded image of the PDG."""
        dot = self._create_digraph()
        self._add_nodes(dot, pdg)
        self._add_edges(dot, pdg)
        self._apply_layout_settings(dot)
        
        # Render to PNG data
        png_data = dot.pipe(format="png")
        return base64.b64encode(png_data).decode("utf-8")

