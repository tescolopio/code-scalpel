import pytest
import networkx as nx
from code_scalpel.pdg_tools.visualizer import PDGVisualizer, VisualizationConfig, VisualizationFormat
from unittest.mock import MagicMock, patch

class TestPDGVisualizer:
    @pytest.fixture
    def simple_pdg(self):
        pdg = nx.DiGraph()
        pdg.add_node("1", type="function", name="test_func", lineno=1)
        pdg.add_node("2", type="assign", lineno=2)
        pdg.add_edge("1", "2", type="control_dependency")
        return pdg

    @pytest.fixture
    def visualizer(self):
        return PDGVisualizer()

    def test_init(self, visualizer):
        assert isinstance(visualizer.config, VisualizationConfig)
        assert visualizer._default_node_colors is not None

    @patch("code_scalpel.pdg_tools.visualizer.Digraph")
    def test_visualize(self, mock_digraph, visualizer, simple_pdg):
        mock_dot = MagicMock()
        mock_digraph.return_value = mock_dot
        
        visualizer.visualize(simple_pdg, "output")
        
        mock_dot.render.assert_called_once()
        mock_dot.node.assert_called()
        mock_dot.edge.assert_called()

    @patch("builtins.open", new_callable=MagicMock)
    def test_create_interactive_html(self, mock_open, visualizer, simple_pdg):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        visualizer.create_interactive_html(simple_pdg, "output.html")
        
        mock_open.assert_called_with("output.html", "w")
        mock_file.write.assert_called()
        # Check if D3 script is included
        args, _ = mock_file.write.call_args
        assert "d3.forceSimulation" in args[0]

    @patch("builtins.open", new_callable=MagicMock)
    @patch("code_scalpel.pdg_tools.visualizer.Digraph")
    def test_create_comparison_view(self, mock_digraph, mock_open, visualizer, simple_pdg):
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        mock_dot = MagicMock()
        mock_dot.pipe.return_value = b"fake_png_data"
        mock_digraph.return_value = mock_dot
        
        visualizer.create_comparison_view(simple_pdg, simple_pdg, "comparison.html")
        
        mock_open.assert_called_with("comparison.html", "w")
        mock_file.write.assert_called()
        # Check if base64 image is included
        args, _ = mock_file.write.call_args
        assert "data:image/png;base64" in args[0]

    def test_highlight_subgraph(self, visualizer):
        nodes = {"1", "2"}
        visualizer.highlight_subgraph(nodes, temporary=False)
        assert visualizer.config.highlight_nodes == nodes
        
        visualizer.highlight_subgraph({"3"}, temporary=True)
        assert visualizer._temp_highlights == {"3"}

    def test_wrap_text(self):
        text = "This is a long text that needs wrapping"
        wrapped = PDGVisualizer._wrap_text(text, 10)
        assert "\\n" in wrapped
        assert len(wrapped.split("\\n")) > 1

