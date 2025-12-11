import pytest
import networkx as nx
import json
import os
from code_scalpel.pdg_tools.utils import PDGUtils, NodeInfo, DependencyType, get_node_info, find_paths, export_pdg, import_pdg

class TestPDGUtils:
    @pytest.fixture
    def simple_pdg(self):
        pdg = nx.DiGraph()
        pdg.add_node("1", type="assign", value="x=1")
        pdg.add_node("2", type="assign", value="y=2")
        pdg.add_node("3", type="assign", value="z=x+y")
        pdg.add_node("func", type="function", name="my_func")
        
        pdg.add_edge("1", "3", type="data")
        pdg.add_edge("2", "3", type="data")
        pdg.add_edge("func", "1", type="control") # func contains 1
        pdg.add_edge("func", "2", type="control") # func contains 2
        pdg.add_edge("func", "3", type="control") # func contains 3
        
        return pdg

    def test_analyze_node(self, simple_pdg):
        info = PDGUtils.analyze_node(simple_pdg, "3")
        assert isinstance(info, NodeInfo)
        assert info.id == "3"
        assert info.type == "assign"
        assert len(info.dependencies) == 3 # 1->3 (data), 2->3 (data), func->3 (control)
        assert len(info.dependents) == 0
        assert info.scope == "function:my_func"

    def test_analyze_node_not_found(self, simple_pdg):
        with pytest.raises(ValueError):
            PDGUtils.analyze_node(simple_pdg, "nonexistent")

    def test_find_paths(self, simple_pdg):
        paths = PDGUtils.find_paths(simple_pdg, "func", "3")
        # Direct path: func -> 3
        # Indirect paths: func -> 1 -> 3, func -> 2 -> 3
        assert len(paths) >= 3
        assert ["func", "3"] in paths
        assert ["func", "1", "3"] in paths

    def test_find_paths_with_dep_types(self, simple_pdg):
        paths = PDGUtils.find_paths(simple_pdg, "func", "3", dep_types={"control"})
        # Only func -> 3 is purely control? 
        # func->1 is control, 1->3 is data. So func->1->3 is mixed.
        # func->3 is control.
        assert ["func", "3"] in paths
        assert ["func", "1", "3"] not in paths

    def test_calculate_node_metrics(self, simple_pdg):
        metrics = PDGUtils.calculate_node_metrics(simple_pdg, "3")
        assert metrics["in_degree"] == 3
        assert metrics["out_degree"] == 0
        assert "betweenness_centrality" in metrics

    def test_get_dependency_types(self, simple_pdg):
        deps = PDGUtils.get_dependency_types(simple_pdg, "3")
        assert deps["in_data"] == 2
        assert deps["in_control"] == 1

    def test_get_node_scope(self, simple_pdg):
        scope = PDGUtils.get_node_scope(simple_pdg, "1")
        assert scope == "function:my_func"
        
        scope_func = PDGUtils.get_node_scope(simple_pdg, "func")
        assert scope_func is None

    def test_find_common_ancestors(self, simple_pdg):
        ancestors = PDGUtils.find_common_ancestors(simple_pdg, ["1", "2"])
        assert "func" in ancestors

    def test_find_common_descendants(self, simple_pdg):
        descendants = PDGUtils.find_common_descendants(simple_pdg, ["1", "2"])
        assert "3" in descendants

    def test_get_subgraph_between(self, simple_pdg):
        subgraph = PDGUtils.get_subgraph_between(simple_pdg, ["func"], ["3"])
        assert "1" in subgraph.nodes
        assert "2" in subgraph.nodes
        assert "3" in subgraph.nodes
        assert "func" in subgraph.nodes

    def test_compute_node_hash(self):
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        assert PDGUtils.compute_node_hash(data1) == PDGUtils.compute_node_hash(data2)

    def test_find_similar_nodes(self, simple_pdg):
        # 1 and 2 are similar (both assign, similar deps)
        similar = PDGUtils.find_similar_nodes(simple_pdg, "1", similarity_threshold=0.5)
        assert "2" in similar

    def test_calculate_node_similarity(self, simple_pdg):
        sim = PDGUtils.calculate_node_similarity(simple_pdg, "1", "2")
        assert sim > 0.5

    def test_find_node_clusters(self, simple_pdg):
        clusters = PDGUtils.find_node_clusters(simple_pdg, min_size=2)
        # 1 and 2 should likely cluster
        found = False
        for cluster in clusters:
            if "1" in cluster and "2" in cluster:
                found = True
                break
        assert found

    def test_export_import_json(self, simple_pdg, tmp_path):
        filepath = tmp_path / "pdg.json"
        PDGUtils.export_to_json(simple_pdg, str(filepath))
        
        assert os.path.exists(filepath)
        
        imported_pdg = PDGUtils.import_from_json(str(filepath))
        assert len(imported_pdg.nodes) == len(simple_pdg.nodes)
        assert len(imported_pdg.edges) == len(simple_pdg.edges)
        assert imported_pdg.nodes["1"]["type"] == "assign"

    def test_calculate_dict_similarity(self):
        d1 = {"a": 10, "b": 5}
        d2 = {"a": 10, "b": 5}
        assert PDGUtils._calculate_dict_similarity(d1, d2) == 1.0
        
        d3 = {"a": 0, "b": 0}
        assert PDGUtils._calculate_dict_similarity(d1, d3) < 1.0

    def test_clean_for_json(self):
        data = {"a": {1, 2, 3}, "b": complex(1, 2)}
        cleaned = PDGUtils._clean_for_json(data)
        assert isinstance(cleaned["a"], list)
        assert isinstance(cleaned["b"], str)

    def test_convenience_functions(self, simple_pdg, tmp_path):
        info = get_node_info(simple_pdg, "3")
        assert info.id == "3"
        
        paths = find_paths(simple_pdg, "func", "3")
        assert len(paths) > 0
        
        filepath = tmp_path / "pdg_conv.json"
        export_pdg(simple_pdg, str(filepath))
        assert os.path.exists(filepath)
        
        imported = import_pdg(str(filepath))
        assert len(imported.nodes) > 0
