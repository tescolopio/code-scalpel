"""
Comprehensive tests for MCP Resource endpoints.

Tests cover all 5 resource URIs:
- scalpel://project/call-graph
- scalpel://project/dependencies  
- scalpel://project/structure
- scalpel://version
- scalpel://capabilities

Each test verifies:
1. Resource function exists and is callable
2. Returns valid data format (JSON or text)
3. Data structure matches documented schema
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the server module to access resource functions
import code_scalpel.mcp.server as server_module
from code_scalpel.mcp.server import (
    get_project_call_graph,
    get_project_dependencies,
    get_project_structure,
    get_version,
    get_capabilities,
)


class TestProjectCallGraphResource:
    """Tests for scalpel://project/call-graph resource."""

    def test_resource_function_exists(self):
        """Verify the resource function exists."""
        assert callable(get_project_call_graph)

    def test_returns_valid_json(self):
        """Verify resource returns valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple Python file
            Path(tmpdir, "main.py").write_text("""
def caller():
    helper()

def helper():
    print("helping")
""")
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_call_graph()
                
                # Should be valid JSON
                data = json.loads(result)
                assert isinstance(data, dict)

    def test_call_graph_structure(self):
        """Verify call graph has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "example.py").write_text("""
def main():
    process()
    validate()

def process():
    print("processing")

def validate():
    pass
""")
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_call_graph()
                data = json.loads(result)
                
                # Should have file:function keys
                assert any("main" in key for key in data.keys())
                
                # Values should be lists
                for key, value in data.items():
                    assert isinstance(value, list)

    def test_call_graph_empty_project(self):
        """Verify empty project returns empty graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_call_graph()
                data = json.loads(result)
                assert data == {}


class TestProjectDependenciesResource:
    """Tests for scalpel://project/dependencies resource."""

    def test_resource_function_exists(self):
        """Verify the resource function exists."""
        assert callable(get_project_dependencies)

    def test_returns_valid_json(self):
        """Verify resource returns valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_dependencies()
                data = json.loads(result)
                assert isinstance(data, dict)

    def test_parses_requirements_txt(self):
        """Verify requirements.txt is parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("""
requests>=2.28.0
flask>=2.0.0
""")
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_dependencies()
                data = json.loads(result)
                
                assert "python" in data
                names = [d["name"] for d in data["python"]]
                assert "requests" in names
                assert "flask" in names

    def test_parses_pyproject_toml(self):
        """Verify pyproject.toml is parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text("""
[project]
dependencies = [
    "pydantic>=2.0.0",
    "click>=8.0.0",
]
""")
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_dependencies()
                data = json.loads(result)
                
                assert "python" in data
                names = [d["name"] for d in data["python"]]
                assert "pydantic" in names

    def test_parses_package_json(self):
        """Verify package.json is parsed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg = {
                "dependencies": {"express": "^4.18.0"},
                "devDependencies": {"jest": "^29.0.0"}
            }
            Path(tmpdir, "package.json").write_text(json.dumps(pkg))
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_dependencies()
                data = json.loads(result)
                
                assert "javascript" in data
                names = [d["name"] for d in data["javascript"]]
                assert "express" in names
                assert "jest" in names

    def test_empty_project_returns_empty_dict(self):
        """Verify empty project returns empty deps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_dependencies()
                data = json.loads(result)
                assert data == {}


class TestProjectStructureResource:
    """Tests for scalpel://project/structure resource."""

    def test_resource_function_exists(self):
        """Verify the resource function exists."""
        assert callable(get_project_structure)

    def test_returns_valid_json(self):
        """Verify resource returns valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                assert isinstance(data, dict)

    def test_structure_has_required_fields(self):
        """Verify structure has name, type, and children."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file.txt").touch()
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                
                assert "name" in data
                assert "type" in data
                assert data["type"] == "directory"
                assert "children" in data

    def test_includes_files_and_directories(self):
        """Verify files and directories are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "file.py").touch()
            subdir = Path(tmpdir, "subdir")
            subdir.mkdir()
            Path(subdir, "nested.py").touch()
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                
                children_names = [c["name"] for c in data["children"]]
                assert "file.py" in children_names
                assert "subdir" in children_names
                
                # Find subdir and check its children
                subdir_node = next(c for c in data["children"] if c["name"] == "subdir")
                assert subdir_node["type"] == "directory"
                nested_names = [c["name"] for c in subdir_node["children"]]
                assert "nested.py" in nested_names

    def test_excludes_hidden_files(self):
        """Verify hidden files are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, ".hidden").touch()
            Path(tmpdir, "visible.py").touch()
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                
                children_names = [c["name"] for c in data["children"]]
                assert ".hidden" not in children_names
                assert "visible.py" in children_names

    def test_excludes_pycache(self):
        """Verify __pycache__ is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir, "__pycache__")
            cache_dir.mkdir()
            Path(cache_dir, "module.pyc").touch()
            Path(tmpdir, "module.py").touch()
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                
                children_names = [c["name"] for c in data["children"]]
                assert "__pycache__" not in children_names
                assert "module.py" in children_names

    def test_excludes_venv(self):
        """Verify venv directory is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir, "venv")
            venv_dir.mkdir()
            Path(tmpdir, "main.py").touch()
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                
                children_names = [c["name"] for c in data["children"]]
                assert "venv" not in children_names

    def test_excludes_node_modules(self):
        """Verify node_modules is excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            node_dir = Path(tmpdir, "node_modules")
            node_dir.mkdir()
            Path(tmpdir, "index.js").touch()
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                result = get_project_structure()
                data = json.loads(result)
                
                children_names = [c["name"] for c in data["children"]]
                assert "node_modules" not in children_names


class TestVersionResource:
    """Tests for scalpel://version resource."""

    def test_resource_function_exists(self):
        """Verify the resource function exists."""
        assert callable(get_version)

    def test_returns_string(self):
        """Verify resource returns a string."""
        result = get_version()
        assert isinstance(result, str)

    def test_contains_version_number(self):
        """Verify version string contains version number."""
        result = get_version()
        # Should contain "Code Scalpel v" followed by version
        assert "Code Scalpel v" in result

    def test_contains_feature_list(self):
        """Verify version info contains features."""
        result = get_version()
        assert "AST Analysis" in result or "AST" in result
        assert "Security" in result or "security" in result.lower()

    def test_contains_language_info(self):
        """Verify version info mentions supported languages."""
        result = get_version()
        assert "Python" in result


class TestCapabilitiesResource:
    """Tests for scalpel://capabilities resource."""

    def test_resource_function_exists(self):
        """Verify the resource function exists."""
        assert callable(get_capabilities)

    def test_returns_string(self):
        """Verify resource returns a string."""
        result = get_capabilities()
        assert isinstance(result, str)

    def test_contains_tools_section(self):
        """Verify capabilities lists available tools."""
        result = get_capabilities()
        # Should mention the main tools
        assert "analyze_code" in result or "analyze" in result.lower()

    def test_is_markdown_format(self):
        """Verify output is Markdown formatted."""
        result = get_capabilities()
        # Should have Markdown headers
        assert "#" in result


class TestResourceIntegration:
    """Integration tests for MCP resources."""

    def test_all_resources_work_together(self):
        """Test that all resources can be called in sequence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup a mini project
            Path(tmpdir, "main.py").write_text("def main(): pass")
            Path(tmpdir, "requirements.txt").write_text("requests>=2.0\n")
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                # Call all resources
                call_graph = get_project_call_graph()
                deps = get_project_dependencies()
                structure = get_project_structure()
                version = get_version()
                capabilities = get_capabilities()
                
                # All should return data
                assert json.loads(call_graph) is not None
                assert json.loads(deps) is not None
                assert json.loads(structure) is not None
                assert len(version) > 0
                assert len(capabilities) > 0

    def test_resources_handle_complex_project(self):
        """Test resources with a more complex project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create src directory
            src = Path(tmpdir, "src")
            src.mkdir()
            Path(src, "__init__.py").touch()
            Path(src, "core.py").write_text("""
from .utils import helper

def main():
    helper()
""")
            Path(src, "utils.py").write_text("""
def helper():
    pass
""")
            
            # Create tests directory
            tests = Path(tmpdir, "tests")
            tests.mkdir()
            Path(tests, "test_core.py").write_text("""
def test_main():
    pass
""")
            
            # Create config files
            Path(tmpdir, "pyproject.toml").write_text("""
[project]
name = "test-project"
dependencies = ["click>=8.0"]
""")
            
            with patch.object(server_module, "PROJECT_ROOT", Path(tmpdir)):
                # All resources should work
                call_graph = json.loads(get_project_call_graph())
                deps = json.loads(get_project_dependencies())
                structure = json.loads(get_project_structure())
                
                # Verify call graph has entries
                assert len(call_graph) > 0
                
                # Verify deps found
                assert "python" in deps
                
                # Verify structure has src and tests
                children = [c["name"] for c in structure["children"]]
                assert "src" in children
                assert "tests" in children
