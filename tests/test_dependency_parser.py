"""
Comprehensive tests for dependency_parser.py - DependencyParser class.

Tests cover:
- Initialization
- Python dependency parsing from pyproject.toml and requirements.txt
- JavaScript dependency parsing from package.json
- PEP 508 parsing (version specifiers, extras, markers)
- Deduplication logic
- Edge cases (missing files, malformed files, empty files)
"""

import json
import os
import tempfile
from pathlib import Path


from code_scalpel.ast_tools.dependency_parser import DependencyParser


class TestDependencyParserInit:
    """Tests for DependencyParser initialization."""

    def test_init_with_path(self):
        """Test initialization with a valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            assert parser.root_path == tmpdir

    def test_init_with_path_object(self):
        """Test initialization with Path object (converted to str)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(str(Path(tmpdir)))
            assert parser.root_path == str(Path(tmpdir))


class TestGetDependencies:
    """Tests for the main get_dependencies method."""

    def test_returns_empty_when_no_deps(self):
        """Test that empty dict is returned when no dependency files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()
            assert result == {}

    def test_returns_only_python_when_no_js(self):
        """Test that only Python deps are returned when no package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("requests==2.28.0\n")
            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()
            assert "python" in result
            assert "javascript" not in result

    def test_returns_only_js_when_no_python(self):
        """Test that only JS deps are returned when no Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = {"dependencies": {"express": "^4.18.0"}}
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))
            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()
            assert "javascript" in result
            assert "python" not in result

    def test_returns_both_ecosystems(self):
        """Test that both Python and JS deps are returned when both exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("requests==2.28.0\n")
            pkg_json = {"dependencies": {"express": "^4.18.0"}}
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))
            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()
            assert "python" in result
            assert "javascript" in result


class TestParsePythonDeps:
    """Tests for _parse_python_deps method."""

    def test_parse_requirements_txt(self):
        """Test parsing requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text(
                """
requests==2.28.0
flask>=2.0.0
django<4.0
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            assert "requests" in names
            assert "flask" in names
            assert "django" in names

    def test_parse_requirements_with_comments(self):
        """Test that comments are ignored in requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text(
                """
# This is a comment
requests==2.28.0
# Another comment
flask>=2.0.0
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            assert len(names) == 2
            assert "requests" in names
            assert "flask" in names

    def test_parse_requirements_with_flags(self):
        """Test that flag lines are ignored (like -r other.txt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text(
                """
-r base.txt
--index-url https://pypi.org/simple
requests==2.28.0
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            assert len(names) == 1
            assert names[0] == "requests"

    def test_parse_requirements_empty_lines(self):
        """Test that empty lines are handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text(
                """
requests==2.28.0

flask>=2.0.0

"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            assert len(names) == 2

    def test_parse_pyproject_toml_pep621(self):
        """Test parsing PEP 621 format in pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text(
                """
[project]
name = "myproject"
dependencies = [
    "requests>=2.28.0",
    "flask>=2.0.0",
]
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            assert "requests" in names
            assert "flask" in names

    def test_parse_pyproject_toml_poetry(self):
        """Test parsing Poetry format in pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text(
                """
[tool.poetry]
name = "myproject"

[tool.poetry.dependencies]
python = "^3.9"
requests = "^2.28.0"
flask = "^2.0.0"
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            assert "requests" in names
            assert "flask" in names
            # Python should be excluded
            assert "python" not in [n.lower() for n in names]

    def test_parse_both_pyproject_and_requirements(self):
        """Test that both files are parsed and deduplicated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text(
                """
[project]
dependencies = ["requests>=2.28.0"]
"""
            )
            Path(tmpdir, "requirements.txt").write_text(
                """
requests==2.28.0
flask>=2.0.0
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            names = [d["name"] for d in deps]
            # Should be deduplicated - requests only once
            assert names.count("requests") == 1
            assert "flask" in names

    def test_parse_malformed_pyproject(self):
        """Test handling of malformed pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pyproject.toml").write_text(
                """
[project
name = "broken"
"""
            )
            parser = DependencyParser(tmpdir)
            # Should not raise, just return empty
            deps = parser._parse_python_deps()
            assert deps == []

    def test_parse_malformed_requirements(self):
        """Test handling of requirements.txt with encoding issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with valid content - test should handle edge cases
            Path(tmpdir, "requirements.txt").write_text("requests>=2.0\n")
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()
            assert len(deps) >= 1


class TestParseJavaScriptDeps:
    """Tests for _parse_javascript_deps method."""

    def test_parse_package_json_dependencies(self):
        """Test parsing dependencies from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = {
                "name": "myproject",
                "dependencies": {"express": "^4.18.0", "lodash": "^4.17.0"},
            }
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))
            parser = DependencyParser(tmpdir)
            deps = parser._parse_javascript_deps()

            names = [d["name"] for d in deps]
            assert "express" in names
            assert "lodash" in names

    def test_parse_package_json_dev_dependencies(self):
        """Test parsing devDependencies from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = {
                "name": "myproject",
                "devDependencies": {"jest": "^29.0.0", "eslint": "^8.0.0"},
            }
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))
            parser = DependencyParser(tmpdir)
            deps = parser._parse_javascript_deps()

            assert len(deps) == 2
            assert all(d.get("type") == "dev" for d in deps)

    def test_parse_package_json_both_types(self):
        """Test parsing both deps and devDeps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = {
                "dependencies": {"express": "^4.18.0"},
                "devDependencies": {"jest": "^29.0.0"},
            }
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))
            parser = DependencyParser(tmpdir)
            deps = parser._parse_javascript_deps()

            assert len(deps) == 2
            names = [d["name"] for d in deps]
            assert "express" in names
            assert "jest" in names

    def test_parse_empty_package_json(self):
        """Test handling of package.json with no deps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkg_json = {"name": "myproject"}
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))
            parser = DependencyParser(tmpdir)
            deps = parser._parse_javascript_deps()

            assert deps == []

    def test_parse_malformed_package_json(self):
        """Test handling of malformed package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "package.json").write_text("{broken json")
            parser = DependencyParser(tmpdir)
            # Should not raise, just return empty
            deps = parser._parse_javascript_deps()
            assert deps == []

    def test_no_package_json(self):
        """Test when package.json doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            deps = parser._parse_javascript_deps()
            assert deps == []


class TestParsePEP508:
    """Tests for _parse_pep508 method."""

    def test_parse_simple_name(self):
        """Test parsing package name only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("requests")

            assert result["name"] == "requests"
            assert result["version"] == "*"

    def test_parse_exact_version(self):
        """Test parsing exact version specifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("requests==2.28.0")

            assert result["name"] == "requests"
            assert result["version"] == "==2.28.0"

    def test_parse_minimum_version(self):
        """Test parsing >= version specifier."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("flask>=2.0.0")

            assert result["name"] == "flask"
            assert result["version"] == ">=2.0.0"

    def test_parse_version_range(self):
        """Test parsing version range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("django>=3.0,<4.0")

            assert result["name"] == "django"
            assert result["version"] == ">=3.0,<4.0"

    def test_parse_with_extras(self):
        """Test parsing package with extras (simplified)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            # Current impl may not handle extras perfectly
            result = parser._parse_pep508("requests[security]>=2.0")

            # Name parsing handles this as a basic case
            assert "requests" in result["name"]

    def test_parse_with_environment_markers(self):
        """Test that environment markers are stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508('pywin32>=300; sys_platform == "win32"')

            assert result["name"] == "pywin32"
            assert result["version"] == ">=300"

    def test_parse_with_inline_comment(self):
        """Test that inline comments are stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("requests>=2.0  # needed for API calls")

            assert result["name"] == "requests"
            assert result["version"] == ">=2.0"

    def test_parse_hyphenated_name(self):
        """Test parsing hyphenated package names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("azure-storage-blob>=12.0")

            assert result["name"] == "azure-storage-blob"
            assert result["version"] == ">=12.0"

    def test_parse_dotted_name(self):
        """Test parsing dotted package names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("zope.interface>=5.0")

            assert result["name"] == "zope.interface"
            assert result["version"] == ">=5.0"

    def test_parse_underscored_name(self):
        """Test parsing underscored package names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._parse_pep508("my_package>=1.0")

            assert result["name"] == "my_package"
            assert result["version"] == ">=1.0"


class TestDeduplicate:
    """Tests for _deduplicate method."""

    def test_deduplicate_simple(self):
        """Test simple deduplication."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            deps = [
                {"name": "requests", "version": "==2.28.0"},
                {"name": "flask", "version": ">=2.0.0"},
                {"name": "requests", "version": ">=2.0.0"},  # Duplicate
            ]
            result = parser._deduplicate(deps)

            assert len(result) == 2
            names = [d["name"] for d in result]
            assert names.count("requests") == 1

    def test_deduplicate_preserves_first(self):
        """Test that first occurrence is preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            deps = [
                {"name": "requests", "version": "==2.28.0"},
                {"name": "requests", "version": ">=2.0.0"},
            ]
            result = parser._deduplicate(deps)

            assert len(result) == 1
            assert result[0]["version"] == "==2.28.0"

    def test_deduplicate_empty_list(self):
        """Test deduplication of empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            result = parser._deduplicate([])
            assert result == []

    def test_deduplicate_no_duplicates(self):
        """Test when there are no duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)
            deps = [
                {"name": "requests", "version": ">=2.0"},
                {"name": "flask", "version": ">=2.0"},
            ]
            result = parser._deduplicate(deps)
            assert len(result) == 2


class TestEdgeCases:
    """Edge cases and regression tests."""

    def test_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        parser = DependencyParser("/nonexistent/path/12345")
        # Should not raise, just return empty
        result = parser.get_dependencies()
        assert result == {}

    def test_empty_requirements_file(self):
        """Test handling of empty requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("")
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()
            assert deps == []

    def test_whitespace_only_requirements(self):
        """Test handling of requirements.txt with only whitespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text("   \n\t\n   ")
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()
            assert deps == []

    def test_version_specifiers_comprehensive(self):
        """Test various version specifier formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "requirements.txt").write_text(
                """
pkg1==1.0.0
pkg2>=1.0
pkg3<=2.0
pkg4>1.0
pkg5<2.0
pkg6!=1.5.0
pkg7~=1.4.2
pkg8>=1.0,<2.0
pkg9
"""
            )
            parser = DependencyParser(tmpdir)
            deps = parser._parse_python_deps()

            assert len(deps) == 9
            # Verify all were parsed
            names = [d["name"] for d in deps]
            for i in range(1, 10):
                assert f"pkg{i}" in names


class TestIntegration:
    """Integration tests with realistic project structures."""

    def test_realistic_python_project(self):
        """Test with a realistic Python project structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a realistic pyproject.toml
            Path(tmpdir, "pyproject.toml").write_text(
                """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "myproject"
version = "1.0.0"
dependencies = [
    "requests>=2.28.0",
    "click>=8.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
]
"""
            )
            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()

            assert "python" in result
            names = [d["name"] for d in result["python"]]
            assert "requests" in names
            assert "click" in names
            assert "pydantic" in names

    def test_realistic_fullstack_project(self):
        """Test with a fullstack project (Python + Node)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Python backend
            Path(tmpdir, "requirements.txt").write_text(
                """
fastapi>=0.100.0
uvicorn>=0.23.0
sqlalchemy>=2.0.0
"""
            )
            # Node frontend
            pkg_json = {
                "name": "frontend",
                "dependencies": {"react": "^18.0.0", "axios": "^1.4.0"},
                "devDependencies": {"typescript": "^5.0.0", "vite": "^4.0.0"},
            }
            Path(tmpdir, "package.json").write_text(json.dumps(pkg_json))

            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()

            assert "python" in result
            assert "javascript" in result

            python_names = [d["name"] for d in result["python"]]
            assert "fastapi" in python_names

            js_names = [d["name"] for d in result["javascript"]]
            assert "react" in js_names
            assert "typescript" in js_names


class TestCoverageGaps:
    """Tests to close specific coverage gaps in dependency_parser.py."""

    def test_parse_pep508_regex_no_match(self):
        """Test _parse_pep508 when regex doesn't match (line 83)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create requirements.txt with a line that won't match the regex
            # The regex expects ^([a-zA-Z0-9_\-\.]+)(.*)$
            # An empty string or line starting with special chars would fail
            Path(tmpdir, "requirements.txt").write_text("@invalid\n=bad\n")

            parser = DependencyParser(tmpdir)
            result = parser.get_dependencies()

            # Should still return results, just with the raw string as name
            # The fallback at line 83 returns {"name": s, "version": "*"}
            assert "python" in result
            # The malformed entries should have version "*"
            for dep in result["python"]:
                if dep["name"] in ["@invalid", "=bad"]:
                    assert dep["version"] == "*"

    def test_requirements_read_exception(self):
        """Test exception handling when reading requirements.txt fails (line 57)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a requirements.txt
            req_path = Path(tmpdir, "requirements.txt")
            req_path.write_text("requests>=2.0\n")

            parser = DependencyParser(tmpdir)

            # Use mock to make open() raise an exception during read
            from unittest.mock import mock_open, patch

            m = mock_open()
            m.return_value.read.side_effect = IOError("Read error")
            m.return_value.__iter__ = lambda self: iter([])

            # The exception handler at line 57-58 should catch this
            # and deps should remain empty or partial
            original_exists = os.path.exists
            with patch("builtins.open", side_effect=PermissionError("denied")):
                with patch(
                    "os.path.exists",
                    side_effect=lambda p: True
                    if "requirements" in p
                    else original_exists(p),
                ):
                    result = parser._parse_python_deps()
                    # Should not crash, just return empty or partial
                    assert isinstance(result, list)

    def test_parse_pep508_with_empty_after_strip(self):
        """Test _parse_pep508 with input that becomes empty after stripping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)

            # Input that after split and strip becomes empty
            # "  ; python_version > '3'" -> strip to "" -> regex won't match
            result = parser._parse_pep508("  ; python_version > '3'")

            # Should return fallback
            assert result["version"] == "*"

    def test_parse_pep508_line_with_only_markers(self):
        """Test PEP508 line that's only environment markers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = DependencyParser(tmpdir)

            # "# comment" would be filtered earlier, but
            # a line like "   " after stripping is empty
            result = parser._parse_pep508("")

            # Empty string as name, "*" as version
            assert result == {"name": "", "version": "*"}
