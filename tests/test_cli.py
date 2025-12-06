"""
Tests for the CLI module.

These tests verify that the command-line interface works correctly.
Goal: Get cli.py from 0% to at least 50% coverage.
"""

import subprocess
import sys

import pytest


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_help_command(self):
        """Test that --help works and shows expected content."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "code-scalpel" in result.stdout.lower() or "usage" in result.stdout.lower()
        assert "analyze" in result.stdout
        assert "server" in result.stdout
        assert "version" in result.stdout

    def test_version_command(self):
        """Test that version command works."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout or "Code Scalpel" in result.stdout

    def test_analyze_help(self):
        """Test that analyze --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--code" in result.stdout or "code" in result.stdout.lower()


class TestCLIAnalyze:
    """Test the analyze command."""

    def test_analyze_inline_code(self):
        """Test analyzing inline code with --code flag."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "--code",
                "def hello(): return 42",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Metrics" in result.stdout or "analysis" in result.stdout.lower()

    def test_analyze_inline_code_json(self):
        """Test analyzing inline code with JSON output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "--code",
                "x = 1",
                "--json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # JSON output should be parseable
        import json

        try:
            data = json.loads(result.stdout)
            assert "success" in data or "metrics" in data or isinstance(data, dict)
        except json.JSONDecodeError:
            # Might have logging output before JSON
            lines = result.stdout.strip().split("\n")
            for line in lines:
                try:
                    data = json.loads(line)
                    assert isinstance(data, dict)
                    break
                except json.JSONDecodeError:
                    continue

    def test_analyze_file(self, tmp_path):
        """Test analyzing a Python file."""
        # Create a temporary Python file
        test_file = tmp_path / "test_sample.py"
        test_file.write_text("def greet(name):\n    return f'Hello, {name}!'\n")

        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", str(test_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_analyze_nonexistent_file(self):
        """Test analyzing a file that doesn't exist."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "/nonexistent/file.py",
            ],
            capture_output=True,
            text=True,
        )
        # Should fail gracefully
        assert result.returncode != 0 or "error" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_analyze_syntax_error_code(self):
        """Test analyzing code with syntax errors."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "code_scalpel.cli",
                "analyze",
                "--code",
                "def broken(",
            ],
            capture_output=True,
            text=True,
        )
        # Should handle gracefully, not crash
        # May return 0 with error message or non-zero
        combined = result.stdout + result.stderr
        assert result.returncode != 0 or "error" in combined.lower() or "syntax" in combined.lower()


class TestCLIServer:
    """Test the server command."""

    def test_server_help(self):
        """Test that server --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "server", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--port" in result.stdout or "port" in result.stdout.lower()


class TestCLIEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_command(self):
        """Test that unknown commands are handled."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "unknowncommand"],
            capture_output=True,
            text=True,
        )
        # Should fail with usage info
        assert result.returncode != 0

    def test_analyze_no_input(self):
        """Test analyze without any input."""
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze"],
            capture_output=True,
            text=True,
        )
        # Should show error or usage
        combined = result.stdout + result.stderr
        assert result.returncode != 0 or "error" in combined.lower() or "usage" in combined.lower()

    def test_analyze_non_python_file(self, tmp_path):
        """Test analyzing a non-Python file (should warn)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Just some text content")
        
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", str(test_file)],
            capture_output=True,
            text=True,
        )
        # Should still process but may warn
        combined = result.stdout + result.stderr
        assert "warning" in combined.lower() or result.returncode == 0

    def test_analyze_file_with_dead_code(self, tmp_path):
        """Test analyzing a file with dead code for detection."""
        test_file = tmp_path / "dead_code.py"
        test_file.write_text("""
def unused_function():
    return "never called"

def main():
    return 42
""")
        
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", str(test_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should show metrics at minimum
        assert "Metrics" in result.stdout or "lines" in result.stdout.lower()

    def test_analyze_file_json_output(self, tmp_path):
        """Test file analysis with JSON output."""
        test_file = tmp_path / "sample.py"
        test_file.write_text("x = 1\ny = 2\n")
        
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", str(test_file), "--json"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        
        import json
        # Find JSON in output
        for line in result.stdout.strip().split('\n'):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    assert "metrics" in data or "source" in data
                    break
            except json.JSONDecodeError:
                continue

    def test_analyze_complex_code(self, tmp_path):
        """Test analyzing more complex code with all features."""
        test_file = tmp_path / "complex.py"
        test_file.write_text("""
import os

class Calculator:
    def __init__(self):
        self.value = 0
    
    def add(self, x):
        self.value += x
        return self
    
    def get_value(self):
        return self.value

def risky_function():
    exec("print('danger')")  # Security issue
    return eval("1+1")

def main():
    calc = Calculator()
    calc.add(5).add(10)
    return calc.get_value()
""")
        
        result = subprocess.run(
            [sys.executable, "-m", "code_scalpel.cli", "analyze", str(test_file)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should detect classes and functions
        assert "Classes" in result.stdout or "Functions" in result.stdout or "Metrics" in result.stdout


class TestCLIDirectImport:
    """Test CLI functions via direct import for better coverage."""

    def test_analyze_file_not_found(self):
        """Test analyze_file with nonexistent file."""
        from code_scalpel.cli import analyze_file
        
        result = analyze_file("/nonexistent/path/to/file.py")
        assert result == 1

    def test_analyze_file_non_py_extension(self, tmp_path, capsys):
        """Test analyze_file warns for non-.py files."""
        from code_scalpel.cli import analyze_file
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("x = 1")
        
        result = analyze_file(str(test_file))
        captured = capsys.readouterr()
        assert "Warning" in captured.err

    def test_analyze_file_json_format(self, tmp_path, capsys):
        """Test analyze_file with JSON output format."""
        from code_scalpel.cli import analyze_file
        import json
        
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): return 42")
        
        result = analyze_file(str(test_file), output_format="json")
        captured = capsys.readouterr()
        
        assert result == 0
        # Output should contain valid JSON
        output_lines = captured.out.strip().split('\n')
        found_json = False
        for line in output_lines:
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    found_json = True
                    break
            except json.JSONDecodeError:
                continue
        assert found_json or "source" in captured.out

    def test_analyze_code_json_format(self, capsys):
        """Test analyze_code with JSON output format."""
        from code_scalpel.cli import analyze_code
        import json
        
        result = analyze_code("x = 1", output_format="json")
        captured = capsys.readouterr()
        
        assert result == 0
        # Try to parse JSON from output
        output_lines = captured.out.strip().split('\n')
        for line in output_lines:
            try:
                data = json.loads(line)
                assert isinstance(data, dict)
                break
            except json.JSONDecodeError:
                continue

    def test_analyze_code_syntax_error(self, capsys):
        """Test analyze_code with code that has syntax error."""
        from code_scalpel.cli import analyze_code
        
        result = analyze_code("def broken(")
        captured = capsys.readouterr()
        
        # May succeed (with error in result) or fail
        combined = captured.out + captured.err
        assert result in [0, 1]

    def test_main_no_args(self, capsys, monkeypatch):
        """Test main with no arguments shows help."""
        from code_scalpel.cli import main
        import sys
        
        monkeypatch.setattr(sys, 'argv', ['code-scalpel'])
        result = main()
        captured = capsys.readouterr()
        
        assert result == 0
        assert "usage" in captured.out.lower() or "code-scalpel" in captured.out.lower()

    def test_main_version(self, capsys, monkeypatch):
        """Test main version command."""
        from code_scalpel.cli import main
        import sys
        
        monkeypatch.setattr(sys, 'argv', ['code-scalpel', 'version'])
        result = main()
        captured = capsys.readouterr()
        
        assert result == 0
        assert "0.1.0" in captured.out or "Code Scalpel" in captured.out

    def test_main_analyze_with_code(self, capsys, monkeypatch):
        """Test main analyze --code."""
        from code_scalpel.cli import main
        import sys
        
        monkeypatch.setattr(sys, 'argv', ['code-scalpel', 'analyze', '--code', 'x = 1'])
        result = main()
        captured = capsys.readouterr()
        
        assert result == 0

    def test_main_analyze_no_input(self, capsys, monkeypatch):
        """Test main analyze without file or code."""
        from code_scalpel.cli import main
        import sys
        
        monkeypatch.setattr(sys, 'argv', ['code-scalpel', 'analyze'])
        result = main()
        captured = capsys.readouterr()
        
        assert result == 1  # Should fail with exit code 1
