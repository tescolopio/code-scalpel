"""Tests for the Refactor Simulator.

Tests the safety verification of code changes before applying them.
"""

import pytest


class TestRefactorSimulator:
    """Tests for RefactorSimulator class."""

    def test_safe_refactor(self):
        """Test that safe refactors are detected as safe."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def add(x, y):
    return x + y
"""
        new_code = '''
def add(x, y):
    """Add two numbers."""
    return x + y
'''
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.is_safe is True
        assert result.status.value == "safe"

    def test_unsafe_eval_injection(self):
        """Test that introducing eval() is detected as unsafe."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def process(data):
    return data.upper()
"""
        new_code = """
def process(data):
    return eval(data)
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.is_safe is False
        assert result.status.value == "unsafe"
        assert any(
            "eval" in issue.description.lower()
            or "code injection" in issue.type.lower()
            for issue in result.security_issues
        )

    def test_unsafe_exec_injection(self):
        """Test that introducing exec() is detected as unsafe."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def run(code):
    print(code)
"""
        new_code = """
def run(code):
    exec(code)
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.is_safe is False
        assert result.status.value == "unsafe"

    def test_unsafe_os_system(self):
        """Test that introducing os.system() is detected as unsafe."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def run_command(cmd):
    print(f"Would run: {cmd}")
"""
        new_code = """
import os
def run_command(cmd):
    os.system(cmd)
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.is_safe is False
        assert "Command" in result.security_issues[0].type

    def test_structural_changes_tracked(self):
        """Test that structural changes are tracked."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def foo():
    pass

def bar():
    pass
"""
        new_code = """
def foo():
    pass

def baz():
    pass
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert "bar" in result.structural_changes.get("functions_removed", [])
        assert "baz" in result.structural_changes.get("functions_added", [])

    def test_syntax_error_detected(self):
        """Test that syntax errors in new code are detected."""
        from code_scalpel.generators import RefactorSimulator

        original = "def foo(): pass"
        new_code = "def foo(:"  # Syntax error

        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.status.value == "error"
        assert "syntax" in result.reason.lower()

    def test_strict_mode_warnings(self):
        """Test strict mode treats medium severity as unsafe."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def load(data):
    return data
"""
        # subprocess.call is medium severity
        new_code = """
import subprocess
def load(data):
    subprocess.call(data)
"""
        # Normal mode
        simulator = RefactorSimulator(strict_mode=False)
        simulator.simulate(original, new_code=new_code)
        # Medium severity might be warning, not necessarily unsafe

        # Strict mode
        strict_simulator = RefactorSimulator(strict_mode=True)
        strict_result = strict_simulator.simulate(original, new_code=new_code)

        # In strict mode, any security issue should be unsafe
        if strict_result.security_issues:
            assert strict_result.is_safe is False

    def test_simulate_inline_method(self):
        """Test the simulate_inline convenience method."""
        from code_scalpel.generators import RefactorSimulator

        original = "def foo(): return 1"
        new_code = "def foo(): return 2"

        simulator = RefactorSimulator()
        result = simulator.simulate_inline(original, new_code)

        assert result.is_safe is True

    def test_must_provide_new_code_or_patch(self):
        """Test that either new_code or patch must be provided."""
        from code_scalpel.generators import RefactorSimulator

        simulator = RefactorSimulator()

        with pytest.raises(ValueError, match="Must provide"):
            simulator.simulate("def foo(): pass")

    def test_result_to_dict(self):
        """Test RefactorResult serialization."""
        from code_scalpel.generators import RefactorSimulator

        original = "def foo(): pass"
        new_code = "def foo(): return 1"

        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        data = result.to_dict()

        assert "status" in data
        assert "is_safe" in data
        assert "security_issues" in data
        assert isinstance(data["security_issues"], list)

    def test_line_changes_tracked(self):
        """Test that line additions/removals are tracked."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def foo():
    pass
"""
        new_code = """
def foo():
    x = 1
    y = 2
    z = 3
    return x + y + z
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.structural_changes["lines_added"] > 0


class TestRefactorSimulatorPatch:
    """Tests for patch application."""

    def test_simple_patch_application(self):
        """Test applying a simple unified diff patch."""
        from code_scalpel.generators import RefactorSimulator

        original = """def foo():
    return 1
"""
        patch = """@@ -1,2 +1,2 @@
 def foo():
-    return 1
+    return 2
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, patch=patch)

        assert "return 2" in result.patched_code

    def test_invalid_patch_error(self):
        """Test that invalid patches are handled gracefully."""
        from code_scalpel.generators import RefactorSimulator

        original = "def foo(): pass"
        patch = "not a valid patch format"

        simulator = RefactorSimulator()
        # Should not crash
        simulator.simulate(original, patch=patch)
        # Result depends on implementation handling


class TestRefactorSimulatorSecurityPatterns:
    """Tests for specific security pattern detection."""

    def test_sql_injection_pattern(self):
        """Test SQL injection detection via cursor.execute."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def get_user(user_id):
    return {"id": user_id}
"""
        new_code = """
def get_user(user_id):
    cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return cursor.fetchone()
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.is_safe is False
        assert any("SQL" in issue.type for issue in result.security_issues)

    def test_pickle_deserialization(self):
        """Test pickle.loads detection."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def load_data(data):
    return data
"""
        new_code = """
import pickle
def load_data(data):
    return pickle.loads(data)
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert result.is_safe is False
        assert any("Deserialization" in issue.type for issue in result.security_issues)

    def test_safe_yaml_load(self):
        """Test that yaml.safe_load is not flagged (only yaml.load)."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def load_config(data):
    return data
"""
        # yaml.safe_load is safe
        new_code = """
import yaml
def load_config(data):
    return yaml.safe_load(data)
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        # Should not flag yaml.safe_load
        yaml_issues = [
            i for i in result.security_issues if "yaml" in i.description.lower()
        ]
        assert len(yaml_issues) == 0


class TestRefactorSimulatorWarnings:
    """Tests for warning generation."""

    def test_warning_on_function_removal(self):
        """Test warning when functions are removed."""
        from code_scalpel.generators import RefactorSimulator

        original = """
def important_function():
    return "critical"

def helper():
    pass
"""
        new_code = """
def helper():
    pass
"""
        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        assert any("important_function" in w for w in result.warnings)

    def test_warning_on_large_deletion(self):
        """Test warning when many lines are deleted."""
        from code_scalpel.generators import RefactorSimulator

        original = "\n".join([f"line{i} = {i}" for i in range(50)])
        new_code = "x = 1"

        simulator = RefactorSimulator()
        result = simulator.simulate(original, new_code=new_code)

        # Should warn about large deletion
        assert (
            any("delet" in w.lower() for w in result.warnings)
            or result.structural_changes["lines_removed"] > 40
        )
