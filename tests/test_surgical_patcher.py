"""Tests for the Surgical Patcher module.

Tests the precision code modification capabilities for safe LLM-driven refactoring.
"""

import pytest
from code_scalpel.surgical_patcher import (
    SurgicalPatcher,
    PatchResult,
    update_function_in_file,
    update_class_in_file,
    update_method_in_file,
)


class TestSurgicalPatcher:
    """Tests for the SurgicalPatcher class."""

    @pytest.fixture
    def sample_code(self):
        """Sample code with functions, classes, and methods."""
        return '''import os
from typing import List

CONSTANT = 42


def helper():
    """A helper function."""
    return CONSTANT


def calculate(x, y):
    """Calculate using helper."""
    return helper() + x + y


class Calculator:
    """A calculator class."""

    def __init__(self, value=0):
        self.value = value

    def add(self, x):
        """Add to value."""
        self.value += x
        return self

    def multiply(self, x):
        """Multiply value."""
        self.value *= x
        return self


def standalone():
    """No dependencies."""
    return 1 + 1
'''

    def test_patcher_init(self, sample_code):
        """Test patcher initialization."""
        patcher = SurgicalPatcher(sample_code)
        assert patcher.original_code == sample_code
        assert patcher.current_code == sample_code
        assert patcher.file_path is None

    def test_patcher_from_file(self, tmp_path, sample_code):
        """Test creating patcher from file."""
        test_file = tmp_path / "test.py"
        test_file.write_text(sample_code)

        patcher = SurgicalPatcher.from_file(str(test_file))
        assert patcher.original_code == sample_code
        assert patcher.file_path == str(test_file)

    def test_patcher_file_not_found(self):
        """Test FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            SurgicalPatcher.from_file("/nonexistent/path.py")


class TestUpdateFunction:
    """Tests for function replacement."""

    @pytest.fixture
    def patcher(self):
        """Create a patcher with sample code."""
        code = '''
def old_func():
    """Old implementation."""
    return 1


def other_func():
    return 2
'''
        return SurgicalPatcher(code)

    def test_update_function_basic(self, patcher):
        """Test basic function replacement."""
        new_code = '''def old_func():
    """New implementation!"""
    return 42
'''
        result = patcher.update_function("old_func", new_code)

        assert result.success is True
        assert result.target_name == "old_func"
        assert result.target_type == "function"
        assert "return 42" in patcher.get_modified_code()
        assert "return 2" in patcher.get_modified_code()  # other_func preserved

    def test_update_function_not_found(self, patcher):
        """Test error when function doesn't exist."""
        result = patcher.update_function("nonexistent", "def nonexistent(): pass")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_update_function_wrong_type(self):
        """Test error when target is not a function."""
        code = "class MyClass: pass"
        patcher = SurgicalPatcher(code)
        result = patcher.update_function("MyClass", "def MyClass(): pass")

        assert result.success is False
        assert "class" in result.error.lower()

    def test_update_function_invalid_replacement(self, patcher):
        """Test error when replacement code is invalid."""
        result = patcher.update_function("old_func", "not valid python(")

        assert result.success is False
        assert "syntax" in result.error.lower()

    def test_update_function_wrong_replacement_type(self, patcher):
        """Test error when replacement is wrong type."""
        result = patcher.update_function("old_func", "class WrongType: pass")

        assert result.success is False
        assert "function" in result.error.lower()


class TestUpdateClass:
    """Tests for class replacement."""

    @pytest.fixture
    def patcher(self):
        """Create a patcher with class code."""
        code = '''
class OldClass:
    """Old class."""
    def method(self):
        return 1


def helper():
    return 2
'''
        return SurgicalPatcher(code)

    def test_update_class_basic(self, patcher):
        """Test basic class replacement."""
        new_code = '''class OldClass:
    """New class!"""
    def method(self):
        return 42

    def new_method(self):
        return 100
'''
        result = patcher.update_class("OldClass", new_code)

        assert result.success is True
        assert result.target_name == "OldClass"
        assert result.target_type == "class"
        assert "return 42" in patcher.get_modified_code()
        assert "new_method" in patcher.get_modified_code()
        assert "helper" in patcher.get_modified_code()  # helper preserved

    def test_update_class_not_found(self, patcher):
        """Test error when class doesn't exist."""
        result = patcher.update_class("NonexistentClass", "class NonexistentClass: pass")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestUpdateMethod:
    """Tests for method replacement."""

    @pytest.fixture
    def patcher(self):
        """Create a patcher with class containing methods."""
        code = '''
class Calculator:
    """A calculator."""

    def __init__(self):
        self.value = 0

    def add(self, x):
        """Add x to value."""
        self.value += x
        return self

    def subtract(self, x):
        """Subtract x from value."""
        self.value -= x
        return self


def helper():
    return 42
'''
        return SurgicalPatcher(code)

    def test_update_method_basic(self, patcher):
        """Test basic method replacement."""
        new_code = '''def add(self, x):
    """Add x to value with logging."""
    print(f"Adding {x}")
    self.value += x
    return self
'''
        result = patcher.update_method("Calculator", "add", new_code)

        assert result.success is True
        assert result.target_name == "Calculator.add"
        assert result.target_type == "method"
        assert "print" in patcher.get_modified_code()
        assert "subtract" in patcher.get_modified_code()  # Other method preserved

    def test_update_method_not_found(self, patcher):
        """Test error when method doesn't exist."""
        result = patcher.update_method("Calculator", "nonexistent", "def nonexistent(self): pass")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_update_method_class_not_found(self, patcher):
        """Test error when class doesn't exist."""
        result = patcher.update_method("NonexistentClass", "method", "def method(self): pass")

        assert result.success is False
        assert "not found" in result.error.lower()


class TestFileSaveAndBackup:
    """Tests for file save operations."""

    def test_save_creates_backup(self, tmp_path):
        """Test that save creates a backup file."""
        code = "def old(): return 1"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        patcher = SurgicalPatcher.from_file(str(test_file))
        patcher.update_function("old", "def old(): return 42")
        backup_path = patcher.save(backup=True)

        assert backup_path is not None
        assert backup_path.endswith(".bak")
        assert (tmp_path / "test.py.bak").exists()
        assert (tmp_path / "test.py.bak").read_text() == code

    def test_save_modifies_file(self, tmp_path):
        """Test that save modifies the original file."""
        code = "def old(): return 1"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        patcher = SurgicalPatcher.from_file(str(test_file))
        patcher.update_function("old", "def old(): return 42")
        patcher.save()

        assert "return 42" in test_file.read_text()

    def test_save_no_backup(self, tmp_path):
        """Test save without backup."""
        code = "def old(): return 1"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        patcher = SurgicalPatcher.from_file(str(test_file))
        patcher.update_function("old", "def old(): return 42")
        backup_path = patcher.save(backup=False)

        assert backup_path is None
        assert not (tmp_path / "test.py.bak").exists()

    def test_save_no_file_path_error(self):
        """Test error when saving without file_path."""
        patcher = SurgicalPatcher("def old(): return 1")
        patcher.update_function("old", "def old(): return 42")

        with pytest.raises(ValueError, match="file_path"):
            patcher.save()

    def test_discard_changes(self, tmp_path):
        """Test discarding changes."""
        code = "def old(): return 1"
        patcher = SurgicalPatcher(code)
        patcher.update_function("old", "def old(): return 42")

        assert "return 42" in patcher.get_modified_code()

        patcher.discard_changes()

        assert "return 1" in patcher.get_modified_code()
        assert "return 42" not in patcher.get_modified_code()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_update_function_in_file(self, tmp_path):
        """Test update_function_in_file convenience function."""
        code = "def old(): return 1"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        result = update_function_in_file(
            str(test_file), "old", "def old(): return 42"
        )

        assert result.success is True
        assert "return 42" in test_file.read_text()

    def test_update_class_in_file(self, tmp_path):
        """Test update_class_in_file convenience function."""
        code = "class Old:\n    pass"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        result = update_class_in_file(
            str(test_file), "Old", "class Old:\n    value = 42"
        )

        assert result.success is True
        assert "value = 42" in test_file.read_text()

    def test_update_method_in_file(self, tmp_path):
        """Test update_method_in_file convenience function."""
        code = "class Calc:\n    def add(self): return 1"
        test_file = tmp_path / "test.py"
        test_file.write_text(code)

        result = update_method_in_file(
            str(test_file), "Calc", "add", "def add(self): return 42"
        )

        assert result.success is True
        assert "return 42" in test_file.read_text()


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_function_with_decorator(self):
        """Test updating a decorated function."""
        code = '''
@decorator
def decorated():
    return 1
'''
        patcher = SurgicalPatcher(code)
        new_code = '''@decorator
def decorated():
    return 42
'''
        result = patcher.update_function("decorated", new_code)

        assert result.success is True
        assert "@decorator" in patcher.get_modified_code()
        assert "return 42" in patcher.get_modified_code()

    def test_async_function(self):
        """Test updating an async function."""
        code = '''
async def fetch():
    return await something()
'''
        patcher = SurgicalPatcher(code)
        new_code = '''async def fetch():
    result = await something()
    return result
'''
        result = patcher.update_function("fetch", new_code)

        assert result.success is True
        assert "result = await" in patcher.get_modified_code()

    def test_preserve_surrounding_code(self):
        """Test that surrounding code is preserved exactly."""
        code = '''# Header comment
import os

def target():
    return 1

# Middle comment

def other():
    return 2

# Footer comment
'''
        patcher = SurgicalPatcher(code)
        patcher.update_function("target", "def target():\n    return 42")

        modified = patcher.get_modified_code()
        assert "# Header comment" in modified
        assert "import os" in modified
        assert "# Middle comment" in modified
        assert "def other():" in modified
        assert "# Footer comment" in modified

    def test_multiline_function(self):
        """Test updating a multi-line function."""
        code = '''
def long_function(
    arg1,
    arg2,
    arg3
):
    result = arg1 + arg2
    result += arg3
    return result
'''
        patcher = SurgicalPatcher(code)
        new_code = '''def long_function(arg1, arg2, arg3):
    return arg1 + arg2 + arg3
'''
        result = patcher.update_function("long_function", new_code)

        assert result.success is True
        assert result.lines_before > result.lines_after  # Got shorter

    def test_class_with_multiple_methods(self):
        """Test updating a method in a class with many methods."""
        code = '''
class Service:
    def method1(self):
        return 1

    def target_method(self):
        return "old"

    def method3(self):
        return 3
'''
        patcher = SurgicalPatcher(code)
        new_code = '''def target_method(self):
    return "new"
'''
        result = patcher.update_method("Service", "target_method", new_code)

        assert result.success is True
        modified = patcher.get_modified_code()
        assert "method1" in modified
        assert "method3" in modified
        assert '"new"' in modified

    def test_empty_replacement_error(self):
        """Test error on empty replacement code."""
        patcher = SurgicalPatcher("def old(): pass")
        result = patcher.update_function("old", "")

        assert result.success is False

    def test_lines_delta(self):
        """Test that lines_delta is calculated correctly."""
        code = "def func():\n    return 1"
        patcher = SurgicalPatcher(code)
        new_code = "def func():\n    x = 1\n    y = 2\n    return x + y"
        result = patcher.update_function("func", new_code)

        assert result.success is True
        assert result.lines_delta == result.lines_after - result.lines_before
        assert result.lines_after > result.lines_before
