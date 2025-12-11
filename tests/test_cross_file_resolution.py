"""
Tests for cross-file dependency resolution in SurgicalExtractor.

Directive 3: When calculate_tax imports TaxRate from models.py,
the extractor should optionally grab the definition of TaxRate.
"""

import os
import tempfile
import pytest
from pathlib import Path

from code_scalpel.surgical_extractor import (
    SurgicalExtractor,
    CrossFileSymbol,
    CrossFileResolution,
)


class TestCrossFileResolution:
    """Test basic cross-file dependency resolution."""

    @pytest.fixture
    def temp_project(self, tmp_path: Path):
        """Create a temporary project structure with multiple files."""
        # models.py - defines TaxRate
        models_py = tmp_path / "models.py"
        models_py.write_text(
            '''"""Models module."""

class TaxRate:
    """Tax rate configuration."""
    
    def __init__(self, rate: float):
        self.rate = rate
    
    def calculate(self, amount: float) -> float:
        """Calculate tax for an amount."""
        return amount * self.rate


def get_default_rate() -> float:
    """Get the default tax rate."""
    return 0.1


TAX_CONSTANT = 0.15
'''
        )

        # utils.py - imports from models
        utils_py = tmp_path / "utils.py"
        utils_py.write_text(
            '''"""Utilities module."""

from models import TaxRate, get_default_rate, TAX_CONSTANT


def calculate_tax(amount: float) -> float:
    """Calculate tax using TaxRate."""
    rate = TaxRate(get_default_rate())
    return rate.calculate(amount)


def apply_flat_tax(amount: float) -> float:
    """Apply flat tax using constant."""
    return amount * TAX_CONSTANT
'''
        )

        # services/order.py - nested import
        services_dir = tmp_path / "services"
        services_dir.mkdir()
        order_py = services_dir / "order.py"
        order_py.write_text(
            '''"""Order service."""

from models import TaxRate


class OrderProcessor:
    """Process orders with tax."""
    
    def __init__(self):
        self.tax_rate = TaxRate(0.08)
    
    def process(self, amount: float) -> float:
        return amount + self.tax_rate.calculate(amount)
'''
        )

        return tmp_path

    def test_resolve_function_with_class_import(self, temp_project: Path):
        """Test resolving a function that imports a class."""
        utils_path = temp_project / "utils.py"
        extractor = SurgicalExtractor.from_file(str(utils_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="calculate_tax",
            target_type="function",
        )

        assert result.success
        assert result.target.success
        assert result.target.name == "calculate_tax"

        # Should have resolved TaxRate from models.py
        external_names = [sym.name for sym in result.external_symbols]
        assert "TaxRate" in external_names

        # Check that TaxRate code was extracted
        tax_rate_sym = next(
            (s for s in result.external_symbols if s.name == "TaxRate"), None
        )
        assert tax_rate_sym is not None
        assert "class TaxRate:" in tax_rate_sym.code
        assert "models.py" in tax_rate_sym.source_file

    def test_resolve_function_with_function_import(self, temp_project: Path):
        """Test resolving a function that imports another function."""
        utils_path = temp_project / "utils.py"
        extractor = SurgicalExtractor.from_file(str(utils_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="calculate_tax",
            target_type="function",
        )

        assert result.success

        # Should have resolved get_default_rate from models.py
        external_names = [sym.name for sym in result.external_symbols]
        assert "get_default_rate" in external_names

        # Check the function code
        get_rate_sym = next(
            (s for s in result.external_symbols if s.name == "get_default_rate"), None
        )
        assert get_rate_sym is not None
        assert "def get_default_rate" in get_rate_sym.code

    def test_resolve_function_with_variable_import(self, temp_project: Path):
        """Test resolving a function that imports a variable."""
        utils_path = temp_project / "utils.py"
        extractor = SurgicalExtractor.from_file(str(utils_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="apply_flat_tax",
            target_type="function",
        )

        assert result.success

        # Should have resolved TAX_CONSTANT from models.py
        external_names = [sym.name for sym in result.external_symbols]
        assert "TAX_CONSTANT" in external_names

    def test_resolve_class_dependencies(self, temp_project: Path):
        """Test resolving class dependencies from external files."""
        order_path = temp_project / "services" / "order.py"
        extractor = SurgicalExtractor.from_file(str(order_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="OrderProcessor",
            target_type="class",
        )

        assert result.success
        assert result.target.name == "OrderProcessor"

        # Should have resolved TaxRate from models.py
        external_names = [sym.name for sym in result.external_symbols]
        assert "TaxRate" in external_names

    def test_full_code_property(self, temp_project: Path):
        """Test that full_code combines external symbols with target."""
        utils_path = temp_project / "utils.py"
        extractor = SurgicalExtractor.from_file(str(utils_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="calculate_tax",
            target_type="function",
        )

        full_code = result.full_code

        # Should contain the external symbols
        assert "# From" in full_code
        assert "models.py" in full_code
        assert "class TaxRate:" in full_code

        # Should contain the target function
        assert "def calculate_tax" in full_code

    def test_token_estimate(self, temp_project: Path):
        """Test token estimation for cross-file resolution."""
        utils_path = temp_project / "utils.py"
        extractor = SurgicalExtractor.from_file(str(utils_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="calculate_tax",
            target_type="function",
        )

        # Token estimate should be roughly chars / 4
        expected = len(result.full_code) // 4
        assert result.token_estimate == expected


class TestCrossFileEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_project(self, tmp_path: Path):
        """Create a project with various edge cases."""
        # File with missing import target
        broken_py = tmp_path / "broken.py"
        broken_py.write_text(
            '''"""Broken imports."""

from nonexistent import DoesNotExist


def use_missing():
    return DoesNotExist()
'''
        )

        # File with relative imports
        pkg_dir = tmp_path / "package"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")

        helper_py = pkg_dir / "helper.py"
        helper_py.write_text(
            '''"""Helper module."""

def helper_func():
    return 42
'''
        )

        main_py = pkg_dir / "main.py"
        main_py.write_text(
            '''"""Main module."""

from .helper import helper_func


def main():
    return helper_func()
'''
        )

        # Standalone file
        standalone_py = tmp_path / "standalone.py"
        standalone_py.write_text(
            '''"""No imports."""

def pure_function(x: int) -> int:
    return x * 2
'''
        )

        return tmp_path

    def test_unresolved_import(self, temp_project: Path):
        """Test handling of imports that cannot be resolved."""
        broken_path = temp_project / "broken.py"
        extractor = SurgicalExtractor.from_file(str(broken_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="use_missing",
            target_type="function",
        )

        assert result.success  # Extraction succeeds
        assert result.target.success

        # But the import is unresolved
        assert len(result.unresolved_imports) > 0
        assert any("DoesNotExist" in u for u in result.unresolved_imports)

    def test_no_file_path(self):
        """Test cross-file resolution without file_path set."""
        code = """
def my_func():
    return 42
"""
        extractor = SurgicalExtractor(code)  # No file_path

        result = extractor.resolve_cross_file_dependencies(
            target_name="my_func",
            target_type="function",
        )

        assert result.success
        assert result.target.success
        # Should have error message about file_path
        assert "file_path not set" in result.error

    def test_function_not_found(self, temp_project: Path):
        """Test resolving a non-existent function."""
        standalone_path = temp_project / "standalone.py"
        extractor = SurgicalExtractor.from_file(str(standalone_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="does_not_exist",
            target_type="function",
        )

        assert not result.success
        assert "not found" in result.target.error

    def test_no_external_deps(self, temp_project: Path):
        """Test function with no external dependencies."""
        standalone_path = temp_project / "standalone.py"
        extractor = SurgicalExtractor.from_file(str(standalone_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="pure_function",
            target_type="function",
        )

        assert result.success
        assert result.target.success
        assert len(result.external_symbols) == 0

    def test_relative_import(self, temp_project: Path):
        """Test resolving relative imports."""
        main_path = temp_project / "package" / "main.py"
        extractor = SurgicalExtractor.from_file(str(main_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="main",
            target_type="function",
        )

        assert result.success

        # Should resolve helper_func from relative import
        external_names = [sym.name for sym in result.external_symbols]
        assert "helper_func" in external_names


class TestCrossFileDepthLimits:
    """Test depth limiting for dependency resolution."""

    @pytest.fixture
    def chain_project(self, tmp_path: Path):
        """Create a project with chained dependencies."""
        # level0.py
        (tmp_path / "level0.py").write_text(
            '''"""Base level."""

BASE_VALUE = 100

def level0_func():
    return BASE_VALUE
'''
        )

        # level1.py imports level0
        (tmp_path / "level1.py").write_text(
            '''"""Level 1."""

from level0 import level0_func, BASE_VALUE


def level1_func():
    return level0_func() + BASE_VALUE
'''
        )

        # level2.py imports level1
        (tmp_path / "level2.py").write_text(
            '''"""Level 2."""

from level1 import level1_func


def level2_func():
    return level1_func() * 2
'''
        )

        return tmp_path

    def test_depth_1_resolution(self, chain_project: Path):
        """Test resolution with depth=1 (default)."""
        level2_path = chain_project / "level2.py"
        extractor = SurgicalExtractor.from_file(str(level2_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="level2_func",
            target_type="function",
            max_depth=1,
        )

        assert result.success

        # Should resolve level1_func but not level0_func
        external_names = [sym.name for sym in result.external_symbols]
        assert "level1_func" in external_names
        # Depth=1 shouldn't resolve level0 dependencies
        assert "level0_func" not in external_names

    def test_depth_2_resolution(self, chain_project: Path):
        """Test resolution with depth=2 (transitive)."""
        level2_path = chain_project / "level2.py"
        extractor = SurgicalExtractor.from_file(str(level2_path))

        result = extractor.resolve_cross_file_dependencies(
            target_name="level2_func",
            target_type="function",
            max_depth=2,
        )

        assert result.success

        external_names = [sym.name for sym in result.external_symbols]
        # Should resolve both level1 and level0 dependencies
        assert "level1_func" in external_names
        assert "level0_func" in external_names


class TestImportMapBuilding:
    """Test the _build_import_map method."""

    def test_from_import(self, tmp_path: Path):
        """Test mapping from 'from X import Y' syntax."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''from models import User, Order
from utils import helper as h

def func():
    pass
'''
        )

        extractor = SurgicalExtractor.from_file(str(test_file))
        import_map = extractor._build_import_map()

        assert "User" in import_map
        assert "Order" in import_map
        assert "h" in import_map  # aliased import

    def test_import_module(self, tmp_path: Path):
        """Test mapping from 'import X' syntax."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''import os
import sys as system

def func():
    pass
'''
        )

        extractor = SurgicalExtractor.from_file(str(test_file))
        import_map = extractor._build_import_map()

        assert "os" in import_map
        assert "system" in import_map  # aliased


class TestCrossFileSymbol:
    """Test CrossFileSymbol dataclass."""

    def test_symbol_creation(self):
        """Test creating a CrossFileSymbol."""
        symbol = CrossFileSymbol(
            name="TaxRate",
            source_file="/path/to/models.py",
            code="class TaxRate:\n    pass",
            node_type="class",
            import_statement="from models import TaxRate",
        )

        assert symbol.name == "TaxRate"
        assert symbol.node_type == "class"
        assert "class TaxRate" in symbol.code


class TestCrossFileResolutionDataclass:
    """Test CrossFileResolution dataclass."""

    def test_empty_resolution(self):
        """Test resolution with no external symbols."""
        from code_scalpel.surgical_extractor import ExtractionResult

        target = ExtractionResult(
            success=True,
            name="test",
            code="def test(): pass",
            node_type="function",
        )

        resolution = CrossFileResolution(
            success=True,
            target=target,
        )

        assert resolution.success
        assert len(resolution.external_symbols) == 0
        assert resolution.full_code == "def test(): pass"

    def test_resolution_with_symbols(self):
        """Test resolution with external symbols."""
        from code_scalpel.surgical_extractor import ExtractionResult

        target = ExtractionResult(
            success=True,
            name="test",
            code="def test(): pass",
            node_type="function",
        )

        symbol = CrossFileSymbol(
            name="Helper",
            source_file="/path/models.py",
            code="class Helper: pass",
            node_type="class",
            import_statement="from models import Helper",
        )

        resolution = CrossFileResolution(
            success=True,
            target=target,
            external_symbols=[symbol],
        )

        full_code = resolution.full_code

        assert "# From /path/models.py" in full_code
        assert "class Helper: pass" in full_code
        assert "def test(): pass" in full_code
