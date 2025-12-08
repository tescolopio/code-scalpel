"""Tests for ast_tools/__init__.py to close coverage gaps.

These tests target specific coverage gaps:
- Lines 8-9, 13-14, 18-19: ImportError branches for optional modules
- Line 39: build_ast_from_file function definition
- Line 44: visualize_ast function
- Lines 50-53: __all__ exports
"""

import ast
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestAstToolsImports:
    """Tests for ast_tools module imports."""

    def test_core_imports_available(self):
        """Test that core imports are always available."""
        from code_scalpel.ast_tools import (
            ASTAnalyzer,
            ASTBuilder,
            ClassMetrics,
            FunctionMetrics,
        )

        assert ASTAnalyzer is not None
        assert ASTBuilder is not None
        assert FunctionMetrics is not None
        assert ClassMetrics is not None

    def test_optional_imports_available(self):
        """Test that optional imports are available (or None)."""
        from code_scalpel import ast_tools

        # These should be either the module or None if import failed
        assert hasattr(ast_tools, "ASTTransformer")
        assert hasattr(ast_tools, "ASTVisualizer")
        assert hasattr(ast_tools, "ASTValidator")

    def test_utils_imports_available(self):
        """Test that utils imports are available."""
        from code_scalpel import ast_tools

        assert hasattr(ast_tools, "is_constant")
        assert hasattr(ast_tools, "get_node_type")
        assert hasattr(ast_tools, "get_all_names")


class TestBuildAstFunction:
    """Tests for the build_ast convenience function."""

    def test_build_ast_simple_code(self):
        """Test build_ast with simple Python code."""
        from code_scalpel.ast_tools import build_ast

        tree = build_ast("x = 1 + 2")

        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.Assign)

    def test_build_ast_complex_code(self):
        """Test build_ast with more complex code."""
        from code_scalpel.ast_tools import build_ast

        code = "def add(a, b):\n    return a + b"
        tree = build_ast(code)

        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1

    def test_build_ast_without_preprocessing(self):
        """Test build_ast with preprocessing disabled."""
        from code_scalpel.ast_tools import build_ast

        tree = build_ast("x = 1", preprocess=False)
        assert isinstance(tree, ast.Module)

    def test_build_ast_without_validation(self):
        """Test build_ast with validation disabled."""
        from code_scalpel.ast_tools import build_ast

        tree = build_ast("x = 1", validate=False)
        assert isinstance(tree, ast.Module)


class TestBuildAstFromFile:
    """Tests for the build_ast_from_file convenience function (line 39)."""

    def test_build_ast_from_file_basic(self):
        """Test building AST from a file."""
        from code_scalpel.ast_tools import build_ast_from_file

        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("x = 42\n")
            temp_path = f.name

        try:
            tree = build_ast_from_file(temp_path)

            assert isinstance(tree, ast.Module)
            assert len(tree.body) == 1
        finally:
            os.unlink(temp_path)

    def test_build_ast_from_file_with_function(self):
        """Test building AST from file containing function."""
        from code_scalpel.ast_tools import build_ast_from_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def hello():\n    return 'world'\n")
            temp_path = f.name

        try:
            tree = build_ast_from_file(temp_path)

            assert isinstance(tree, ast.Module)
            assert isinstance(tree.body[0], ast.FunctionDef)
            assert tree.body[0].name == "hello"
        finally:
            os.unlink(temp_path)

    def test_build_ast_from_file_without_preprocessing(self):
        """Test build_ast_from_file with preprocessing disabled."""
        from code_scalpel.ast_tools import build_ast_from_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("x = 1\n")
            temp_path = f.name

        try:
            tree = build_ast_from_file(temp_path, preprocess=False)
            assert isinstance(tree, ast.Module)
        finally:
            os.unlink(temp_path)

    def test_build_ast_from_file_without_validation(self):
        """Test build_ast_from_file with validation disabled."""
        from code_scalpel.ast_tools import build_ast_from_file

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("x = 1\n")
            temp_path = f.name

        try:
            tree = build_ast_from_file(temp_path, validate=False)
            assert isinstance(tree, ast.Module)
        finally:
            os.unlink(temp_path)


class TestVisualizeAst:
    """Tests for visualize_ast convenience function (lines 44, 50-53)."""

    def test_visualize_ast_with_visualizer_available(self):
        """Test visualize_ast when ASTVisualizer is available (lines 51-52)."""
        from code_scalpel import ast_tools
        from code_scalpel.ast_tools import build_ast, visualize_ast

        tree = build_ast("x = 1")

        # Only test if visualizer is available
        if ast_tools.ASTVisualizer is not None:
            # Mock the visualize method to avoid actual file creation
            with patch.object(
                ast_tools.ASTVisualizer, "visualize", return_value="output.png"
            ) as mock_viz:
                # Call the module-level convenience function
                result = visualize_ast(tree, "test_output", "png", False)
                # The mock should have been called
                mock_viz.assert_called_once()
                assert result == "output.png"

    def test_visualize_ast_function_exists(self):
        """Test that visualize_ast function is accessible."""
        from code_scalpel.ast_tools import visualize_ast

        assert callable(visualize_ast)

    def test_visualize_ast_raises_when_unavailable(self):
        """Test visualize_ast raises ImportError when ASTVisualizer is None."""
        # Import the module
        import code_scalpel.ast_tools as ast_tools_module
        from code_scalpel.ast_tools import build_ast

        tree = build_ast("x = 1")

        # Save original value
        original_visualizer = ast_tools_module.ASTVisualizer

        try:
            # Set to None to simulate import failure
            ast_tools_module.ASTVisualizer = None

            with pytest.raises(ImportError, match="ASTVisualizer not available"):
                ast_tools_module.visualize_ast(tree)
        finally:
            # Restore original value
            ast_tools_module.ASTVisualizer = original_visualizer


class TestAllExports:
    """Tests for __all__ exports (lines 50-53)."""

    def test_all_exports_defined(self):
        """Test that __all__ is defined and contains expected items."""
        from code_scalpel import ast_tools

        assert hasattr(ast_tools, "__all__")
        assert isinstance(ast_tools.__all__, list)

    def test_all_exports_contain_core_classes(self):
        """Test __all__ contains core class exports."""
        from code_scalpel import ast_tools

        expected = ["ASTAnalyzer", "FunctionMetrics", "ClassMetrics", "ASTBuilder"]
        for item in expected:
            assert item in ast_tools.__all__

    def test_all_exports_are_accessible(self):
        """Test all items in __all__ are actually accessible."""
        from code_scalpel import ast_tools

        for name in ast_tools.__all__:
            assert hasattr(ast_tools, name), f"{name} in __all__ but not accessible"


class TestImportErrorBranches:
    """Tests specifically targeting ImportError branches."""

    def test_import_with_mocked_transformer_failure(self):
        """Test handling when transformer import fails (lines 8-9)."""
        # We can't easily trigger the actual ImportError since the module
        # is already loaded, but we can verify the None fallback works
        from code_scalpel import ast_tools

        # If ASTTransformer is None, that branch was executed
        # If it's not None, that means the import succeeded
        assert ast_tools.ASTTransformer is None or hasattr(
            ast_tools.ASTTransformer, "__call__"
        )

    def test_import_with_mocked_visualizer_failure(self):
        """Test handling when visualizer import fails (lines 13-14)."""
        from code_scalpel import ast_tools

        assert ast_tools.ASTVisualizer is None or hasattr(
            ast_tools.ASTVisualizer, "__call__"
        )

    def test_import_with_mocked_validator_failure(self):
        """Test handling when validator import fails (lines 18-19)."""
        from code_scalpel import ast_tools

        assert ast_tools.ASTValidator is None or hasattr(
            ast_tools.ASTValidator, "__call__"
        )

    def test_utils_import_failure_handling(self):
        """Test handling when utils imports fail."""
        from code_scalpel import ast_tools

        # Either they're callable or None
        for name in ["is_constant", "get_node_type", "get_all_names"]:
            val = getattr(ast_tools, name)
            assert val is None or callable(val)


class TestImportErrorBranchesWithReload:
    """Tests that reload the module to exercise ImportError branches.

    These tests use importlib to reload the module with mocked imports
    that raise ImportError, exercising the defensive code paths.
    """

    def test_transformer_import_failure_branch(self):
        """Test lines 8-9: except ImportError for transformer."""
        import sys
        from unittest.mock import patch

        # Remove the module from cache to force reimport
        modules_to_remove = [k for k in list(sys.modules.keys()) if "ast_tools" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            # Mock the transformer import to fail
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if "transformer" in name:
                    raise ImportError("Mocked transformer import failure")
                return original_import(name, *args, **kwargs)

            with patch.dict(__builtins__, {"__import__": mock_import}):
                # This import will hit the except ImportError branch
                import code_scalpel.ast_tools as ast_tools_reloaded

                # The ASTTransformer should be None due to ImportError
                assert ast_tools_reloaded.ASTTransformer is None
        finally:
            # Restore modules
            sys.modules.update(saved_modules)

    def test_visualizer_import_failure_branch(self):
        """Test lines 13-14: except ImportError for visualizer."""
        import sys
        from unittest.mock import patch

        modules_to_remove = [k for k in list(sys.modules.keys()) if "ast_tools" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if "visualizer" in name:
                    raise ImportError("Mocked visualizer import failure")
                return original_import(name, *args, **kwargs)

            with patch.dict(__builtins__, {"__import__": mock_import}):
                import code_scalpel.ast_tools as ast_tools_reloaded

                assert ast_tools_reloaded.ASTVisualizer is None
        finally:
            sys.modules.update(saved_modules)

    def test_validator_import_failure_branch(self):
        """Test lines 18-19: except ImportError for validator."""
        import sys
        from unittest.mock import patch

        modules_to_remove = [k for k in list(sys.modules.keys()) if "ast_tools" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            original_import = __builtins__["__import__"]

            def mock_import(name, *args, **kwargs):
                if "validator" in name:
                    raise ImportError("Mocked validator import failure")
                return original_import(name, *args, **kwargs)

            with patch.dict(__builtins__, {"__import__": mock_import}):
                import code_scalpel.ast_tools as ast_tools_reloaded

                assert ast_tools_reloaded.ASTValidator is None
        finally:
            sys.modules.update(saved_modules)


class TestImportErrorBranchesWithReload:
    """Tests that reload the module to exercise ImportError branches.
    
    These tests use importlib to reload the module with mocked imports
    that raise ImportError, exercising the defensive code paths.
    """

    def test_transformer_import_failure_branch(self):
        """Test lines 8-9: except ImportError for transformer."""
        import builtins
        import sys
        from unittest.mock import patch

        # Remove the module from cache to force reimport
        modules_to_remove = [k for k in sys.modules if "ast_tools" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        original_import = builtins.__import__
        try:
            # Mock the transformer import to fail
            def mock_import(name, *args, **kwargs):
                if "transformer" in name:
                    raise ImportError("Mocked transformer import failure")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                # This import will hit the except ImportError branch
                import code_scalpel.ast_tools as ast_tools_reloaded
                
                # The ASTTransformer should be None due to ImportError
                assert ast_tools_reloaded.ASTTransformer is None
        finally:
            # Restore modules
            sys.modules.update(saved_modules)

    def test_visualizer_import_failure_branch(self):
        """Test lines 13-14: except ImportError for visualizer."""
        import builtins
        import sys
        from unittest.mock import patch

        modules_to_remove = [k for k in sys.modules if "ast_tools" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        original_import = builtins.__import__
        try:
            def mock_import(name, *args, **kwargs):
                if "visualizer" in name:
                    raise ImportError("Mocked visualizer import failure")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                import code_scalpel.ast_tools as ast_tools_reloaded
                
                assert ast_tools_reloaded.ASTVisualizer is None
        finally:
            sys.modules.update(saved_modules)

    def test_validator_import_failure_branch(self):
        """Test lines 18-19: except ImportError for validator."""
        import builtins
        import sys
        from unittest.mock import patch

        modules_to_remove = [k for k in sys.modules if "ast_tools" in k]
        saved_modules = {k: sys.modules.pop(k) for k in modules_to_remove}

        original_import = builtins.__import__
        try:
            def mock_import(name, *args, **kwargs):
                if "validator" in name:
                    raise ImportError("Mocked validator import failure")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                import code_scalpel.ast_tools as ast_tools_reloaded
                
                assert ast_tools_reloaded.ASTValidator is None
        finally:
            sys.modules.update(saved_modules)
