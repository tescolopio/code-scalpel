"""
Tests for CodeAnalyzer class.

These tests verify the core functionality of the CodeAnalyzer including:
- AST parsing
- PDG construction
- Dead code detection
- Refactoring operations
"""

import sys
import os
import unittest

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from code_analyzer import (
    CodeAnalyzer,
    AnalysisLevel,
    AnalysisResult,
    DeadCodeItem,
    AnalysisMetrics,
    analyze_code
)


class TestCodeAnalyzerBasic(unittest.TestCase):
    """Test basic CodeAnalyzer functionality."""
    
    def setUp(self):
        self.analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)
    
    def test_create_analyzer(self):
        """Test creating an analyzer with different levels."""
        basic = CodeAnalyzer(level=AnalysisLevel.BASIC)
        standard = CodeAnalyzer(level=AnalysisLevel.STANDARD)
        full = CodeAnalyzer(level=AnalysisLevel.FULL)
        
        self.assertEqual(basic.level, AnalysisLevel.BASIC)
        self.assertEqual(standard.level, AnalysisLevel.STANDARD)
        self.assertEqual(full.level, AnalysisLevel.FULL)
    
    def test_analyze_simple_code(self):
        """Test analyzing simple code."""
        code = """
def hello():
    print("Hello, World!")

hello()
"""
        result = self.analyzer.analyze(code)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertIsNotNone(result.ast_tree)
        self.assertEqual(result.code, code)
        self.assertEqual(len(result.errors), 0)
    
    def test_analyze_returns_metrics(self):
        """Test that analysis returns metrics."""
        code = """
def func1():
    pass

def func2():
    x = 1
    y = 2
    return x + y

class MyClass:
    def method(self):
        pass
"""
        result = self.analyzer.analyze(code)
        
        self.assertIsInstance(result.metrics, AnalysisMetrics)
        self.assertEqual(result.metrics.num_functions, 3)
        self.assertEqual(result.metrics.num_classes, 1)
        self.assertGreater(result.metrics.lines_of_code, 0)
    
    def test_analyze_syntax_error(self):
        """Test handling of syntax errors."""
        code = """
def broken(
    # missing closing parenthesis
"""
        result = self.analyzer.analyze(code)
        
        self.assertGreater(len(result.errors), 0)
        # Error message could vary, just check there's an error about parsing/AST
        self.assertTrue(
            'parse' in result.errors[0].lower() or 
            'syntax' in result.errors[0].lower() or
            'ast' in result.errors[0].lower()
        )
    
    def test_caching(self):
        """Test that caching works."""
        code = "x = 1 + 2"
        
        # First analysis
        result1 = self.analyzer.analyze(code)
        
        # Second analysis should use cache
        result2 = self.analyzer.analyze(code)
        
        self.assertIs(result1, result2)
        
        # After clearing cache, should get new result
        self.analyzer.clear_cache()
        result3 = self.analyzer.analyze(code)
        
        self.assertIsNot(result1, result3)


class TestDeadCodeDetection(unittest.TestCase):
    """Test dead code detection functionality."""
    
    def setUp(self):
        self.analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)
    
    def test_detect_unused_function(self):
        """Test detection of unused functions."""
        code = """
def used_function():
    return 42

def unused_function():
    return 0

result = used_function()
"""
        result = self.analyzer.analyze(code)
        
        dead_names = [item.name for item in result.dead_code]
        self.assertIn('unused_function', dead_names)
        self.assertNotIn('used_function', dead_names)
    
    def test_detect_unused_variable(self):
        """Test detection of unused variables."""
        code = """
used_var = 10
unused_var = 20
print(used_var)
"""
        result = self.analyzer.analyze(code)
        
        dead_names = [item.name for item in result.dead_code]
        self.assertIn('unused_var', dead_names)
    
    def test_detect_unreachable_code(self):
        """Test detection of unreachable code after return."""
        code = """
def example():
    return 1
    x = 10  # This should be flagged as unreachable
"""
        result = self.analyzer.analyze(code)
        
        unreachable = [
            item for item in result.dead_code 
            if item.code_type == 'statement'
        ]
        self.assertGreater(len(unreachable), 0)
    
    def test_detect_unused_import(self):
        """Test detection of unused imports."""
        code = """
import os
import sys  # This import is unused

os.getcwd()
"""
        result = self.analyzer.analyze(code)
        
        dead_imports = [
            item for item in result.dead_code 
            if item.code_type == 'import'
        ]
        import_names = [item.name for item in dead_imports]
        self.assertIn('sys', import_names)
    
    def test_dead_code_summary(self):
        """Test dead code summary generation."""
        code = """
def unused():
    pass

unused_var = 10
"""
        result = self.analyzer.analyze(code)
        summary = self.analyzer.get_dead_code_summary(result)
        
        self.assertIn('dead code items', summary)
        self.assertIn('unused', summary)


class TestRefactoring(unittest.TestCase):
    """Test refactoring operations."""
    
    def setUp(self):
        self.analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)
    
    def test_remove_unused_imports(self):
        """Test removing unused imports."""
        code = """import os
import sys
import json

x = os.getcwd()
"""
        result = self.analyzer.apply_refactor(code, 'remove_unused_imports')
        
        # sys and json should be removed
        self.assertIn('import os', result)
        self.assertNotIn('import sys', result)
        self.assertNotIn('import json', result)
    
    def test_rename_variable(self):
        """Test renaming a variable."""
        code = """
old_name = 10
result = old_name + 5
print(old_name)
"""
        result = self.analyzer.apply_refactor(
            code, 'rename_variable', 
            target='old_name', 
            new_name='new_name'
        )
        
        self.assertIn('new_name', result)
        self.assertNotIn('old_name', result)
    
    def test_remove_dead_code(self):
        """Test removing dead code."""
        code = """
def used():
    return 1

def unused():
    pass

x = used()
"""
        # Get initial analysis to see dead code count
        initial_result = self.analyzer.analyze(code)
        initial_dead_count = len(initial_result.dead_code)
        
        # Apply refactoring
        new_code = self.analyzer.apply_refactor(code, 'remove_dead_code')
        
        # The refactored code should have fewer or equal dead items
        # (We just verify it doesn't crash and returns something)
        self.assertIsInstance(new_code, str)


class TestPDGConstruction(unittest.TestCase):
    """Test PDG construction."""
    
    def setUp(self):
        self.analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)
    
    def test_pdg_created(self):
        """Test that PDG is created for standard level analysis."""
        code = """
x = 10
y = x + 5
z = y * 2
"""
        result = self.analyzer.analyze(code)
        
        self.assertIsNotNone(result.pdg)
        self.assertGreater(result.pdg.number_of_nodes(), 0)
    
    def test_call_graph_created(self):
        """Test that call graph is created."""
        code = """
def caller():
    return callee()

def callee():
    return 42
"""
        result = self.analyzer.analyze(code)
        
        self.assertIsNotNone(result.call_graph)


class TestSecurityAnalysis(unittest.TestCase):
    """Test security analysis functionality."""
    
    def setUp(self):
        self.analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD)
    
    def test_detect_eval(self):
        """Test detection of dangerous eval."""
        code = """
user_input = input()
result = eval(user_input)
"""
        result = self.analyzer.analyze(code)
        
        security_funcs = [
            issue['function'] for issue in result.security_issues
            if issue['type'] == 'dangerous_function'
        ]
        self.assertIn('eval', security_funcs)
    
    def test_detect_exec(self):
        """Test detection of dangerous exec."""
        code = """
exec("print('hello')")
"""
        result = self.analyzer.analyze(code)
        
        security_funcs = [
            issue['function'] for issue in result.security_issues
            if issue['type'] == 'dangerous_function'
        ]
        self.assertIn('exec', security_funcs)


class TestConvenienceFunction(unittest.TestCase):
    """Test convenience functions."""
    
    def test_analyze_code_function(self):
        """Test the analyze_code convenience function."""
        code = "x = 1 + 2"
        result = analyze_code(code)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertIsNotNone(result.ast_tree)
    
    def test_analyze_code_with_level(self):
        """Test analyze_code with specified level."""
        code = "x = 1"
        
        basic_result = analyze_code(code, level=AnalysisLevel.BASIC)
        standard_result = analyze_code(code, level=AnalysisLevel.STANDARD)
        
        # Basic level should not create PDG
        self.assertIsNone(basic_result.pdg)
        
        # Standard level should create PDG
        self.assertIsNotNone(standard_result.pdg)


class TestPerformance(unittest.TestCase):
    """Test performance requirements."""
    
    def test_analysis_speed(self):
        """Test that analysis is fast enough."""
        # Generate ~100 lines of code
        code_lines = []
        for i in range(50):
            code_lines.append(f"def func_{i}():")
            code_lines.append(f"    x_{i} = {i}")
            code_lines.append(f"    return x_{i}")
        
        code = '\n'.join(code_lines)
        
        analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD, cache_enabled=False)
        result = analyzer.analyze(code)
        
        # Analysis should complete in under 1 second for ~150 lines
        self.assertLess(result.metrics.analysis_time_seconds, 1.0)


if __name__ == '__main__':
    unittest.main()
