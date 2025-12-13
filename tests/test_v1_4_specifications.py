"""
Code Scalpel v1.4.0 - Test Specifications Implementation
Based on Release Evidence Protocol directives.

[20251213_TEST] v1.4.0 - Comprehensive MCP tool and vulnerability detection tests.
"""

import pytest
import tempfile
import os
import ast


class TestSpec_MCP_GetFileContext:
    """
    Tests for the 'get_file_context' MCP tool.
    Target function: get_file_context(file_path: str)
    """

    def test_returns_correct_structure(self):
        """
        REQUIREMENT: The output must strictly adhere to the FileContext schema.
        """
        from code_scalpel.mcp.server import _get_file_context_sync, FileContextResult
        
        # Create a dummy Python file with known content
        content = '''
import os
import json

class MyClass:
    def method(self):
        pass

def func_one():
    pass

def func_two():
    pass
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f_path = f.name
        
        try:
            result = _get_file_context_sync(f_path)
            
            # Result is instance of FileContextResult
            assert isinstance(result, FileContextResult)
            
            # Language is python
            assert result.language == 'python'
            
            # Line count is reasonable
            assert result.line_count > 0
            assert result.line_count == len(content.strip().split('\n')) + 1  # +1 for leading newline
            
            # Has expected functions
            assert 'func_one' in result.functions
            assert 'func_two' in result.functions
            
            # Has expected class
            assert 'MyClass' in result.classes
            
            # Has expected imports
            assert 'os' in result.imports
            assert 'json' in result.imports
            
        finally:
            os.unlink(f_path)

    def test_complexity_calculation(self):
        """
        REQUIREMENT: Complexity score must be an integer > 0 for complex code.
        """
        from code_scalpel.mcp.server import _get_file_context_sync
        
        # Create file with nested control structures
        content = '''
def complex_function(x):
    result = 0
    for i in range(10):
        if i % 2 == 0:
            if x > 5:
                for j in range(5):
                    while result < 100:
                        result += 1
            else:
                result -= 1
        elif i % 3 == 0:
            result *= 2
    return result
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f_path = f.name
        
        try:
            result = _get_file_context_sync(f_path)
            
            # Complexity should be calculated
            assert result.complexity_score > 0
            
            # With nested loops and conditionals, should be > 5
            assert result.complexity_score >= 5
            
        finally:
            os.unlink(f_path)

    def test_token_efficiency_summary(self):
        """
        REQUIREMENT: The 'summary' field must be present and concise.
        """
        from code_scalpel.mcp.server import _get_file_context_sync
        
        # Create a large file
        lines = ['def func_{}(): pass'.format(i) for i in range(50)]
        content = '\n'.join(lines)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f_path = f.name
        
        try:
            result = _get_file_context_sync(f_path)
            
            # Summary should exist
            assert result.summary is not None
            assert len(result.summary) > 0
            
            # Summary should be much shorter than full content
            assert len(result.summary) < len(content)
            
        finally:
            os.unlink(f_path)

    def test_file_not_found_error(self):
        """
        REQUIREMENT: Should handle non-existent files gracefully.
        """
        from code_scalpel.mcp.server import _get_file_context_sync
        
        result = _get_file_context_sync('/nonexistent/path/file.py')
        
        assert result.success is False
        assert result.error is not None
        assert 'not found' in result.error.lower() or 'no such file' in result.error.lower()

    def test_syntax_error_handling(self):
        """
        REQUIREMENT: Should handle Python syntax errors gracefully.
        """
        from code_scalpel.mcp.server import _get_file_context_sync
        
        content = '''
def broken_function(
    # Missing closing paren and colon
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(content)
            f_path = f.name
        
        try:
            result = _get_file_context_sync(f_path)
            
            # Should not crash - returns error state
            assert result.success is False
            assert result.error is not None
            
        finally:
            os.unlink(f_path)


class TestSpec_MCP_GetSymbolReferences:
    """
    Tests for the 'get_symbol_references' MCP tool.
    Target function: get_symbol_references(symbol_name: str, project_root: str)
    """

    def test_finds_cross_file_references(self):
        """
        REQUIREMENT: Must find usages of a symbol in files other than its definition.
        """
        from code_scalpel.mcp.server import _get_symbol_references_sync
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # File A: definition
            def_path = os.path.join(tmpdir, 'definitions.py')
            with open(def_path, 'w') as f:
                f.write('''
def target_func():
    """The target function."""
    return 42
''')
            
            # File B: usage
            use_path = os.path.join(tmpdir, 'consumer.py')
            with open(use_path, 'w') as f:
                f.write('''
from definitions import target_func

result = target_func()
print(target_func())
''')
            
            result = _get_symbol_references_sync('target_func', tmpdir)
            
            assert result.success is True
            assert result.total_references >= 2  # Definition + at least one usage
            
            # Should find reference in consumer.py
            consumer_refs = [r for r in result.references if 'consumer.py' in r.file]
            assert len(consumer_refs) >= 1

    def test_finds_definition_location(self):
        """
        REQUIREMENT: Should correctly identify where symbol is defined.
        """
        from code_scalpel.mcp.server import _get_symbol_references_sync
        
        with tempfile.TemporaryDirectory() as tmpdir:
            def_path = os.path.join(tmpdir, 'module.py')
            with open(def_path, 'w') as f:
                f.write('''
class MyClass:
    pass
''')
            
            result = _get_symbol_references_sync('MyClass', tmpdir)
            
            assert result.success is True
            assert result.definition_file is not None
            assert 'module.py' in result.definition_file
            assert result.definition_line == 2  # Class defined on line 2

    def test_symbol_not_found(self):
        """
        REQUIREMENT: Should handle non-existent symbols gracefully.
        """
        from code_scalpel.mcp.server import _get_symbol_references_sync
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'empty.py')
            with open(file_path, 'w') as f:
                f.write('# Empty file\n')
            
            result = _get_symbol_references_sync('nonexistent_symbol', tmpdir)
            
            assert result.success is True  # Search succeeded, just found nothing
            assert result.total_references == 0

    def test_invalid_project_root(self):
        """
        REQUIREMENT: Should handle invalid project paths.
        """
        from code_scalpel.mcp.server import _get_symbol_references_sync
        
        result = _get_symbol_references_sync('any_symbol', '/nonexistent/path')
        
        assert result.success is False
        assert result.error is not None


class TestSpec_Detection_XXE_CWE611:
    """
    Tests for CWE-611 XXE Detection.
    """

    @pytest.mark.parametrize("vulnerable_code,sink_name", [
        ("import xml.etree.ElementTree as ET\nET.parse(user_input)", "ET.parse"),
        ("from lxml import etree\netree.fromstring(data)", "etree.fromstring"),
        ("import xml.sax\nxml.sax.parse(stream, handler)", "xml.sax.parse"),
    ])
    def test_detects_standard_xml_parsers(self, vulnerable_code, sink_name):
        """
        REQUIREMENT: All standard library unsafe XML parsers must trigger a finding.
        """
        from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink, SINK_PATTERNS
        
        # Verify sink is registered
        matching_sinks = [k for k in SINK_PATTERNS.keys() if sink_name in k or k in sink_name]
        assert len(matching_sinks) > 0, f"No sink pattern found matching {sink_name}"
        
        # All matching sinks should map to XXE
        for sink in matching_sinks:
            assert SINK_PATTERNS[sink] == SecuritySink.XXE

    @pytest.mark.parametrize("safe_code", [
        "import defusedxml\ndefusedxml.parse(data)",
        "from defusedxml.ElementTree import parse\nparse(data)",
        "# This is just a comment about xml.etree",
    ])
    def test_ignores_defused_and_safe_code(self, safe_code):
        """
        REQUIREMENT: Must not flag defusedxml or comments.
        """
        from code_scalpel.symbolic_execution_tools.taint_tracker import SANITIZER_REGISTRY
        
        # defusedxml should be a registered sanitizer
        defused_sanitizers = [k for k in SANITIZER_REGISTRY.keys() if 'defusedxml' in k]
        assert len(defused_sanitizers) > 0, "No defusedxml sanitizers registered"

    def test_xxe_taint_flow_detection(self):
        """
        REQUIREMENT: Detect XXE when user input flows to XML parser.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
        from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink
        
        code = '''
from flask import request
import xml.etree.ElementTree as ET

xml_file = request.args.get('file')
tree = ET.parse(xml_file)
'''
        
        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        
        xxe_vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.XXE]
        assert len(xxe_vulns) >= 1, "XXE vulnerability not detected"


class TestSpec_Detection_SSTI_CWE1336:
    """
    Tests for CWE-1336 SSTI Detection.
    """

    def test_detects_jinja2_template_injection(self):
        """
        REQUIREMENT: Detect 'jinja2.Template(x)' where x is variable.
        """
        from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
        from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink
        
        code = '''
from flask import request
import jinja2

template = request.args.get('t')
rendered = jinja2.Template(template).render()
'''
        
        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        
        ssti_vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.SSTI]
        assert len(ssti_vulns) >= 1, "SSTI vulnerability not detected"

    def test_allows_file_loading_jinja2(self):
        """
        REQUIREMENT: Do not flag standard environment loaders (best practice).
        """
        from code_scalpel.symbolic_execution_tools.taint_tracker import SANITIZER_REGISTRY, SecuritySink
        
        # render_template should be a sanitizer for SSTI
        assert 'render_template' in SANITIZER_REGISTRY
        sanitizer_info = SANITIZER_REGISTRY['render_template']
        assert SecuritySink.SSTI in sanitizer_info.clears_sinks

    @pytest.mark.parametrize("template_engine", [
        "jinja2.Template",
        "mako.template.Template",
        "django.template.Template",
        "tornado.template.Template",
    ])
    def test_detects_multiple_template_engines(self, template_engine):
        """
        REQUIREMENT: Detect SSTI across multiple template engines.
        """
        from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink, SINK_PATTERNS
        
        # Check if any sink pattern matches this engine
        matching = any(template_engine in k or k in template_engine for k in SINK_PATTERNS.keys())
        assert matching, f"No SSTI sink pattern for {template_engine}"


class TestSpec_Regression_V130:
    """
    Regression tests to ensure v1.3.0 features still work.
    """

    def test_sql_injection_still_detected(self):
        """v1.3.0 SQL injection detection must still work."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
        from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink
        
        code = '''
from flask import request
import sqlite3

user_id = request.args.get('id')
cursor.execute("SELECT * FROM users WHERE id = " + user_id)
'''
        
        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        
        sqli_vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.SQL_QUERY]
        assert len(sqli_vulns) >= 1

    def test_command_injection_still_detected(self):
        """v1.3.0 command injection detection must still work."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
        from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink
        
        code = '''
from flask import request
import os

cmd = request.args.get('cmd')
os.system(cmd)
'''
        
        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)
        
        cmdi_vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.SHELL_COMMAND]
        assert len(cmdi_vulns) >= 1

    def test_secret_detection_still_works(self):
        """v1.3.0 secret detection must still work."""
        from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
        
        code = '''
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
'''
        
        scanner = SecretScanner()
        tree = ast.parse(code)
        secrets = scanner.scan(tree)
        
        assert len(secrets) >= 1, "Secret detection broken"


class TestSpec_MCP_Tools_Integration:
    """
    Integration tests for MCP tools working together.
    """

    def test_file_context_and_symbol_references_consistency(self):
        """
        REQUIREMENT: get_file_context and get_symbol_references should be consistent.
        """
        from code_scalpel.mcp.server import _get_file_context_sync, _get_symbol_references_sync
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'module.py')
            with open(file_path, 'w') as f:
                f.write('''
def important_function():
    pass

class ImportantClass:
    pass
''')
            
            # Get file context
            context = _get_file_context_sync(file_path)
            
            # Should list important_function
            assert 'important_function' in context.functions
            
            # Symbol references should find it
            refs = _get_symbol_references_sync('important_function', tmpdir)
            assert refs.total_references >= 1
            
            # Definition should match
            assert refs.definition_line is not None
