from ..base_parser import BaseParser, ParseResult, PreprocessorConfig, Language
import ast
from typing import Dict, List, Optional, Any
from collections import defaultdict
import tokenize
from io import StringIO
import subprocess
import json

class PythonParser(BaseParser):
    def parse_code(self, code: str, preprocess: bool = True, config: Optional[PreprocessorConfig] = None) -> ParseResult:
        if preprocess:
            code = self._preprocess_code(code, config or PreprocessorConfig())
        return self._parse_python(code)

    def _preprocess_code(self, code: str, config: PreprocessorConfig) -> str:
        # Python-specific preprocessing logic
        if config.remove_comments:
            code = self._remove_comments(code, Language.PYTHON)
        if config.normalize_whitespace:
            code = self._normalize_whitespace(code)
        return code

    def _parse_python(self, code: str) -> ParseResult:
        """Parse Python code with detailed analysis."""
        errors = []
        warnings = []
        metrics = defaultdict(int)
        
        try:
            # Parse into AST
            tree = ast.parse(code)
            
            # Get token stream
            tokens = list(tokenize.generate_tokens(StringIO(code).readline))
            
            # Analyze code structure
            metrics.update(self._analyze_python_code(tree))
            
            # Check for potential issues using prospector
            warnings.extend(self._check_python_code_with_prospector(code))
            
            return ParseResult(
                ast=tree,
                errors=errors,
                warnings=warnings,
                tokens=tokens,
                metrics=dict(metrics),
                language=Language.PYTHON
            )
            
        except SyntaxError as e:
            errors.append(self._format_syntax_error(e))
            return ParseResult(
                ast=None,
                errors=errors,
                warnings=warnings,
                tokens=[],
                metrics=dict(metrics),
                language=Language.PYTHON
            )
        
    def _analyze_python_code(self, tree: ast.AST) -> Dict[str, int]:
        """Analyze Python code structure."""
        metrics = defaultdict(int)
        
        for node in ast.walk(tree):
            # Count different node types
            metrics[f'count_{type(node).__name__}'] += 1
            
            # Analyze complexity
            if isinstance(node, ast.FunctionDef):
                metrics['function_count'] += 1
                metrics['max_function_complexity'] = max(
                    metrics['max_function_complexity'],
                    self._calculate_complexity(node)
                )
            elif isinstance(node, ast.ClassDef):
                metrics['class_count'] += 1
                
        return dict(metrics)
    
    def _check_python_code_with_prospector(self, code: str) -> List[str]:
        """Check for potential code issues using prospector."""
        warnings = []
        
        # Run prospector as a subprocess
        result = subprocess.run(
            ['prospector', '--output-format', 'json', '--no-autodetect', '--die-on-tool-error', '--absolute-paths', '--path', '-'],
            input=code,
            text=True,
            capture_output=True
        )
        
        if result.returncode != 0 and result.stdout:
            prospector_output = json.loads(result.stdout)
            for issue in prospector_output.get('messages', []):
                warnings.append(
                    f"{issue['code']} at line {issue['location']['line']}, column {issue['location']['character']}: {issue['message']}"
                )
        
        return warnings

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate the complexity of a function node."""
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.And, ast.Or)):
                complexity += 1
        return complexity

    def _format_syntax_error(self, e: SyntaxError) -> Dict[str, Any]:
        """Format a syntax error for inclusion in the error list."""
        return {
            'type': 'SyntaxError',
            'message': e.msg,
            'line': e.lineno,
            'column': e.offset
        }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Python parser using prospector for analysis')
    parser.add_argument('filename', help='Path to the Python file')
    args = parser.parse_args()

    try:
        with open(args.filename, 'r') as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: {args.filename} not found.")
        exit(1)

    python_parser = PythonParser()
    result = python_parser.parse_code(code)
    
    if result.errors:
        print("Errors:")
        for error in result.errors:
            print(error)
    
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(warning)
    
    print("Metrics:")
    for key, value in result.metrics.items():
        print(f"{key}: {value}")