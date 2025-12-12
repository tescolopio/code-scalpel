"""
TypeScript/JavaScript Code Analyzer.

Provides higher-level analysis on top of the parser, including:
- Structural analysis (functions, classes, imports)
- Complexity metrics
- Security pattern detection
- Dependency graph construction

Status: STUB - Core interface only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .parser import TypeScriptParser, TSParseResult


@dataclass
class TSAnalysisResult:
    """Result of TypeScript/JavaScript code analysis."""
    
    success: bool
    language: str = "typescript"
    
    # Structural metrics
    num_functions: int = 0
    num_classes: int = 0
    num_imports: int = 0
    num_exports: int = 0
    
    # Complexity metrics
    cyclomatic_complexity: int = 0
    max_nesting_depth: int = 0
    lines_of_code: int = 0
    
    # Extracted items
    functions: list[dict[str, Any]] = field(default_factory=list)
    classes: list[dict[str, Any]] = field(default_factory=list)
    imports: list[dict[str, Any]] = field(default_factory=list)
    exports: list[dict[str, Any]] = field(default_factory=list)
    
    # Security findings
    security_issues: list[dict[str, Any]] = field(default_factory=list)
    
    # Errors
    errors: list[str] = field(default_factory=list)


class TypeScriptAnalyzer:
    """
    Analyzer for TypeScript and JavaScript code.
    
    Provides structural analysis, complexity metrics, and security scanning
    for TypeScript/JavaScript source code.
    
    Example:
        >>> analyzer = TypeScriptAnalyzer()
        >>> result = analyzer.analyze('''
        ... export async function fetchUser(id: string): Promise<User> {
        ...     const response = await fetch(`/api/users/${id}`);
        ...     return response.json();
        ... }
        ... ''')
        >>> print(f"Functions: {result.num_functions}")
        Functions: 1
    """
    
    def __init__(self, language: str = "typescript"):
        """
        Initialize the analyzer.
        
        Args:
            language: "typescript" or "javascript"
        """
        self.parser = TypeScriptParser(language)
        self.language = language
    
    def analyze(self, code: str, filename: str | None = None) -> TSAnalysisResult:
        """
        Analyze TypeScript/JavaScript source code.
        
        Args:
            code: Source code to analyze
            filename: Optional filename for context
            
        Returns:
            TSAnalysisResult with metrics and extracted items
        """
        # Parse the code
        parse_result = self.parser.parse(code, filename)
        
        if not parse_result.success:
            return TSAnalysisResult(
                success=False,
                language=self.language,
                errors=parse_result.errors
            )
        
        # Calculate metrics
        lines = code.split('\n')
        loc = len([l for l in lines if l.strip() and not l.strip().startswith('//')])
        
        # Estimate complexity (stub - would use AST for real calculation)
        complexity = self._estimate_complexity(code)
        nesting = self._estimate_nesting(code)
        
        # Security scan (stub)
        security_issues = self._scan_security(code, parse_result)
        
        return TSAnalysisResult(
            success=True,
            language=parse_result.language,
            num_functions=len(parse_result.functions),
            num_classes=len(parse_result.classes),
            num_imports=len(parse_result.imports),
            num_exports=len(parse_result.exports),
            cyclomatic_complexity=complexity,
            max_nesting_depth=nesting,
            lines_of_code=loc,
            functions=parse_result.functions,
            classes=parse_result.classes,
            imports=parse_result.imports,
            exports=parse_result.exports,
            security_issues=security_issues
        )
    
    def analyze_file(self, filepath: str | Path) -> TSAnalysisResult:
        """
        Analyze a TypeScript/JavaScript file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            TSAnalysisResult with metrics and extracted items
        """
        path = Path(filepath)
        if not path.exists():
            return TSAnalysisResult(
                success=False,
                errors=[f"File not found: {filepath}"]
            )
        
        code = path.read_text(encoding='utf-8')
        return self.analyze(code, str(path))
    
    def _estimate_complexity(self, code: str) -> int:
        """
        Estimate cyclomatic complexity from code.
        
        STUB: Real implementation would use AST for accurate calculation.
        """
        import re
        
        # Count decision points (rough estimate)
        patterns = [
            r'\bif\s*\(',
            r'\belse\s+if\s*\(',
            r'\bwhile\s*\(',
            r'\bfor\s*\(',
            r'\bcase\s+',
            r'\bcatch\s*\(',
            r'\?\s*[^:]+:',  # Ternary
            r'\&\&',
            r'\|\|',
            r'\?\?',  # Nullish coalescing
        ]
        
        complexity = 1  # Base complexity
        for pattern in patterns:
            complexity += len(re.findall(pattern, code))
        
        return complexity
    
    def _estimate_nesting(self, code: str) -> int:
        """
        Estimate maximum nesting depth.
        
        STUB: Real implementation would use AST.
        """
        max_depth = 0
        current_depth = 0
        
        for char in code:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _scan_security(
        self, 
        code: str, 
        parse_result: TSParseResult
    ) -> list[dict[str, Any]]:
        """
        Scan for common security issues in TypeScript/JavaScript.
        
        STUB: Real implementation would use taint analysis.
        
        Patterns detected:
        - eval() usage
        - innerHTML assignment
        - SQL string concatenation
        - Hardcoded secrets
        - Unsafe regex
        """
        import re
        
        issues: list[dict[str, Any]] = []
        lines = code.split('\n')
        
        # Dangerous patterns
        dangerous_patterns = [
            (r'\beval\s*\(', 'eval-usage', 'CWE-95', 'HIGH',
             'Use of eval() can lead to code injection'),
            (r'\.innerHTML\s*=', 'innerhtml-xss', 'CWE-79', 'HIGH',
             'innerHTML assignment may cause XSS'),
            (r'document\.write\s*\(', 'document-write', 'CWE-79', 'MEDIUM',
             'document.write can cause XSS'),
            (r'dangerouslySetInnerHTML', 'react-dangerous-html', 'CWE-79', 'MEDIUM',
             'dangerouslySetInnerHTML may cause XSS'),
            (r'new\s+Function\s*\(', 'function-constructor', 'CWE-95', 'HIGH',
             'Function constructor can lead to code injection'),
            (r'child_process\.exec\s*\(', 'command-injection', 'CWE-78', 'CRITICAL',
             'exec() may allow command injection'),
            (r'\.query\s*\(\s*[`"\'][^`"\']*\$\{', 'sql-injection', 'CWE-89', 'CRITICAL',
             'SQL query with template literal may allow injection'),
            (r'(api[_-]?key|secret|password|token)\s*[=:]\s*[\'"][^\'"]{8,}[\'"]',
             'hardcoded-secret', 'CWE-798', 'HIGH',
             'Possible hardcoded secret detected'),
        ]
        
        for line_num, line in enumerate(lines, 1):
            for pattern, issue_type, cwe, severity, message in dangerous_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        'type': issue_type,
                        'cwe': cwe,
                        'severity': severity,
                        'message': message,
                        'line': line_num,
                        'code': line.strip()[:100]
                    })
        
        return issues


# Normalized IR for cross-language analysis
@dataclass
class NormalizedFunction:
    """
    Normalized function representation for cross-language analysis.
    
    This structure is language-agnostic and can represent functions
    from Python, TypeScript, Java, etc.
    """
    name: str
    parameters: list[dict[str, Any]]
    return_type: str | None
    body_start_line: int
    body_end_line: int
    is_async: bool = False
    is_generator: bool = False
    is_exported: bool = False
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    complexity: int = 1
    
    # Source language
    source_language: str = "unknown"


@dataclass  
class NormalizedClass:
    """
    Normalized class representation for cross-language analysis.
    """
    name: str
    base_classes: list[str]
    interfaces: list[str]  # TypeScript interfaces / Java implements
    methods: list[NormalizedFunction]
    properties: list[dict[str, Any]]
    start_line: int
    end_line: int
    is_abstract: bool = False
    is_exported: bool = False
    docstring: str | None = None
    
    # Source language
    source_language: str = "unknown"


def normalize_typescript_function(ts_func: dict[str, Any]) -> NormalizedFunction:
    """Convert TypeScript function dict to normalized representation."""
    return NormalizedFunction(
        name=ts_func.get('name', 'anonymous'),
        parameters=[],  # Would parse from params string
        return_type=ts_func.get('return_type'),
        body_start_line=ts_func.get('line', 0),
        body_end_line=ts_func.get('end_line', ts_func.get('line', 0)),
        is_async=ts_func.get('is_async', False),
        is_exported=ts_func.get('is_exported', False),
        source_language="typescript"
    )


def normalize_typescript_class(ts_class: dict[str, Any]) -> NormalizedClass:
    """Convert TypeScript class dict to normalized representation."""
    return NormalizedClass(
        name=ts_class.get('name', 'Anonymous'),
        base_classes=[ts_class['extends']] if ts_class.get('extends') else [],
        interfaces=ts_class.get('implements', '').split(',') if ts_class.get('implements') else [],
        methods=[],  # Would extract from class body
        properties=[],
        start_line=ts_class.get('line', 0),
        end_line=ts_class.get('end_line', ts_class.get('line', 0)),
        is_exported=ts_class.get('is_exported', False),
        source_language="typescript"
    )
