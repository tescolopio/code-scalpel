"""
CodeAnalyzer - Stable analysis pipeline for code-scalpel.

This module provides a unified interface for:
- AST analysis and parsing
- PDG (Program Dependence Graph) construction
- Symbolic execution for path analysis
- Dead code detection
- Refactoring via PDG-guided transformations
"""

import ast
import time
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import astor

# Configure module logger
logger = logging.getLogger(__name__)


class AnalysisLevel(Enum):
    """Level of analysis to perform."""
    BASIC = 'basic'        # AST only
    STANDARD = 'standard'  # AST + PDG
    FULL = 'full'          # AST + PDG + Symbolic Execution


@dataclass
class DeadCodeItem:
    """Represents a detected dead code element."""
    name: str
    code_type: str  # 'function', 'variable', 'class', 'import', 'statement'
    line_start: int
    line_end: int
    reason: str
    confidence: float  # 0.0 to 1.0


@dataclass
class AnalysisMetrics:
    """Metrics from code analysis."""
    lines_of_code: int = 0
    num_functions: int = 0
    num_classes: int = 0
    num_variables: int = 0
    cyclomatic_complexity: int = 0
    analysis_time_seconds: float = 0.0


@dataclass
class RefactorSuggestion:
    """A suggested refactoring operation."""
    refactor_type: str
    description: str
    target_node: str
    priority: int  # 1-5, higher is more important
    estimated_impact: str


@dataclass 
class AnalysisResult:
    """Complete result of code analysis."""
    code: str
    ast_tree: Optional[ast.AST] = None
    pdg: Optional[nx.DiGraph] = None
    call_graph: Optional[nx.DiGraph] = None
    dead_code: List[DeadCodeItem] = field(default_factory=list)
    metrics: AnalysisMetrics = field(default_factory=AnalysisMetrics)
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    refactor_suggestions: List[RefactorSuggestion] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    symbolic_paths: List[Dict[str, Any]] = field(default_factory=list)


class CodeAnalyzer:
    """
    Unified code analysis pipeline with AST, PDG, and symbolic execution.
    
    This class provides a stable interface for comprehensive code analysis
    including dead code detection and PDG-guided refactoring.
    
    Example usage:
        analyzer = CodeAnalyzer()
        result = analyzer.analyze(code)
        print(result.dead_code)
        
        # Apply refactoring
        new_code = analyzer.apply_refactor(code, 'remove_dead_code')
    """
    
    def __init__(self, 
                 level: AnalysisLevel = AnalysisLevel.STANDARD,
                 cache_enabled: bool = True,
                 max_symbolic_depth: int = 50,
                 max_loop_iterations: int = 10):
        """
        Initialize the CodeAnalyzer.
        
        Args:
            level: Analysis level (BASIC, STANDARD, or FULL)
            cache_enabled: Whether to cache analysis results
            max_symbolic_depth: Maximum depth for symbolic execution
            max_loop_iterations: Maximum loop iterations for symbolic execution
        """
        self.level = level
        self.cache_enabled = cache_enabled
        self.max_symbolic_depth = max_symbolic_depth
        self.max_loop_iterations = max_loop_iterations
        
        # Caches
        self._ast_cache: Dict[str, ast.AST] = {}
        self._pdg_cache: Dict[str, Tuple[nx.DiGraph, nx.DiGraph]] = {}
        self._analysis_cache: Dict[str, AnalysisResult] = {}
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CodeAnalyzer')

    def analyze(self, code: str, level: Optional[AnalysisLevel] = None) -> AnalysisResult:
        """
        Perform comprehensive code analysis.
        
        Args:
            code: Python source code to analyze
            level: Override the default analysis level
            
        Returns:
            AnalysisResult containing all analysis data
        """
        start_time = time.time()
        analysis_level = level or self.level
        
        # Check cache
        cache_key = f"{hash(code)}_{analysis_level.value}"
        if self.cache_enabled and cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]
        
        result = AnalysisResult(code=code)
        
        try:
            # Step 1: Parse to AST
            result.ast_tree = self._parse_to_ast(code)
            if result.ast_tree is None:
                result.errors.append("Failed to parse code to AST")
                return result
            
            # Compute basic metrics
            result.metrics = self._compute_metrics(result.ast_tree, code)
            
            # Step 2: Build PDG (if STANDARD or FULL level)
            if analysis_level in (AnalysisLevel.STANDARD, AnalysisLevel.FULL):
                result.pdg, result.call_graph = self._build_pdg(code)
            
            # Step 3: Symbolic Execution (if FULL level)
            if analysis_level == AnalysisLevel.FULL:
                result.symbolic_paths = self._run_symbolic_execution(code)
            
            # Step 4: Dead code detection
            result.dead_code = self._detect_dead_code(
                result.ast_tree, 
                result.pdg,
                result.call_graph
            )
            
            # Step 5: Security analysis
            result.security_issues = self._analyze_security(result.ast_tree)
            
            # Step 6: Generate refactoring suggestions
            result.refactor_suggestions = self._generate_refactor_suggestions(
                result.ast_tree,
                result.pdg,
                result.dead_code
            )
            
        except SyntaxError as e:
            result.errors.append(f"Syntax error: {str(e)}")
        except Exception as e:
            result.errors.append(f"Analysis error: {str(e)}")
            self.logger.error(f"Analysis failed: {str(e)}")
        
        # Record analysis time
        result.metrics.analysis_time_seconds = time.time() - start_time
        
        # Cache result
        if self.cache_enabled:
            self._analysis_cache[cache_key] = result
        
        return result

    def _parse_to_ast(self, code: str) -> Optional[ast.AST]:
        """Parse code to AST with caching."""
        if self.cache_enabled and code in self._ast_cache:
            return self._ast_cache[code]
        
        try:
            tree = ast.parse(code)
            if self.cache_enabled:
                self._ast_cache[code] = tree
            return tree
        except SyntaxError as e:
            self.logger.error(f"Parse error at line {e.lineno}: {e.msg}")
            return None

    def _build_pdg(self, code: str) -> Tuple[nx.DiGraph, nx.DiGraph]:
        """Build Program Dependence Graph and Call Graph."""
        if self.cache_enabled and code in self._pdg_cache:
            return self._pdg_cache[code]
        
        tree = self._parse_to_ast(code)
        if tree is None:
            return nx.DiGraph(), nx.DiGraph()
        
        builder = self._PDGBuilder()
        builder.visit(tree)
        
        result = (builder.graph, builder.call_graph)
        if self.cache_enabled:
            self._pdg_cache[code] = result
        
        return result

    def _compute_metrics(self, tree: ast.AST, code: str) -> AnalysisMetrics:
        """Compute code metrics from AST."""
        metrics = AnalysisMetrics()
        
        # Count lines of code (non-empty, non-comment)
        lines = [line for line in code.split('\n') 
                 if line.strip() and not line.strip().startswith('#')]
        metrics.lines_of_code = len(lines)
        
        # Walk AST to count elements
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics.num_functions += 1
            elif isinstance(node, ast.AsyncFunctionDef):
                metrics.num_functions += 1
            elif isinstance(node, ast.ClassDef):
                metrics.num_classes += 1
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                metrics.num_variables += 1
        
        # Calculate cyclomatic complexity
        metrics.cyclomatic_complexity = self._calculate_complexity(tree)
        
        return metrics

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of code."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.Assert, ast.With)):
                complexity += 1
        
        return complexity

    def _run_symbolic_execution(self, code: str) -> List[Dict[str, Any]]:
        """Run symbolic execution on code."""
        paths = []
        
        try:
            tree = self._parse_to_ast(code)
            if tree is None:
                return paths
            
            # Simple path collection - track if/else branches
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    path_info = {
                        'type': 'conditional',
                        'line': node.lineno,
                        'condition': ast.unparse(node.test),
                        'has_else': len(node.orelse) > 0
                    }
                    paths.append(path_info)
                elif isinstance(node, ast.While):
                    path_info = {
                        'type': 'loop',
                        'line': node.lineno,
                        'condition': ast.unparse(node.test)
                    }
                    paths.append(path_info)
                    
        except Exception as e:
            self.logger.warning(f"Symbolic execution limited: {str(e)}")
        
        return paths

    def _detect_dead_code(self, 
                          tree: ast.AST,
                          pdg: Optional[nx.DiGraph],
                          call_graph: Optional[nx.DiGraph]) -> List[DeadCodeItem]:
        """
        Detect dead code in the AST.
        
        This identifies:
        - Unused functions
        - Unused variables
        - Unreachable code after return/raise
        - Unused imports
        """
        dead_code = []
        
        # Collect all definitions and uses
        definitions = self._collect_definitions(tree)
        uses = self._collect_uses(tree)
        
        # Check for unused functions
        for func_name, func_info in definitions.get('functions', {}).items():
            if func_name not in uses.get('function_calls', set()):
                # Skip if it's a special method or main entry point
                if not func_name.startswith('_') and func_name != 'main':
                    dead_code.append(DeadCodeItem(
                        name=func_name,
                        code_type='function',
                        line_start=func_info['line_start'],
                        line_end=func_info['line_end'],
                        reason='Function is defined but never called',
                        confidence=0.9
                    ))
        
        # Check for unused variables
        defined_vars = definitions.get('variables', {})
        used_vars = uses.get('variables', set())
        
        for var_name, var_info in defined_vars.items():
            if var_name not in used_vars:
                # Skip loop variables and private variables
                if not var_name.startswith('_'):
                    dead_code.append(DeadCodeItem(
                        name=var_name,
                        code_type='variable',
                        line_start=var_info['line'],
                        line_end=var_info['line'],
                        reason='Variable is assigned but never used',
                        confidence=0.85
                    ))
        
        # Check for unreachable code after return/raise
        dead_code.extend(self._find_unreachable_code(tree))
        
        # Check for unused imports
        dead_code.extend(self._find_unused_imports(tree, uses))
        
        # Use PDG for additional analysis if available
        if pdg is not None:
            dead_code.extend(self._detect_dead_code_from_pdg(pdg))
        
        return dead_code

    def _collect_definitions(self, tree: ast.AST) -> Dict[str, Any]:
        """Collect all definitions in the code."""
        definitions = {
            'functions': {},
            'classes': {},
            'variables': {},
            'imports': {}
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                definitions['functions'][node.name] = {
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'args': [arg.arg for arg in node.args.args]
                }
            elif isinstance(node, ast.AsyncFunctionDef):
                definitions['functions'][node.name] = {
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'args': [arg.arg for arg in node.args.args]
                }
            elif isinstance(node, ast.ClassDef):
                definitions['classes'][node.name] = {
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno
                }
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        definitions['variables'][target.id] = {
                            'line': node.lineno
                        }
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    definitions['imports'][name] = {
                        'line': node.lineno,
                        'module': alias.name
                    }
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    definitions['imports'][name] = {
                        'line': node.lineno,
                        'module': f"{node.module}.{alias.name}" if node.module else alias.name
                    }
        
        return definitions

    def _collect_uses(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Collect all uses of names in the code."""
        uses = {
            'function_calls': set(),
            'variables': set(),
            'imports': set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    uses['function_calls'].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    uses['function_calls'].add(node.func.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                uses['variables'].add(node.id)
                uses['imports'].add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    uses['imports'].add(node.value.id)
        
        return uses

    def _find_unreachable_code(self, tree: ast.AST) -> List[DeadCodeItem]:
        """Find code that appears after return/raise statements."""
        dead_code = []
        
        class UnreachableVisitor(ast.NodeVisitor):
            def __init__(self):
                self.dead_items = []
            
            def visit_FunctionDef(self, node):
                self._check_body(node.body)
                self.generic_visit(node)
            
            def visit_AsyncFunctionDef(self, node):
                self._check_body(node.body)
                self.generic_visit(node)
            
            def _check_body(self, body):
                found_terminal = False
                terminal_line = 0
                
                for stmt in body:
                    if found_terminal:
                        self.dead_items.append(DeadCodeItem(
                            name=f"statement at line {stmt.lineno}",
                            code_type='statement',
                            line_start=stmt.lineno,
                            line_end=getattr(stmt, 'end_lineno', stmt.lineno),
                            reason=f'Unreachable code after line {terminal_line}',
                            confidence=0.95
                        ))
                    
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        found_terminal = True
                        terminal_line = stmt.lineno
        
        visitor = UnreachableVisitor()
        visitor.visit(tree)
        dead_code.extend(visitor.dead_items)
        
        return dead_code

    def _find_unused_imports(self, tree: ast.AST, uses: Dict[str, Set[str]]) -> List[DeadCodeItem]:
        """Find imports that are never used."""
        dead_code = []
        used_names = uses.get('imports', set())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    if name not in used_names:
                        dead_code.append(DeadCodeItem(
                            name=alias.name,
                            code_type='import',
                            line_start=node.lineno,
                            line_end=node.lineno,
                            reason='Import is never used',
                            confidence=0.9
                        ))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name not in used_names and alias.name != '*':
                        dead_code.append(DeadCodeItem(
                            name=f"{node.module}.{alias.name}" if node.module else alias.name,
                            code_type='import',
                            line_start=node.lineno,
                            line_end=node.lineno,
                            reason='Import is never used',
                            confidence=0.9
                        ))
        
        return dead_code

    def _detect_dead_code_from_pdg(self, pdg: nx.DiGraph) -> List[DeadCodeItem]:
        """Use PDG to detect additional dead code patterns."""
        dead_code = []
        
        # Find nodes with no outgoing data dependencies (potential dead code)
        for node in pdg.nodes():
            node_data = pdg.nodes[node]
            out_edges = list(pdg.out_edges(node, data=True))
            
            # Check if this is a computation with no uses
            if node_data.get('type') == 'assign':
                has_data_dep = any(
                    edge[2].get('type') == 'data_dependency' 
                    for edge in out_edges
                )
                if not has_data_dep and 'lineno' in node_data:
                    dead_code.append(DeadCodeItem(
                        name=node_data.get('target', node),
                        code_type='variable',
                        line_start=node_data['lineno'],
                        line_end=node_data['lineno'],
                        reason='Assignment has no downstream dependencies (PDG analysis)',
                        confidence=0.8
                    ))
        
        return dead_code

    def _analyze_security(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze code for security issues."""
        issues = []
        
        dangerous_functions = {
            'eval': 'Code injection via eval()',
            'exec': 'Code injection via exec()',
            'compile': 'Potential code injection via compile()',
            '__import__': 'Dynamic import could be dangerous'
        }
        
        dangerous_patterns = {
            'os.system': 'Command injection risk',
            'subprocess.call': 'Command injection risk',
            'subprocess.Popen': 'Command injection risk (use list args)',
            'pickle.loads': 'Deserialization vulnerability',
            'yaml.load': 'YAML deserialization vulnerability'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check direct function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in dangerous_functions:
                        issues.append({
                            'type': 'dangerous_function',
                            'function': func_name,
                            'line': node.lineno,
                            'severity': 'high',
                            'description': dangerous_functions[func_name]
                        })
                
                # Check method calls
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        full_name = f"{node.func.value.id}.{node.func.attr}"
                        if full_name in dangerous_patterns:
                            issues.append({
                                'type': 'dangerous_pattern',
                                'pattern': full_name,
                                'line': node.lineno,
                                'severity': 'high',
                                'description': dangerous_patterns[full_name]
                            })
        
        return issues

    def _generate_refactor_suggestions(self,
                                        tree: ast.AST,
                                        pdg: Optional[nx.DiGraph],
                                        dead_code: List[DeadCodeItem]) -> List[RefactorSuggestion]:
        """Generate refactoring suggestions based on analysis."""
        suggestions = []
        
        # Suggest removing dead code
        for item in dead_code:
            if item.confidence >= 0.85:
                suggestions.append(RefactorSuggestion(
                    refactor_type='remove_dead_code',
                    description=f"Remove unused {item.code_type}: {item.name}",
                    target_node=f"line_{item.line_start}",
                    priority=4,
                    estimated_impact='Code cleanup, reduced complexity'
                ))
        
        # Analyze for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_length = (node.end_lineno or node.lineno) - node.lineno
                if func_length > 50:
                    suggestions.append(RefactorSuggestion(
                        refactor_type='extract_method',
                        description=f"Function '{node.name}' is {func_length} lines. Consider splitting.",
                        target_node=node.name,
                        priority=3,
                        estimated_impact='Improved readability and maintainability'
                    ))
                
                # Check for deep nesting
                max_depth = self._max_nesting_depth(node)
                if max_depth > 4:
                    suggestions.append(RefactorSuggestion(
                        refactor_type='reduce_nesting',
                        description=f"Function '{node.name}' has nesting depth {max_depth}. Consider flattening.",
                        target_node=node.name,
                        priority=3,
                        estimated_impact='Improved readability'
                    ))
        
        return suggestions

    def _max_nesting_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth in a node."""
        max_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._max_nesting_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = self._max_nesting_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        
        return max_depth

    def apply_refactor(self, 
                       code: str, 
                       refactor_type: str,
                       target: Optional[str] = None,
                       **options) -> str:
        """
        Apply a PDG-guided refactoring to the code.
        
        Args:
            code: Source code to refactor
            refactor_type: Type of refactoring to apply
                - 'remove_dead_code': Remove detected dead code
                - 'remove_unused_imports': Remove unused imports
                - 'rename_variable': Rename a variable (requires target and new_name)
                - 'extract_function': Extract code into new function
            target: Target element for the refactoring
            **options: Additional refactoring options
            
        Returns:
            Refactored source code
        """
        tree = self._parse_to_ast(code)
        if tree is None:
            return code
        
        if refactor_type == 'remove_dead_code':
            return self._remove_dead_code(code, tree)
        elif refactor_type == 'remove_unused_imports':
            return self._remove_unused_imports(code, tree)
        elif refactor_type == 'rename_variable':
            new_name = options.get('new_name')
            if target and new_name:
                return self._rename_variable(code, tree, target, new_name)
        elif refactor_type == 'inline_constant':
            if target:
                return self._inline_constant(code, tree, target)
        
        # No changes made
        return code

    def _remove_dead_code(self, code: str, tree: ast.AST) -> str:
        """Remove dead code from the source."""
        result = self.analyze(code)
        
        if not result.dead_code:
            return code
        
        # Sort by line number descending to remove from bottom up
        dead_items = sorted(result.dead_code, key=lambda x: x.line_start, reverse=True)
        
        lines = code.split('\n')
        
        for item in dead_items:
            if item.confidence >= 0.85:
                # Remove the lines for this dead code item
                start_idx = item.line_start - 1
                end_idx = item.line_end
                
                # Don't remove if it would break indentation structure
                if start_idx >= 0 and end_idx <= len(lines):
                    del lines[start_idx:end_idx]
        
        return '\n'.join(lines)

    def _remove_unused_imports(self, code: str, tree: ast.AST) -> str:
        """Remove unused import statements."""
        uses = self._collect_uses(tree)
        used_names = uses.get('imports', set())
        
        class ImportRemover(ast.NodeTransformer):
            def visit_Import(self, node):
                remaining = []
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    if name in used_names:
                        remaining.append(alias)
                
                if not remaining:
                    return None
                elif len(remaining) < len(node.names):
                    node.names = remaining
                return node
            
            def visit_ImportFrom(self, node):
                if node.names[0].name == '*':
                    return node
                    
                remaining = []
                for alias in node.names:
                    name = alias.asname or alias.name
                    if name in used_names:
                        remaining.append(alias)
                
                if not remaining:
                    return None
                elif len(remaining) < len(node.names):
                    node.names = remaining
                return node
        
        transformer = ImportRemover()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        try:
            return astor.to_source(new_tree)
        except (AttributeError, ValueError, TypeError) as e:
            self.logger.debug(f"Failed to convert AST to source: {e}")
            return code

    def _rename_variable(self, code: str, tree: ast.AST, old_name: str, new_name: str) -> str:
        """Rename a variable throughout the code."""
        
        class VariableRenamer(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == old_name:
                    node.id = new_name
                return node
            
            def visit_arg(self, node):
                if node.arg == old_name:
                    node.arg = new_name
                return node
        
        transformer = VariableRenamer()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        try:
            return astor.to_source(new_tree)
        except (AttributeError, ValueError, TypeError) as e:
            self.logger.debug(f"Failed to convert AST to source: {e}")
            return code

    def _inline_constant(self, code: str, tree: ast.AST, target: str) -> str:
        """Inline a constant variable."""
        # Find the constant value
        constant_value = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == target:
                        if isinstance(node.value, ast.Constant):
                            constant_value = node.value.value
                            break
        
        if constant_value is None:
            return code
        
        class ConstantInliner(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == target and isinstance(node.ctx, ast.Load):
                    return ast.Constant(value=constant_value)
                return node
        
        transformer = ConstantInliner()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        
        try:
            return astor.to_source(new_tree)
        except (AttributeError, ValueError, TypeError) as e:
            self.logger.debug(f"Failed to convert AST to source: {e}")
            return code

    def clear_cache(self):
        """Clear all analysis caches."""
        self._ast_cache.clear()
        self._pdg_cache.clear()
        self._analysis_cache.clear()

    def get_dead_code_summary(self, result: AnalysisResult) -> str:
        """Generate a human-readable summary of dead code."""
        if not result.dead_code:
            return "No dead code detected."
        
        lines = [f"Found {len(result.dead_code)} dead code items:"]
        
        # Group by type
        by_type: Dict[str, List[DeadCodeItem]] = {}
        for item in result.dead_code:
            if item.code_type not in by_type:
                by_type[item.code_type] = []
            by_type[item.code_type].append(item)
        
        for code_type, items in by_type.items():
            lines.append(f"\n{code_type.title()}s ({len(items)}):")
            for item in items:
                confidence_pct = int(item.confidence * 100)
                lines.append(f"  - {item.name} (line {item.line_start}, {confidence_pct}% confidence)")
                lines.append(f"    Reason: {item.reason}")
        
        return '\n'.join(lines)

    class _PDGBuilder(ast.NodeVisitor):
        """Internal PDG builder for the analyzer."""
        
        def __init__(self):
            self.graph = nx.DiGraph()
            self.call_graph = nx.DiGraph()
            self.var_defs: Dict[str, str] = {}
            self.control_deps: List[str] = []
            self.current_function: Optional[str] = None
            self.node_counter = 0
        
        def _get_node_id(self, prefix: str) -> str:
            """Generate unique node ID."""
            self.node_counter += 1
            return f"{prefix}_{self.node_counter}"
        
        def visit_FunctionDef(self, node: ast.FunctionDef):
            node_id = self._get_node_id('func')
            self.graph.add_node(
                node_id,
                type='function',
                name=node.name,
                lineno=node.lineno,
                args=[arg.arg for arg in node.args.args]
            )
            
            self.call_graph.add_node(node.name)
            
            prev_function = self.current_function
            self.current_function = node.name
            
            # Add parameters
            for arg in node.args.args:
                arg_id = self._get_node_id('param')
                self.graph.add_node(
                    arg_id,
                    type='parameter',
                    name=arg.arg,
                    lineno=arg.lineno if hasattr(arg, 'lineno') else node.lineno
                )
                self.graph.add_edge(node_id, arg_id, type='parameter_dependency')
                self.var_defs[arg.arg] = arg_id
            
            # Process body
            for stmt in node.body:
                self.visit(stmt)
            
            self.current_function = prev_function
        
        def visit_Assign(self, node: ast.Assign):
            node_id = self._get_node_id('assign')
            
            target_names = []
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_names.append(target.id)
            
            self.graph.add_node(
                node_id,
                type='assign',
                target=', '.join(target_names),
                lineno=node.lineno
            )
            
            # Add data dependencies
            for var in self._extract_variables(node.value):
                if var in self.var_defs:
                    self.graph.add_edge(
                        self.var_defs[var],
                        node_id,
                        type='data_dependency'
                    )
            
            # Add control dependencies
            for ctrl_node in self.control_deps:
                self.graph.add_edge(ctrl_node, node_id, type='control_dependency')
            
            # Update definitions
            for name in target_names:
                self.var_defs[name] = node_id
            
            self.generic_visit(node)
        
        def visit_If(self, node: ast.If):
            node_id = self._get_node_id('if')
            
            self.graph.add_node(
                node_id,
                type='if',
                lineno=node.lineno
            )
            
            # Add data dependencies for condition
            for var in self._extract_variables(node.test):
                if var in self.var_defs:
                    self.graph.add_edge(
                        self.var_defs[var],
                        node_id,
                        type='data_dependency'
                    )
            
            # Add control dependencies
            for ctrl_node in self.control_deps:
                self.graph.add_edge(ctrl_node, node_id, type='control_dependency')
            
            # Process branches with control dependency
            self.control_deps.append(node_id)
            
            for stmt in node.body:
                self.visit(stmt)
            
            for stmt in node.orelse:
                self.visit(stmt)
            
            self.control_deps.pop()
        
        def visit_For(self, node: ast.For):
            node_id = self._get_node_id('for')
            
            self.graph.add_node(
                node_id,
                type='for',
                lineno=node.lineno
            )
            
            # Handle loop variable
            if isinstance(node.target, ast.Name):
                self.var_defs[node.target.id] = node_id
            
            # Add data dependencies for iterator
            for var in self._extract_variables(node.iter):
                if var in self.var_defs:
                    self.graph.add_edge(
                        self.var_defs[var],
                        node_id,
                        type='data_dependency'
                    )
            
            # Process body
            self.control_deps.append(node_id)
            for stmt in node.body:
                self.visit(stmt)
            self.control_deps.pop()
        
        def visit_While(self, node: ast.While):
            node_id = self._get_node_id('while')
            
            self.graph.add_node(
                node_id,
                type='while',
                lineno=node.lineno
            )
            
            # Add data dependencies for condition
            for var in self._extract_variables(node.test):
                if var in self.var_defs:
                    self.graph.add_edge(
                        self.var_defs[var],
                        node_id,
                        type='data_dependency'
                    )
            
            # Process body
            self.control_deps.append(node_id)
            for stmt in node.body:
                self.visit(stmt)
            self.control_deps.pop()
        
        def visit_Call(self, node: ast.Call):
            node_id = self._get_node_id('call')
            
            func_name = ''
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            self.graph.add_node(
                node_id,
                type='call',
                function=func_name,
                lineno=node.lineno
            )
            
            # Add to call graph
            if self.current_function and func_name:
                self.call_graph.add_edge(self.current_function, func_name)
            
            # Add data dependencies for arguments
            for arg in node.args:
                for var in self._extract_variables(arg):
                    if var in self.var_defs:
                        self.graph.add_edge(
                            self.var_defs[var],
                            node_id,
                            type='data_dependency'
                        )
            
            self.generic_visit(node)
        
        def visit_Return(self, node: ast.Return):
            node_id = self._get_node_id('return')
            
            self.graph.add_node(
                node_id,
                type='return',
                lineno=node.lineno
            )
            
            if node.value:
                for var in self._extract_variables(node.value):
                    if var in self.var_defs:
                        self.graph.add_edge(
                            self.var_defs[var],
                            node_id,
                            type='data_dependency'
                        )
            
            for ctrl_node in self.control_deps:
                self.graph.add_edge(ctrl_node, node_id, type='control_dependency')
        
        def _extract_variables(self, node: ast.AST) -> Set[str]:
            """Extract variable names from an AST node."""
            variables = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    variables.add(child.id)
            return variables


# Convenience function for quick analysis
def analyze_code(code: str, level: AnalysisLevel = AnalysisLevel.STANDARD) -> AnalysisResult:
    """
    Convenience function to quickly analyze code.
    
    Args:
        code: Python source code
        level: Analysis level
        
    Returns:
        AnalysisResult with all analysis data
    """
    analyzer = CodeAnalyzer(level=level)
    return analyzer.analyze(code)
