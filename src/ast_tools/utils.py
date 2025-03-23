import ast
from typing import Dict, List, Set, Optional, Any, Callable, Union
from collections import defaultdict
import tokenize
from io import StringIO

class ASTUtils:
    """Utility functions for working with ASTs."""
    
    @staticmethod
    def get_all_names(tree: ast.AST) -> Set[str]:
        """Get all names used in the AST."""
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names

    @staticmethod
    def get_function_info(node: ast.FunctionDef) -> Dict[str, Any]:
        """Get detailed information about a function."""
        return {
            'name': node.name,
            'args': [arg.arg for arg in node.args.args],
            'defaults': [ast.unparse(d) for d in node.args.defaults],
            'kwonlyargs': [arg.arg for arg in node.args.kwonlyargs],
            'vararg': node.args.vararg.arg if node.args.vararg else None,
            'kwarg': node.args.kwarg.arg if node.args.kwarg else None,
            'decorators': [ast.unparse(d) for d in node.decorator_list],
            'returns': ast.unparse(node.returns) if node.returns else None,
            'docstring': ast.get_docstring(node),
            'line_number': node.lineno,
            'end_line_number': node.end_lineno if hasattr(node, 'end_lineno') else None
        }

    @staticmethod
    def find_all(tree: ast.AST,
                 condition: Callable[[ast.AST], bool]) -> List[ast.AST]:
        """Find all nodes matching a condition."""
        return [node for node in ast.walk(tree) if condition(node)]

    @staticmethod
    def get_node_source(node: ast.AST, source_lines: List[str]) -> str:
        """Get the source code for a node."""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            return '\n'.join(source_lines[node.lineno-1:node.end_lineno])
        return ast.unparse(node)

    @staticmethod
    def analyze_dependencies(tree: ast.AST) -> Dict[str, Set[str]]:
        """Analyze variable dependencies in the code."""
        deps = defaultdict(set)
        
        class DependencyVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                # Get variables being assigned to
                targets = set()
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        targets.add(target.id)
                
                # Get variables used in the assignment
                used = set()
                for subnode in ast.walk(node.value):
                    if isinstance(subnode, ast.Name):
                        used.add(subnode.id)
                
                # Record dependencies
                for target in targets:
                    deps[target].update(used)
                
                self.generic_visit(node)
        
        DependencyVisitor().visit(tree)
        return dict(deps)

    @staticmethod
    def format_code(tree: ast.AST) -> str:
        """Format AST as properly indented code."""
        return ast.unparse(tree)

    @classmethod
    def find_similar_nodes(cls, 
                          tree: ast.AST,
                          pattern: Union[str, ast.AST],
                          threshold: float = 0.8) -> List[ast.AST]:
        """Find nodes similar to a pattern."""
        if isinstance(pattern, str):
            pattern = ast.parse(pattern).body[0]
            
        similar_nodes = []
        for node in ast.walk(tree):
            if cls.calculate_similarity(node, pattern) >= threshold:
                similar_nodes.append(node)
        return similar_nodes

    @staticmethod
    def calculate_similarity(node1: ast.AST, node2: ast.AST) -> float:
        """Calculate similarity between two AST nodes."""
        # Placeholder for actual similarity calculation logic
        return 1.0 if type(node1) == type(node2) else 0.0

    @staticmethod
    def remove_comments(code: str) -> str:
        """Remove comments while preserving line numbers."""
        result = []
        prev_toktype = tokenize.INDENT
        first_line = True
        
        tokens = tokenize.generate_tokens(StringIO(code).readline)
        
        for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokens:
            if toktype == tokenize.COMMENT:
                continue
            elif toktype == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    result.append(" ")
                result.append(ttext)
            elif toktype == tokenize.NEWLINE:
                result.append(ttext)
            elif toktype == tokenize.INDENT:
                result.append(ttext)
            elif toktype == tokenize.DEDENT:
                pass
            else:
                if not first_line and prev_toktype != tokenize.INDENT:
                    result.append(" ")
                result.append(ttext)
            prev_toktype = toktype
            first_line = False
            
        return "".join(result)

    @staticmethod
    def traverse_ast(tree: ast.AST, strategy: str = 'depth-first') -> List[ast.AST]:
        """Traverse the AST using the specified strategy."""
        nodes = []
        if strategy == 'depth-first':
            nodes = list(ast.walk(tree))
        elif strategy == 'breadth-first':
            queue = [tree]
            while queue:
                node = queue.pop(0)
                nodes.append(node)
                queue.extend(ast.iter_child_nodes(node))
        return nodes

    @staticmethod
    def compare_nodes(node1: ast.AST, node2: ast.AST) -> bool:
        """Compare two AST nodes for structural equality."""
        if type(node1) != type(node2):
            return False
        for field in node1._fields:
            val1 = getattr(node1, field)
            val2 = getattr(node2, field)
            if isinstance(val1, list):
                if not isinstance(val2, list) or len(val1) != len(val2):
                    return False
                for v1, v2 in zip(val1, val2):
                    if not ASTUtils.compare_nodes(v1, v2):
                        return False
            elif isinstance(val1, ast.AST):
                if not ASTUtils.compare_nodes(val1, val2):
                    return False
            elif val1 != val2:
                return False
        return True

    @staticmethod
    def generate_docstring(node: ast.FunctionDef) -> str:
        """Generate a docstring for a function from its AST node."""
        signature = ASTUtils.extract_function_signature(node)
        docstring = f'"""{signature}\n\n'
        docstring += "Args:\n"
        for arg in node.args.args:
            docstring += f"  {arg.arg}: \n"
        if node.returns:
            docstring += f"\nReturns:\n  {ast.unparse(node.returns)}\n"
        docstring += '"""'
        return docstring

    @staticmethod
    def extract_function_signature(node: ast.FunctionDef) -> str:
        """Extract the signature of a function from its AST node."""
        args = [arg.arg for arg in node.args.args]
        defaults = [ast.unparse(d) for d in node.args.defaults]
        args_with_defaults = args[:len(args) - len(defaults)] + [f"{a}={d}" for a, d in zip(args[len(args) - len(defaults):], defaults)]
        return f"def {node.name}({', '.join(args_with_defaults)})"