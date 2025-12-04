import ast
import astor
from typing import Dict, List, Optional, Any

from .ast_tools.builder import build_ast  # Assuming you moved build_ast here
from .pdg_tools.builder import PDGBuilder  # Assuming you moved PDGBuilder here
from .symbolic_execution_tools.engine import SymbolicExecutor  # Assuming you moved SymbolicExecutor here
from .utilities.visualization import visualize_pdg  # Assuming you moved visualize_pdg here

class CodeAnalysisToolkit:
    def __init__(self):
        self.ast_cache = {}
        self.pdg_cache = {}
        self.symbolic_state = {}
        
    def parse_to_ast(self, code: str) -> ast.AST:
        """Parse Python code into an AST."""
        if code not in self.ast_cache:
            self.ast_cache[code] = ast.parse(code)
        return self.ast_cache[code]
    
    def ast_to_code(self, node: ast.AST) -> str:
        """Convert AST back to source code."""
        return astor.to_source(node)

    class PDGBuilder(ast.NodeVisitor):
        def __init__(self):
            self.graph = nx.DiGraph()
            self.current_scope = []
            self.var_defs = {}
            self.control_deps = []
            
        def visit_Assign(self, node):
            target_id = astor.to_source(node.targets[0]).strip()
            value_code = astor.to_source(node.value).strip()
            node_id = f"assign_{target_id}"
            
            self.graph.add_node(node_id, type='assign', 
                              target=target_id, value=value_code)
            
            # Add data dependencies
            for var in self._extract_variables(node.value):
                if var in self.var_defs:
                    self.graph.add_edge(self.var_defs[var], node_id, 
                                      type='data_dependency')
            
            self.var_defs[target_id] = node_id
            self.generic_visit(node)

        def visit_If(self, node):
            cond_code = astor.to_source(node.test).strip()
            node_id = f"if_{cond_code}"
            
            self.graph.add_node(node_id, type='if', condition=cond_code)
            self.control_deps.append(node_id)
            
            # Process body with control dependency
            for stmt in node.body:
                self.visit(stmt)
                stmt_id = list(self.graph.nodes)[-1]
                self.graph.add_edge(node_id, stmt_id, type='control_dependency')
            
            self.control_deps.pop()
            self.generic_visit(node)
        
        def _extract_variables(self, node):
            variables = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    variables.add(child.id)
            return variables

    def build_pdg(self, code: str) -> nx.DiGraph:
        """Build a Program Dependence Graph from code."""
        if code not in self.pdg_cache:
            tree = self.parse_to_ast(code)
            builder = self.PDGBuilder()
            builder.visit(tree)
            self.pdg_cache[code] = builder.graph
        return self.pdg_cache[code]
    
    def visualize_pdg(self, graph: nx.DiGraph, output_file: str = "pdg.png"):
        """Visualize the PDG using graphviz."""
        dot = Digraph(comment='Program Dependence Graph')
        
        for node in graph.nodes:
            attrs = graph.nodes[node]
            label = f"{node}\n{attrs.get('type', '')}"
            if 'value' in attrs:
                label += f"\nvalue: {attrs['value']}"
            dot.node(str(node), label)
        
        for edge in graph.edges:
            edge_type = graph.edges[edge]['type']
            color = 'blue' if edge_type == 'data_dependency' else 'red'
            dot.edge(str(edge[0]), str(edge[1]), color=color)
        
        dot.render(output_file, view=True)