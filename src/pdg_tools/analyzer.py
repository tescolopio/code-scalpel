import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import ast
from collections import defaultdict

class DependencyType(Enum):
    """Types of dependencies in the PDG."""
    DATA = 'data_dependency'
    CONTROL = 'control_dependency'
    CALL = 'call_dependency'
    PARAMETER = 'parameter_dependency'
    RETURN = 'return_dependency'

@dataclass
class DataFlowAnomaly:
    """Represents a data flow anomaly in the code."""
    type: str  # 'undefined', 'unused', 'overwrite'
    variable: str
    location: Tuple[int, int]  # line, column
    severity: str
    message: str

@dataclass
class SecurityVulnerability:
    """Represents a security vulnerability in the code."""
    type: str
    source: str
    sink: str
    path: List[str]
    severity: str
    description: str

class PDGAnalyzer:
    """Advanced Program Dependence Graph Analyzer."""

    def __init__(self, pdg: nx.DiGraph):
        self.pdg = pdg
        self.cached_results = {}

    def analyze_data_flow(self) -> Dict[str, Any]:
        """Perform comprehensive data flow analysis."""
        if 'data_flow' in self.cached_results:
            return self.cached_results['data_flow']

        results = {
            'anomalies': self._find_data_flow_anomalies(),
            'def_use_chains': self._build_def_use_chains(),
            'live_variables': self._analyze_live_variables(),
            'reaching_definitions': self._analyze_reaching_definitions(),
            'value_ranges': self._analyze_value_ranges()
        }

        self.cached_results['data_flow'] = results
        return results

    def analyze_control_flow(self) -> Dict[str, Any]:
        """Perform comprehensive control flow analysis."""
        return {
            'cyclomatic_complexity': self._calculate_cyclomatic_complexity(),
            'control_dependencies': self._analyze_control_dependencies(),
            'unreachable_code': self._find_unreachable_code(),
            'loop_info': self._analyze_loops(),
            'dominators': self._compute_dominators()
        }

    def perform_security_analysis(self) -> List[SecurityVulnerability]:
        """Perform security analysis on the PDG."""
        vulnerabilities = []

        # Perform taint analysis
        taint_results = self._perform_taint_analysis()
        vulnerabilities.extend(taint_results)

        # Analyze information flow
        info_flow_results = self._analyze_information_flow()
        vulnerabilities.extend(info_flow_results)

        # Check for common vulnerabilities
        vulnerabilities.extend(self._check_common_vulnerabilities())

        return vulnerabilities

    def find_optimization_opportunities(self) -> Dict[str, List[Dict]]:
        """Identify potential code optimization opportunities."""
        return {
            'loop_invariant': self._find_loop_invariant_code(),
            'common_subexpressions': self._find_common_subexpressions(),
            'dead_code': self._find_dead_code(),
            'redundant_computations': self._find_redundant_computations()
        }

    def compute_program_slice(self, criterion: str, 
                            direction: str = 'backward') -> nx.DiGraph:
        """Compute a program slice based on a slicing criterion."""
        visited = set()
        slice_graph = nx.DiGraph()

        def traverse_dependencies(node: str):
            if node in visited:
                return
            visited.add(node)
            slice_graph.add_node(node, **self.pdg.nodes[node])

            if direction == 'backward':
                edges = self.pdg.in_edges(node, data=True)
            else:
                edges = self.pdg.out_edges(node, data=True)

            for src, dst, data in edges:
                next_node = src if direction == 'backward' else dst
                if data['type'] in [DependencyType.DATA.value, 
                                  DependencyType.CONTROL.value]:
                    slice_graph.add_edge(src, dst, **data)
                    traverse_dependencies(next_node)

        traverse_dependencies(criterion)
        return slice_graph

    def _find_data_flow_anomalies(self) -> List[DataFlowAnomaly]:
        """Find data flow anomalies in the code."""
        anomalies = []
        definitions = defaultdict(list)
        uses = defaultdict(list)

        # Collect definitions and uses
        for node, data in self.pdg.nodes(data=True):
            if 'defines' in data:
                for var in data['defines']:
                    definitions[var].append(node)
            if 'uses' in data:
                for var in data['uses']:
                    uses[var].append(node)

        # Check for undefined variables
        for var, use_nodes in uses.items():
            if var not in definitions:
                for node in use_nodes:
                    anomalies.append(
                        DataFlowAnomaly(
                            type='undefined',
                            variable=var,
                            location=self.pdg.nodes[node].get('location', (0, 0)),
                            severity='error',
                            message=f"Variable '{var}' used before definition"
                        )
                    )

        # Check for unused variables
        for var, def_nodes in definitions.items():
            if var not in uses:
                for node in def_nodes:
                    anomalies.append(
                        DataFlowAnomaly(
                            type='unused',
                            variable=var,
                            location=self.pdg.nodes[node].get('location', (0, 0)),
                            severity='warning',
                            message=f"Variable '{var}' defined but never used"
                        )
                    )

        return anomalies

    def _perform_taint_analysis(self) -> List[SecurityVulnerability]:
        """Perform taint analysis to identify security vulnerabilities."""
        vulnerabilities = []
        sources = self._identify_taint_sources()
        sinks = self._identify_taint_sinks()

        for source in sources:
            for sink in sinks:
                paths = list(nx.all_simple_paths(self.pdg, source, sink))
                if paths:
                    # Check if taint is sanitized along the path
                    for path in paths:
                        if not self._is_path_sanitized(path):
                            source_type = self.pdg.nodes[source].get('taint_type')
                            sink_type = self.pdg.nodes[sink].get('sink_type')
                            vulnerabilities.append(
                                SecurityVulnerability(
                                    type=f"{source_type}_to_{sink_type}",
                                    source=source,
                                    sink=sink,
                                    path=path,
                                    severity='high',
                                    description=self._generate_vulnerability_description(
                                        source_type, sink_type
                                    )
                                )
                            )

        return vulnerabilities

    def _find_loop_invariant_code(self) -> List[Dict]:
        """Identify loop-invariant code."""
        loop_invariants = []
        loops = self._find_loops()

        for loop in loops:
            invariant_candidates = self._analyze_loop_body(loop)
            for candidate in invariant_candidates:
                if self._is_loop_invariant(candidate, loop):
                    loop_invariants.append({
                        'node': candidate,
                        'loop': loop,
                        'savings': self._estimate_optimization_savings(candidate, loop)
                    })

        return loop_invariants

    def _analyze_value_ranges(self) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
        """Analyze possible value ranges for variables."""
        ranges = {}
        for node, data in self.pdg.nodes(data=True):
            if 'defines' in data:
                for var in data['defines']:
                    ranges[var] = self._compute_value_range(var, node)
        return ranges

    def _compute_value_range(self, var: str, node: str) -> Tuple[Optional[int], Optional[int]]:
        """Compute possible value range for a variable at a given node."""
        constraints = []
        
        # Collect constraints from control dependencies
        for pred, _, data in self.pdg.in_edges(node, data=True):
            if data['type'] == DependencyType.CONTROL.value:
                if 'condition' in self.pdg.nodes[pred]:
                    constraint = self._parse_condition_constraint(
                        self.pdg.nodes[pred]['condition'], var
                    )
                    if constraint:
                        constraints.append(constraint)

        return self._solve_constraints(constraints)

    def _identify_taint_sources(self) -> Set[str]:
        """Identify nodes that can introduce tainted data."""
        sources = set()
        dangerous_functions = {
            'input', 'request.get', 'request.post', 'request.form',
            'file.read', 'subprocess.check_output'
        }

        for node, data in self.pdg.nodes(data=True):
            if ('type' in data and data['type'] == 'call' and
                any(func in str(data.get('call_target', '')) 
                    for func in dangerous_functions)):
                sources.add(node)
                self.pdg.nodes[node]['taint_type'] = 'user_input'

        return sources

    def _identify_taint_sinks(self) -> Set[str]:
        """Identify nodes that are sensitive to tainted data."""
        sinks = set()
        sensitive_functions = {
            'execute', 'eval', 'subprocess.run', 'render_template',
            'Response', 'send_file'
        }

        for node, data in self.pdg.nodes(data=True):
            if ('type' in data and data['type'] == 'call' and
                any(func in str(data.get('call_target', '')) 
                    for func in sensitive_functions)):
                sinks.add(node)
                sink_type = self._determine_sink_type(data.get('call_target', ''))
                self.pdg.nodes[node]['sink_type'] = sink_type

        return sinks

    def _determine_sink_type(self, call_target: str) -> str:
        """Determine the type of sink based on the call target."""
        if 'execute' in call_target:
            return 'sql_injection'
        elif 'eval' in call_target:
            return 'code_injection'
        elif 'subprocess' in call_target:
            return 'command_injection'
        elif 'render_template' in call_target:
            return 'xss'
        return 'unknown'

    def _is_path_sanitized(self, path: List[str]) -> bool:
        """Check if a path contains proper sanitization."""
        sanitizer_functions = {
            'escape', 'sanitize', 'bleach.clean', 'html.escape',
            'parameterize', 'quote'
        }

        for node in path:
            data = self.pdg.nodes[node]
            if ('type' in data and data['type'] == 'call' and
                any(func in str(data.get('call_target', '')) 
                    for func in sanitizer_functions)):
                return True
        return False

    def _find_common_subexpressions(self) -> List[Dict]:
        """Find common subexpressions that could be optimized."""
        expressions = defaultdict(list)
        
        for node, data in self.pdg.nodes(data=True):
            if 'expression' in data:
                expr_hash = self._hash_expression(data['expression'])
                expressions[expr_hash].append(node)

        return [
            {
                'expression': self.pdg.nodes[nodes[0]]['expression'],
                'locations': [self.pdg.nodes[n].get('location') for n in nodes],
                'frequency': len(nodes)
            }
            for expr_hash, nodes in expressions.items()
            if len(nodes) > 1
        ]

    def _hash_expression(self, expression: ast.AST) -> int:
        """Create a hash for an expression for comparison purposes."""
        if isinstance(expression, ast.AST):
            return hash(ast.dump(expression))
        return hash(str(expression))

    def _find_redundant_computations(self) -> List[Dict]:
        """Identify redundant computations that could be optimized."""
        redundant = []
        
        def get_computation_result(node: str) -> Optional[Any]:
            """Try to determine the result of a computation statically."""
            data = self.pdg.nodes[node]
            if 'constant_value' in data:
                return data['constant_value']
            return None

        for node, data in self.pdg.nodes(data=True):
            if 'type' in data and data['type'] == 'computation':
                result = get_computation_result(node)
                if result is not None:
                    # Find other nodes computing the same result
                    for other_node, other_data in self.pdg.nodes(data=True):
                        if (other_node != node and 
                            'type' in other_data and 
                            other_data['type'] == 'computation'):
                            other_result = get_computation_result(other_node)
                            if result == other_result:
                                redundant.append({
                                    'nodes': [node, other_node],
                                    'result': result,
                                    'optimization': 'constant_folding'
                                })

        return redundant

    @staticmethod
    def _generate_vulnerability_description(source_type: str, sink_type: str) -> str:
        """Generate a detailed description of a vulnerability."""
        return (f"Potentially unsafe flow from {source_type} source to {sink_type} "
                f"sink. This could lead to a {sink_type} vulnerability if the input "
                "is not properly sanitized.")