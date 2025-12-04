from typing import Dict, List, Set, Optional, Union, Any, TypeVar, Generic, Iterator, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque
import weakref
from abc import ABC, abstractmethod
import json
from functools import total_ordering

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

@dataclass
class Position:
    """Source code position information."""
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None

class SymbolType(Enum):
    """Types of symbols in code."""
    VARIABLE = 'variable'
    FUNCTION = 'function'
    CLASS = 'class'
    MODULE = 'module'
    PARAMETER = 'parameter'
    IMPORT = 'import'

@dataclass
class Symbol:
    """Symbol table entry."""
    name: str
    type: SymbolType
    position: Position
    scope: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    references: List[Position] = field(default_factory=list)

class TreeNode(Generic[T]):
    """Enhanced tree node with traversal and search capabilities."""
    
    def __init__(self, data: T):
        self.data = data
        self.children: List[TreeNode[T]] = []
        self.parent: Optional[TreeNode[T]] = None
        self._depth: Optional[int] = None
        self._height: Optional[int] = None

    def add_child(self, child: 'TreeNode[T]'):
        """Add child node with parent reference."""
        child.parent = self
        self.children.append(child)
        self._invalidate_cache()

    def remove_child(self, child: 'TreeNode[T]'):
        """Remove child node."""
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            self._invalidate_cache()

    def traverse_preorder(self) -> Iterator[T]:
        """Pre-order traversal."""
        yield self.data
        for child in self.children:
            yield from child.traverse_preorder()

    def traverse_postorder(self) -> Iterator[T]:
        """Post-order traversal."""
        for child in self.children:
            yield from child.traverse_postorder()
        yield self.data

    def traverse_levelorder(self) -> Iterator[T]:
        """Level-order traversal."""
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node.data
            queue.extend(node.children)

    def find(self, predicate: Callable[[T], bool]) -> Optional['TreeNode[T]']:
        """Find first node matching predicate."""
        if predicate(self.data):
            return self
        for child in self.children:
            if result := child.find(predicate):
                return result
        return None

    def find_all(self, predicate: Callable[[T], bool]) -> Iterator['TreeNode[T]']:
        """Find all nodes matching predicate."""
        if predicate(self.data):
            yield self
        for child in self.children:
            yield from child.find_all(predicate)

    @property
    def depth(self) -> int:
        """Get node depth (distance from root)."""
        if self._depth is None:
            self._depth = 0 if self.parent is None else self.parent.depth + 1
        return self._depth

    @property
    def height(self) -> int:
        """Get node height (length of longest path to leaf)."""
        if self._height is None:
            self._height = max((child.height for child in self.children), default=-1) + 1
        return self._height

    def _invalidate_cache(self):
        """Invalidate cached properties."""
        self._depth = None
        self._height = None
        if self.parent:
            self.parent._invalidate_cache()

class Graph(Generic[T]):
    """Enhanced graph with advanced algorithms."""
    
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.nodes: Dict[T, Set[T]] = defaultdict(set)
        self.node_data: Dict[T, Any] = {}
        self.edge_data: Dict[Tuple[T, T], Any] = {}

    def add_node(self, node: T, data: Any = None):
        """Add node with optional data."""
        if node not in self.nodes:
            self.nodes[node] = set()
        if data is not None:
            self.node_data[node] = data

    def add_edge(self, source: T, target: T, data: Any = None):
        """Add edge with optional data."""
        self.add_node(source)
        self.add_node(target)
        self.nodes[source].add(target)
        if not self.directed:
            self.nodes[target].add(source)
        if data is not None:
            self.edge_data[(source, target)] = data
            if not self.directed:
                self.edge_data[(target, source)] = data

    def remove_node(self, node: T):
        """Remove node and all connected edges."""
        if node in self.nodes:
            # Remove edges to this node
            for other in self.nodes:
                self.nodes[other].discard(node)
                if (other, node) in self.edge_data:
                    del self.edge_data[(other, node)]
                if (node, other) in self.edge_data:
                    del self.edge_data[(node, other)]
            # Remove the node
            del self.nodes[node]
            if node in self.node_data:
                del self.node_data[node]

    def get_neighbors(self, node: T) -> Set[T]:
        """Get neighboring nodes."""
        return self.nodes.get(node, set())

    def bfs(self, start: T) -> Iterator[T]:
        """Breadth-first search traversal."""
        visited = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            yield node
            for neighbor in self.nodes[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    def dfs(self, start: T) -> Iterator[T]:
        """Depth-first search traversal."""
        visited = set()
        def _dfs(node: T):
            visited.add(node)
            yield node
            for neighbor in self.nodes[node]:
                if neighbor not in visited:
                    yield from _dfs(neighbor)
        yield from _dfs(start)

    def shortest_path(self, start: T, end: T) -> Optional[List[T]]:
        """Find shortest path between nodes."""
        if start not in self.nodes or end not in self.nodes:
            return None
            
        visited = {start}
        queue = deque([(start, [start])])
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in self.nodes[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def topological_sort(self) -> Optional[List[T]]:
        """Topological sort (for directed acyclic graphs)."""
        if not self.directed:
            return None
            
        in_degree = defaultdict(int)
        for node in self.nodes:
            for neighbor in self.nodes[node]:
                in_degree[neighbor] += 1
                
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self.nodes[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(result) != len(self.nodes):
            return None  # Graph has cycles
        return result

class SymbolTable:
    """Symbol table for code analysis."""
    
    def __init__(self):
        self.scopes: Dict[str, Dict[str, Symbol]] = defaultdict(dict)
        self.current_scope: List[str] = ['global']

    def enter_scope(self, name: str):
        """Enter a new scope."""
        self.current_scope.append(name)

    def exit_scope(self):
        """Exit current scope."""
        if len(self.current_scope) > 1:
            self.current_scope.pop()

    def add_symbol(self, symbol: Symbol):
        """Add symbol to current scope."""
        scope = '.'.join(self.current_scope)
        symbol.scope = scope
        self.scopes[scope][symbol.name] = symbol

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up symbol in current and parent scopes."""
        for i in range(len(self.current_scope), 0, -1):
            scope = '.'.join(self.current_scope[:i])
            if name in self.scopes[scope]:
                return self.scopes[scope][name]
        return None

    def get_symbols_in_scope(self, scope: str) -> Dict[str, Symbol]:
        """Get all symbols in a scope."""
        return dict(self.scopes[scope])

class CallGraph(Graph[str]):
    """Function call graph."""
    
    def __init__(self):
        super().__init__(directed=True)
        self.call_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    def add_call(self, caller: str, callee: str, position: Position):
        """Add function call."""
        self.add_edge(caller, callee, {'positions': [position]})
        self.call_counts[(caller, callee)] += 1

    def get_callers(self, function: str) -> Set[str]:
        """Get functions that call the given function."""
        return {node for node in self.nodes if function in self.nodes[node]}

    def get_callees(self, function: str) -> Set[str]:
        """Get functions called by the given function."""
        return self.get_neighbors(function)

    def get_call_count(self, caller: str, callee: str) -> int:
        """Get number of calls between functions."""
        return self.call_counts[(caller, callee)]

class WeightedGraph(Graph[T]):
    """Graph with weighted edges."""
    
    def add_edge(self, source: T, target: T, weight: float):
        """Add weighted edge."""
        super().add_edge(source, target, {'weight': weight})

    def dijkstra(self, start: T) -> Dict[T, float]:
        """Dijkstra's shortest path algorithm."""
        distances = {node: float('inf') for node in self.nodes}
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor in self.nodes[node]:
                weight = self.edge_data[(node, neighbor)]['weight']
                if distances[node] + weight < distances[neighbor]:
                    distances[neighbor] = distances[node] + weight
                    heapq.heappush(pq, (distances[neighbor], neighbor))
                    
        return distances

def create_tree(data: T) -> TreeNode[T]:
    """Create a new tree node."""
    return TreeNode(data)

def create_graph(directed: bool = False) -> Graph[T]:
    """Create a new graph."""
    return Graph(directed)

def create_symbol_table() -> SymbolTable:
    """Create a new symbol table."""
    return SymbolTable()

def create_call_graph() -> CallGraph:
    """Create a new call graph."""
    return CallGraph()