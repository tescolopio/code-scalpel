import ast
import heapq
import logging
import time
from collections import defaultdict, deque
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import z3


class SearchStrategy(Enum):
    """Available path exploration strategies."""

    DFS = "depth_first"
    BFS = "breadth_first"
    RANDOM = "random"
    HEURISTIC = "heuristic"
    COVERAGE_GUIDED = "coverage_guided"
    CONCOLIC = "concolic"
    PARALLEL = "parallel"


@dataclass
class PathInfo:
    """Information about an execution path."""

    constraints: list[Any]  # Path conditions
    symbolic_state: dict[str, Any]  # Variable values
    executed_nodes: set[str]  # Executed AST nodes
    branch_history: list[tuple[str, bool]]  # Branch decisions
    depth: int
    score: float = 0.0  # For heuristic-based exploration


@dataclass
class ExplorationConfig:
    """Configuration for path exploration."""

    strategy: SearchStrategy = SearchStrategy.DFS
    max_depth: int = 100
    max_paths: Optional[int] = None
    timeout: Optional[float] = None
    coverage_target: float = 0.8
    use_caching: bool = True
    parallel_workers: int = 1


@dataclass
class ExplorationResult:
    """Results of path exploration."""

    paths: list[PathInfo]
    coverage: float
    execution_time: float
    num_paths_explored: int
    num_paths_pruned: int
    solver_stats: dict[str, Any]


class PathExplorer:
    """Advanced path explorer with multiple search strategies."""

    def __init__(self, engine, config: Optional[ExplorationConfig] = None):
        self.engine = engine
        self.config = config or ExplorationConfig()
        self.path_cache = {}
        self.coverage_info = defaultdict(set)
        self.exploration_history = []
        self._setup_logging()

    def explore(self, code: str) -> ExplorationResult:
        """
        Explore execution paths using the configured strategy.

        Args:
            code: Source code to explore

        Returns:
            Exploration results including paths and coverage
        """
        start_time = time.time()
        tree = ast.parse(code)

        try:
            if self.config.strategy == SearchStrategy.DFS:
                paths = self._explore_dfs(tree)
            elif self.config.strategy == SearchStrategy.BFS:
                paths = self._explore_bfs(tree)
            elif self.config.strategy == SearchStrategy.RANDOM:
                paths = self._explore_random(tree)
            elif self.config.strategy == SearchStrategy.HEURISTIC:
                paths = self._explore_heuristic(tree)
            elif self.config.strategy == SearchStrategy.COVERAGE_GUIDED:
                paths = self._explore_coverage_guided(tree)
            elif self.config.strategy == SearchStrategy.CONCOLIC:
                paths = self._explore_concolic(tree)
            else:
                paths = self._explore_parallel(tree)

            execution_time = time.time() - start_time
            coverage = self._calculate_coverage()

            return ExplorationResult(
                paths=list(paths),
                coverage=coverage,
                execution_time=execution_time,
                num_paths_explored=len(self.exploration_history),
                num_paths_pruned=self._count_pruned_paths(),
                solver_stats=self.engine.solver.get_statistics(),
            )

        except Exception as e:
            self.logger.error(f"Exploration error: {str(e)}")
            raise

    def _explore_dfs(self, tree: ast.AST) -> Generator[PathInfo, None, None]:
        """Depth-first exploration of execution paths."""
        stack = [(tree, PathInfo([], {}, set(), [], 0))]

        while stack and self._should_continue():
            node, path_info = stack.pop()

            if self._should_prune(path_info):
                continue

            # Process current node
            path_info.executed_nodes.add(self._get_node_id(node))

            if isinstance(node, ast.If):
                # Handle branches
                for branch, condition in self._get_branches(node):
                    new_path = self._extend_path(path_info, condition)
                    if self._is_feasible(new_path):
                        stack.append((branch, new_path))
            else:
                # Process other node types
                for child in ast.iter_child_nodes(node):
                    stack.append((child, path_info))

            if self._is_complete_path(path_info):
                yield path_info

    def _explore_bfs(self, tree: ast.AST) -> Generator[PathInfo, None, None]:
        """Breadth-first exploration of execution paths."""
        queue = deque([(tree, PathInfo([], {}, set(), [], 0))])

        while queue and self._should_continue():
            node, path_info = queue.popleft()

            if self._should_prune(path_info):
                continue

            path_info.executed_nodes.add(self._get_node_id(node))

            if isinstance(node, ast.If):
                for branch, condition in self._get_branches(node):
                    new_path = self._extend_path(path_info, condition)
                    if self._is_feasible(new_path):
                        queue.append((branch, new_path))
            else:
                for child in ast.iter_child_nodes(node):
                    queue.append((child, path_info))

            if self._is_complete_path(path_info):
                yield path_info

    def _explore_heuristic(self, tree: ast.AST) -> Generator[PathInfo, None, None]:
        """Heuristic-guided exploration of execution paths."""
        # Priority queue ordered by path scores
        paths = [(0, 0, tree, PathInfo([], {}, set(), [], 0))]
        heapq.heapify(paths)

        while paths and self._should_continue():
            _, _, node, path_info = heapq.heappop(paths)

            if self._should_prune(path_info):
                continue

            path_info.executed_nodes.add(self._get_node_id(node))

            if isinstance(node, ast.If):
                for branch, condition in self._get_branches(node):
                    new_path = self._extend_path(path_info, condition)
                    if self._is_feasible(new_path):
                        score = self._calculate_path_score(new_path)
                        heapq.heappush(paths, (-score, len(paths), branch, new_path))
            else:
                for child in ast.iter_child_nodes(node):
                    heapq.heappush(paths, (0, len(paths), child, path_info))

            if self._is_complete_path(path_info):
                yield path_info

    def _explore_coverage_guided(
        self, tree: ast.AST
    ) -> Generator[PathInfo, None, None]:
        """Coverage-guided exploration of execution paths."""
        paths = [(tree, PathInfo([], {}, set(), [], 0))]
        covered_nodes = set()

        while paths and self._should_continue():
            node, path_info = self._select_path_for_coverage(paths, covered_nodes)
            paths.remove((node, path_info))

            if self._should_prune(path_info):
                continue

            node_id = self._get_node_id(node)
            path_info.executed_nodes.add(node_id)
            covered_nodes.add(node_id)

            if isinstance(node, ast.If):
                for branch, condition in self._get_branches(node):
                    new_path = self._extend_path(path_info, condition)
                    if self._is_feasible(new_path):
                        paths.append((branch, new_path))
            else:
                for child in ast.iter_child_nodes(node):
                    paths.append((child, path_info))

            if self._is_complete_path(path_info):
                yield path_info

            if self._coverage_target_reached(covered_nodes):
                break

    def _explore_concolic(self, tree: ast.AST) -> Generator[PathInfo, None, None]:
        """Concolic (concrete + symbolic) exploration."""
        # Start with concrete execution
        initial_inputs = self._generate_initial_inputs()
        paths = self._execute_concrete(tree, initial_inputs)

        while paths and self._should_continue():
            path = paths.pop()

            # Generate new inputs by negating path conditions
            for i, condition in enumerate(path.constraints):
                new_inputs = self._solve_negated_constraint(
                    path.constraints[:i] + [z3.Not(condition)]
                )

                if new_inputs:
                    new_paths = self._execute_concrete(tree, new_inputs)
                    paths.extend(new_paths)

            yield path

    def _is_feasible(self, path_info: PathInfo) -> bool:
        """Check if a path is feasible."""
        if self.config.use_caching:
            cache_key = self._make_cache_key(path_info)
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]

        self.engine.solver.push()
        for constraint in path_info.constraints:
            self.engine.solver.add_constraint(constraint)

        feasible = self.engine.solver.check_sat()
        self.engine.solver.pop()

        if self.config.use_caching:
            self.path_cache[cache_key] = feasible

        return feasible

    def _calculate_path_score(self, path_info: PathInfo) -> float:
        """Calculate heuristic score for a path."""
        # Combine multiple factors
        coverage_score = self._calculate_coverage_score(path_info)
        complexity_score = self._calculate_complexity_score(path_info)
        novelty_score = self._calculate_novelty_score(path_info)

        # Weighted combination
        return 0.4 * coverage_score + 0.3 * complexity_score + 0.3 * novelty_score

    def _calculate_coverage_score(self, path_info: PathInfo) -> float:
        """Calculate coverage contribution of a path."""
        new_nodes = path_info.executed_nodes - self._get_covered_nodes()
        return len(new_nodes) / self._get_total_nodes()

    def _calculate_complexity_score(self, path_info: PathInfo) -> float:
        """Calculate complexity score based on path conditions."""
        if not path_info.constraints:
            return 0.0

        total_complexity = sum(
            self._get_constraint_complexity(c) for c in path_info.constraints
        )
        return total_complexity / len(path_info.constraints)

    def _calculate_novelty_score(self, path_info: PathInfo) -> float:
        """Calculate novelty score based on previously explored paths."""
        if not self.exploration_history:
            return 1.0

        max_similarity = max(
            self._calculate_path_similarity(path_info, prev_path)
            for prev_path in self.exploration_history
        )
        return 1.0 - max_similarity

    def _calculate_path_similarity(self, path1: PathInfo, path2: PathInfo) -> float:
        """Calculate similarity between two paths."""
        common_nodes = len(path1.executed_nodes & path2.executed_nodes)
        total_nodes = len(path1.executed_nodes | path2.executed_nodes)
        return common_nodes / total_nodes if total_nodes > 0 else 0.0

    def _get_constraint_complexity(self, constraint: Any) -> int:
        """Calculate complexity of a constraint."""

        def count_operations(expr):
            if z3.is_const(expr):
                return 0
            return 1 + sum(count_operations(arg) for arg in expr.children())

        return count_operations(constraint)

    def _extend_path(self, path_info: PathInfo, condition: Any) -> PathInfo:
        """Create new path info by extending with a condition."""
        return PathInfo(
            constraints=path_info.constraints + [condition],
            symbolic_state=path_info.symbolic_state.copy(),
            executed_nodes=path_info.executed_nodes.copy(),
            branch_history=path_info.branch_history[:],
            depth=path_info.depth + 1,
        )

    def _should_continue(self) -> bool:
        """Check if exploration should continue."""
        if self.config.timeout and time.time() > self._start_time + self.config.timeout:
            return False

        return not (
            self.config.max_paths
            and len(self.exploration_history) >= self.config.max_paths
        )

    def _should_prune(self, path_info: PathInfo) -> bool:
        """Check if a path should be pruned."""
        if path_info.depth > self.config.max_depth:
            return True

        return bool(not self._is_feasible(path_info))

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("PathExplorer")


def create_explorer(engine, config: Optional[ExplorationConfig] = None) -> PathExplorer:
    """Create a new path explorer instance."""
    return PathExplorer(engine, config)
