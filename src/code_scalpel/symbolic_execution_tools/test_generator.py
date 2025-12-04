import ast
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any, Optional


class CoverageType(Enum):
    """Types of coverage targets."""

    STATEMENT = "statement"
    BRANCH = "branch"
    PATH = "path"
    MCDC = "mcdc"
    CONDITION = "condition"
    MUTATION = "mutation"


@dataclass
class TestCase:
    """Represents a generated test case."""

    inputs: dict[str, Any]
    expected_output: Any
    path_condition: list[Any]
    coverage_info: dict[str, set[str]]
    assertions: list[str]
    description: str
    metadata: dict[str, Any]


@dataclass
class TestSuite:
    """Represents a suite of test cases."""

    test_cases: list[TestCase]
    coverage: float
    execution_time: float
    total_paths_explored: int
    metrics: dict[str, Any]


@dataclass
class TestGenerationConfig:
    """Configuration for test generation."""

    coverage_type: CoverageType = CoverageType.BRANCH
    coverage_target: float = 0.9
    max_test_cases: int = 100
    timeout: Optional[float] = None
    minimize_suite: bool = True
    generate_assertions: bool = True
    mutation_score_target: float = 0.8


class TestGenerator:
    """Advanced test generator with multiple coverage strategies."""

    def __init__(self, engine, config: Optional[TestGenerationConfig] = None):
        self.engine = engine
        self.config = config or TestGenerationConfig()
        self.path_cache = {}
        self.coverage_info = defaultdict(set)
        self._setup_logging()

    def generate_test_suite(self, code: str) -> TestSuite:
        """
        Generate a comprehensive test suite.

        Args:
            code: Source code to test

        Returns:
            Test suite with generated test cases
        """
        start_time = time.time()

        try:
            # Initial test generation based on coverage type
            if self.config.coverage_type == CoverageType.PATH:
                test_cases = self._generate_path_coverage_tests(code)
            elif self.config.coverage_type == CoverageType.MCDC:
                test_cases = self._generate_mcdc_tests(code)
            elif self.config.coverage_type == CoverageType.MUTATION:
                test_cases = self._generate_mutation_tests(code)
            else:
                test_cases = self._generate_branch_coverage_tests(code)

            # Minimize test suite if configured
            if self.config.minimize_suite:
                test_cases = self._minimize_test_suite(test_cases)

            # Generate assertions
            if self.config.generate_assertions:
                test_cases = self._generate_assertions(test_cases)

            # Calculate metrics
            coverage = self._calculate_coverage(test_cases)
            metrics = self._calculate_metrics(test_cases)

            return TestSuite(
                test_cases=test_cases,
                coverage=coverage,
                execution_time=time.time() - start_time,
                total_paths_explored=len(self.path_cache),
                metrics=metrics,
            )

        except Exception as e:
            self.logger.error(f"Test generation error: {str(e)}")
            raise

    def _generate_path_coverage_tests(self, code: str) -> list[TestCase]:
        """Generate tests to achieve path coverage."""
        test_cases = []
        tree = ast.parse(code)

        # Get all possible paths
        paths = self._get_execution_paths(tree)

        for path in paths:
            if len(test_cases) >= self.config.max_test_cases:
                break

            # Generate test case for path
            test_case = self._generate_test_for_path(path)
            if test_case:
                test_cases.append(test_case)

            if self._coverage_target_reached(test_cases):
                break

        return test_cases

    def _generate_mcdc_tests(self, code: str) -> list[TestCase]:
        """Generate tests for MC/DC coverage."""
        test_cases = []
        tree = ast.parse(code)

        # Find all decisions and conditions
        decisions = self._find_decisions(tree)

        for decision in decisions:
            # Generate tests for each condition combination
            condition_tests = self._generate_mcdc_tests_for_decision(decision)
            test_cases.extend(condition_tests)

            if len(test_cases) >= self.config.max_test_cases:
                break

        return test_cases

    def _generate_mutation_tests(self, code: str) -> list[TestCase]:
        """Generate tests based on mutation testing."""
        test_cases = []
        tree = ast.parse(code)

        # Generate mutants
        mutants = self._generate_mutants(tree)

        for mutant in mutants:
            # Generate test to kill mutant
            test_case = self._generate_test_to_kill_mutant(mutant)
            if test_case:
                test_cases.append(test_case)

            if len(test_cases) >= self.config.max_test_cases:
                break

            if self._mutation_score_target_reached(test_cases):
                break

        return test_cases

    def _minimize_test_suite(self, test_cases: list[TestCase]) -> list[TestCase]:
        """Minimize test suite while maintaining coverage."""
        minimized = []
        covered_targets = set()

        # Sort test cases by coverage contribution
        sorted_tests = sorted(
            test_cases,
            key=lambda t: len(self._get_new_coverage(t, covered_targets)),
            reverse=True,
        )

        for test in sorted_tests:
            new_coverage = self._get_new_coverage(test, covered_targets)
            if new_coverage:
                minimized.append(test)
                covered_targets.update(new_coverage)

            if self._coverage_target_reached(minimized):
                break

        return minimized

    def _generate_assertions(self, test_cases: list[TestCase]) -> list[TestCase]:
        """Generate assertions for test cases."""
        for test_case in test_cases:
            assertions = []

            # Generate value assertions
            assertions.extend(self._generate_value_assertions(test_case))

            # Generate invariant assertions
            assertions.extend(self._generate_invariant_assertions(test_case))

            # Generate relationship assertions
            assertions.extend(self._generate_relationship_assertions(test_case))

            test_case.assertions = assertions

        return test_cases

    def _generate_value_assertions(self, test_case: TestCase) -> list[str]:
        """Generate assertions for variable values."""
        assertions = []

        for var_name, value in test_case.inputs.items():
            if isinstance(value, (int, float)):
                assertions.append(f"assert abs({var_name} - {value}) < 1e-6")
            else:
                assertions.append(f"assert {var_name} == {repr(value)}")

        return assertions

    def _generate_invariant_assertions(self, test_case: TestCase) -> list[str]:
        """Generate assertions for program invariants."""
        assertions = []

        # Check numerical invariants
        assertions.extend(self._generate_numerical_invariants(test_case))

        # Check structural invariants
        assertions.extend(self._generate_structural_invariants(test_case))

        return assertions

    def _generate_relationship_assertions(self, test_case: TestCase) -> list[str]:
        """Generate assertions for relationships between variables."""
        assertions = []
        variables = list(test_case.inputs.keys())

        # Check pairwise relationships
        for var1, var2 in combinations(variables, 2):
            relation = self._infer_relationship(test_case, var1, var2)
            if relation:
                assertions.append(relation)

        return assertions

    def _infer_relationship(
        self, test_case: TestCase, var1: str, var2: str
    ) -> Optional[str]:
        """Infer relationship between two variables."""
        val1 = test_case.inputs[var1]
        val2 = test_case.inputs[var2]

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            # Check common numerical relationships
            if abs(val1 - val2) < 1e-6:
                return f"assert abs({var1} - {var2}) < 1e-6"
            elif abs(val1 + val2) < 1e-6:
                return f"assert abs({var1} + {var2}) < 1e-6"
            elif abs(val1 * val2 - 1) < 1e-6:
                return f"assert abs({var1} * {var2} - 1) < 1e-6"

        return None

    def export_to_pytest(self, test_suite: TestSuite, output_file: str) -> None:
        """Export test suite to pytest format."""
        with open(output_file, "w") as f:
            f.write("import pytest\n\n")

            # Write test functions
            for i, test_case in enumerate(test_suite.test_cases):
                f.write(self._generate_pytest_function(test_case, i))
                f.write("\n\n")

    def _generate_pytest_function(self, test_case: TestCase, index: int) -> str:
        """Generate pytest function for a test case."""
        lines = [
            f"def test_case_{index}():",
            "    # Test case generated by SymbolicExecutor",
            f"    # {test_case.description}",
            "",
        ]

        # Add input setup
        for var_name, value in test_case.inputs.items():
            lines.append(f"    {var_name} = {repr(value)}")

        # Add assertions
        lines.extend(f"    {assertion}" for assertion in test_case.assertions)

        return "\n".join(lines)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("TestGenerator")


def create_generator(
    engine, config: Optional[TestGenerationConfig] = None
) -> TestGenerator:
    """Create a new test generator instance."""
    return TestGenerator(engine, config)
