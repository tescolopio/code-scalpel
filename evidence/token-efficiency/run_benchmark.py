#!/usr/bin/env python3
"""
Code Scalpel Token Efficiency Benchmark
=======================================

Measures the token reduction achieved by surgical code extraction
compared to naive full-file approaches.

Demonstrates the claim: "Feed the LLM 50 lines, not 5,000 lines"

Usage:
    python run_benchmark.py [--output results.json]
"""

import sys
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.code_scalpel.code_analyzer import CodeAnalyzer
from src.code_scalpel.surgical_extractor import SurgicalExtractor


@dataclass
class ExtractionTask:
    """A code extraction task for benchmarking."""
    name: str
    description: str
    target_symbol: str
    file_path: str
    task_type: str  # "refactor", "understand", "test", "debug"


@dataclass
class ExtractionResult:
    """Results from a single extraction task."""
    task_name: str
    target_symbol: str

    # Full file approach metrics
    full_file_lines: int
    full_file_tokens: int
    full_file_chars: int

    # Surgical extraction metrics
    surgical_lines: int
    surgical_tokens: int
    surgical_chars: int
    dependencies_included: int

    # Ratios
    line_reduction_ratio: float
    token_reduction_ratio: float
    token_savings_percentage: float

    # Timing
    extraction_time_ms: float


class TokenEfficiencyBenchmark:
    """Benchmark for measuring token efficiency of surgical extraction."""

    SAMPLE_CODEBASE = Path(__file__).parent / "sample_codebase"

    # Realistic extraction tasks that an LLM might need to perform
    TASKS = [
        ExtractionTask(
            name="Refactor password hashing",
            description="Change password hashing algorithm in User model",
            target_symbol="User.hash_password",
            file_path="models/user.py",
            task_type="refactor"
        ),
        ExtractionTask(
            name="Fix login validation",
            description="Debug authentication logic",
            target_symbol="AuthService.authenticate",
            file_path="services/auth_service.py",
            task_type="debug"
        ),
        ExtractionTask(
            name="Add 2FA verification",
            description="Understand 2FA flow for enhancement",
            target_symbol="AuthService.authenticate_with_2fa",
            file_path="services/auth_service.py",
            task_type="understand"
        ),
        ExtractionTask(
            name="Test token generation",
            description="Write tests for JWT token generation",
            target_symbol="AuthService._generate_tokens",
            file_path="services/auth_service.py",
            task_type="test"
        ),
        ExtractionTask(
            name="Refactor user preferences",
            description="Add new preference field to UserPreferences",
            target_symbol="UserPreferences",
            file_path="models/user.py",
            task_type="refactor"
        ),
        ExtractionTask(
            name="Fix permission check",
            description="Debug permission validation logic",
            target_symbol="PermissionService.has_permission",
            file_path="services/auth_service.py",
            task_type="debug"
        ),
        ExtractionTask(
            name="Understand user serialization",
            description="Review user to_dict method",
            target_symbol="User.to_dict",
            file_path="models/user.py",
            task_type="understand"
        ),
        ExtractionTask(
            name="Add login endpoint validation",
            description="Review login endpoint for security",
            target_symbol="login",
            file_path="api/routes.py",
            task_type="understand"
        ),
        ExtractionTask(
            name="Refactor user repository",
            description="Add caching to UserRepository",
            target_symbol="UserRepository.find_by_id",
            file_path="models/user.py",
            task_type="refactor"
        ),
        ExtractionTask(
            name="Test password change",
            description="Write tests for password change flow",
            target_symbol="AuthService.change_password",
            file_path="services/auth_service.py",
            task_type="test"
        ),
    ]

    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.results: List[ExtractionResult] = []

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: chars/4)."""
        return len(text) // 4

    def count_lines(self, text: str) -> int:
        """Count non-empty lines."""
        return len([l for l in text.split('\n') if l.strip()])

    def get_full_file_content(self, file_path: str) -> str:
        """Read entire file content."""
        full_path = self.SAMPLE_CODEBASE / file_path
        if full_path.exists():
            return full_path.read_text()
        return ""

    def get_all_related_files(self, file_path: str) -> str:
        """Get all potentially related files (naive RAG simulation)."""
        # Simulate what a naive RAG system might retrieve
        content_parts = []

        # Always include the target file
        target_content = self.get_full_file_content(file_path)
        content_parts.append(f"# File: {file_path}\n{target_content}")

        # Include all files in the same directory
        target_dir = (self.SAMPLE_CODEBASE / file_path).parent
        for other_file in target_dir.glob("*.py"):
            if other_file.name != Path(file_path).name:
                rel_path = other_file.relative_to(self.SAMPLE_CODEBASE)
                content_parts.append(f"# File: {rel_path}\n{other_file.read_text()}")

        # Include commonly imported files
        for related_dir in ["models", "services", "api"]:
            related_path = self.SAMPLE_CODEBASE / related_dir
            if related_path.exists() and related_path != target_dir:
                for py_file in related_path.glob("*.py"):
                    if not py_file.name.startswith("__"):
                        rel_path = py_file.relative_to(self.SAMPLE_CODEBASE)
                        content_parts.append(f"# File: {rel_path}\n{py_file.read_text()}")

        return "\n\n".join(content_parts)

    def surgical_extract(self, file_path: str, target_symbol: str) -> tuple[str, int]:
        """Perform surgical extraction of target symbol with dependencies."""
        full_path = self.SAMPLE_CODEBASE / file_path
        if not full_path.exists():
            return "", 0

        code = full_path.read_text()

        try:
            extractor = SurgicalExtractor(code)

            # Extract the target symbol
            if '.' in target_symbol:
                class_name, method_name = target_symbol.split('.', 1)
                result = extractor.get_method(class_name, method_name)
            else:
                result = extractor.get_function(target_symbol)
                if not result:
                    result = extractor.get_class(target_symbol)

            if result and result.code:
                # Build context with dependencies
                parts = [f"# Target: {target_symbol}\n{result.code}"]
                dep_count = 0

                # Add dependencies
                if hasattr(result, 'dependencies') and result.dependencies:
                    for dep in result.dependencies:
                        dep_code = extractor.get_function(dep)
                        if dep_code and dep_code.code:
                            parts.append(f"# Dependency: {dep}\n{dep_code.code}")
                            dep_count += 1

                # Add required imports
                if hasattr(result, 'imports_needed') and result.imports_needed:
                    imports_str = "\n".join(result.imports_needed)
                    parts.insert(0, f"# Required imports:\n{imports_str}")

                return "\n\n".join(parts), dep_count

            return code, 0  # Fallback to full file

        except Exception as e:
            print(f"Extraction error for {target_symbol}: {e}")
            return code, 0

    def run_single_task(self, task: ExtractionTask) -> ExtractionResult:
        """Run a single extraction task and measure results."""
        start_time = time.time()

        # Get full file approach (simulating naive RAG)
        full_content = self.get_all_related_files(task.file_path)
        full_file_lines = self.count_lines(full_content)
        full_file_chars = len(full_content)
        full_file_tokens = self.estimate_tokens(full_content)

        # Get surgical extraction
        surgical_content, dep_count = self.surgical_extract(task.file_path, task.target_symbol)
        surgical_lines = self.count_lines(surgical_content)
        surgical_chars = len(surgical_content)
        surgical_tokens = self.estimate_tokens(surgical_content)

        extraction_time = (time.time() - start_time) * 1000

        # Calculate ratios
        line_ratio = surgical_lines / full_file_lines if full_file_lines > 0 else 1
        token_ratio = surgical_tokens / full_file_tokens if full_file_tokens > 0 else 1
        savings_pct = (1 - token_ratio) * 100

        return ExtractionResult(
            task_name=task.name,
            target_symbol=task.target_symbol,
            full_file_lines=full_file_lines,
            full_file_tokens=full_file_tokens,
            full_file_chars=full_file_chars,
            surgical_lines=surgical_lines,
            surgical_tokens=surgical_tokens,
            surgical_chars=surgical_chars,
            dependencies_included=dep_count,
            line_reduction_ratio=round(line_ratio, 4),
            token_reduction_ratio=round(token_ratio, 4),
            token_savings_percentage=round(savings_pct, 2),
            extraction_time_ms=round(extraction_time, 2)
        )

    def run_all_tasks(self) -> Dict[str, Any]:
        """Run all extraction tasks and generate report."""
        print(f"\n{'='*70}")
        print("CODE SCALPEL TOKEN EFFICIENCY BENCHMARK")
        print(f"{'='*70}")
        print(f"Sample codebase: {self.SAMPLE_CODEBASE}")
        print(f"Total tasks: {len(self.TASKS)}")
        print(f"Started: {datetime.now().isoformat()}")
        print()

        self.results = []
        start_time = time.time()

        for i, task in enumerate(self.TASKS, 1):
            print(f"[{i}/{len(self.TASKS)}] {task.name}: {task.target_symbol}")
            result = self.run_single_task(task)
            self.results.append(result)
            print(f"         Full: {result.full_file_tokens} tokens -> Surgical: {result.surgical_tokens} tokens ({result.token_savings_percentage}% saved)")

        total_time = time.time() - start_time

        return self.generate_report(total_time)

    def generate_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        # Aggregate metrics
        total_full_tokens = sum(r.full_file_tokens for r in self.results)
        total_surgical_tokens = sum(r.surgical_tokens for r in self.results)
        avg_savings = sum(r.token_savings_percentage for r in self.results) / len(self.results)
        avg_ratio = sum(r.token_reduction_ratio for r in self.results) / len(self.results)

        # Best and worst cases
        best_case = max(self.results, key=lambda r: r.token_savings_percentage)
        worst_case = min(self.results, key=lambda r: r.token_savings_percentage)

        report = {
            "benchmark_info": {
                "name": "Code Scalpel Token Efficiency Benchmark",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "total_execution_time_seconds": round(total_time, 2),
                "codebase_path": str(self.SAMPLE_CODEBASE),
                "total_tasks": len(self.TASKS)
            },
            "summary": {
                "total_tokens_naive_approach": total_full_tokens,
                "total_tokens_surgical_approach": total_surgical_tokens,
                "total_tokens_saved": total_full_tokens - total_surgical_tokens,
                "average_token_savings_percentage": round(avg_savings, 2),
                "average_reduction_ratio": round(avg_ratio, 4),
                "compression_factor": round(1 / avg_ratio, 1) if avg_ratio > 0 else 0
            },
            "best_case": {
                "task": best_case.task_name,
                "target": best_case.target_symbol,
                "savings_percentage": best_case.token_savings_percentage,
                "full_tokens": best_case.full_file_tokens,
                "surgical_tokens": best_case.surgical_tokens
            },
            "worst_case": {
                "task": worst_case.task_name,
                "target": worst_case.target_symbol,
                "savings_percentage": worst_case.token_savings_percentage,
                "full_tokens": worst_case.full_file_tokens,
                "surgical_tokens": worst_case.surgical_tokens
            },
            "by_task_type": self._aggregate_by_task_type(),
            "detailed_results": [asdict(r) for r in self.results],
            "claim_validation": {
                "claim": "Feed the LLM 50 lines, not 5,000 lines",
                "validated": avg_savings >= 70,
                "evidence": f"Average token reduction: {round(avg_savings, 1)}%",
                "interpretation": self._get_interpretation(avg_savings)
            }
        }

        return report

    def _aggregate_by_task_type(self) -> Dict[str, Dict]:
        """Aggregate results by task type."""
        task_types = {}
        for task, result in zip(self.TASKS, self.results):
            t_type = task.task_type
            if t_type not in task_types:
                task_types[t_type] = {"count": 0, "total_savings": 0}
            task_types[t_type]["count"] += 1
            task_types[t_type]["total_savings"] += result.token_savings_percentage

        for t_type, data in task_types.items():
            data["average_savings_percentage"] = round(data["total_savings"] / data["count"], 2)
            del data["total_savings"]

        return task_types

    def _get_interpretation(self, avg_savings: float) -> str:
        """Get human-readable interpretation of results."""
        if avg_savings >= 90:
            return "Excellent: Surgical extraction provides >10x token reduction"
        elif avg_savings >= 80:
            return "Very Good: Surgical extraction provides 5-10x token reduction"
        elif avg_savings >= 70:
            return "Good: Surgical extraction provides 3-5x token reduction"
        elif avg_savings >= 50:
            return "Moderate: Surgical extraction provides 2x token reduction"
        else:
            return "Limited: Surgical extraction provides marginal improvement"

    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable summary."""
        print(f"\n{'='*70}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*70}\n")

        summary = report["summary"]
        print(f"Total tokens (naive RAG approach):     {summary['total_tokens_naive_approach']:,}")
        print(f"Total tokens (surgical extraction):   {summary['total_tokens_surgical_approach']:,}")
        print(f"Tokens saved:                         {summary['total_tokens_saved']:,}")
        print(f"Average token savings:                {summary['average_token_savings_percentage']}%")
        print(f"Compression factor:                   {summary['compression_factor']}x")

        print(f"\n{'='*70}")
        print("BEST / WORST CASES")
        print(f"{'='*70}\n")

        best = report["best_case"]
        worst = report["worst_case"]
        print(f"Best case:  {best['task']} - {best['savings_percentage']}% saved")
        print(f"            ({best['full_tokens']} -> {best['surgical_tokens']} tokens)")
        print(f"Worst case: {worst['task']} - {worst['savings_percentage']}% saved")
        print(f"            ({worst['full_tokens']} -> {worst['surgical_tokens']} tokens)")

        print(f"\n{'='*70}")
        print("BY TASK TYPE")
        print(f"{'='*70}\n")

        for task_type, data in report["by_task_type"].items():
            print(f"{task_type:<12}: {data['average_savings_percentage']}% avg savings ({data['count']} tasks)")

        print(f"\n{'='*70}")
        print("CLAIM VALIDATION")
        print(f"{'='*70}\n")

        claim = report["claim_validation"]
        status = "VALIDATED" if claim["validated"] else "NOT VALIDATED"
        print(f"Claim: \"{claim['claim']}\"")
        print(f"Status: {status}")
        print(f"Evidence: {claim['evidence']}")
        print(f"Interpretation: {claim['interpretation']}")

        print(f"\n{'='*70}")
        print("DETAILED RESULTS")
        print(f"{'='*70}\n")

        print(f"{'Task':<35} {'Full Tokens':<12} {'Surgical':<12} {'Savings':<10}")
        print("-" * 70)

        for result in self.results:
            print(f"{result.task_name:<35} {result.full_file_tokens:<12} {result.surgical_tokens:<12} {result.token_savings_percentage}%")

        print(f"\n{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Code Scalpel token efficiency benchmark")
    parser.add_argument("--output", "-o", default="results.json", help="Output file for JSON results")
    args = parser.parse_args()

    benchmark = TokenEfficiencyBenchmark()
    report = benchmark.run_all_tasks()
    benchmark.print_summary(report)

    # Save results
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Detailed results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
