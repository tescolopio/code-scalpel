#!/usr/bin/env python3
"""
Benchmark runner for CodeAnalyzer.

This script runs the CodeAnalyzer against sample codebases and measures:
- Processing time
- Dead code detection accuracy
- Analysis metrics
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from code_analyzer import CodeAnalyzer, AnalysisLevel, AnalysisResult

# Benchmark configurations
BENCHMARKS = [
    {
        "name": "Flask App",
        "file": "sample_flask_app.py",
        "description": "Sample Flask web application",
        "expected_dead_code": [
            "unused_database_cleanup",
            "deprecated_decorator",
            "deprecated_method",
            "unused_helper_function",
        ],
    },
    {
        "name": "ML Script",
        "file": "sample_ml_script.py",
        "description": "Machine learning training pipeline",
        "expected_dead_code": [
            "DeprecatedOptimizer",
            "unused_regularization",
            "deprecated_normalize",
            "unused_feature_selection",
        ],
    },
    {
        "name": "Data Processor",
        "file": "sample_data_processor.py",
        "description": "Data processing library",
        "expected_dead_code": [
            "DeprecatedAggregator",
            "deprecated_parse_xml",
            "unused_validation_method",
            "deprecated_parallel_run",
            "unused_utility_function",
        ],
    },
]


def load_sample_code(filename: str) -> str:
    """Load sample code from benchmarks directory."""
    benchmarks_dir = Path(__file__).parent
    filepath = benchmarks_dir / filename

    with open(filepath, "r") as f:
        return f.read()


def count_lines_of_code(code: str) -> int:
    """Count non-empty, non-comment lines."""
    lines = 0
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines += 1
    return lines


def calculate_accuracy(result: AnalysisResult, expected: List[str]) -> Dict[str, float]:
    """Calculate dead code detection accuracy."""
    detected_names = {item.name for item in result.dead_code}

    # True positives: expected items that were detected
    true_positives = sum(1 for name in expected if name in detected_names)

    # False negatives: expected items that were NOT detected
    false_negatives = len(expected) - true_positives

    # Recall: what percentage of expected dead code was found
    recall = true_positives / len(expected) if expected else 1.0

    # Total detected
    total_detected = len(result.dead_code)

    return {
        "expected": len(expected),
        "detected": total_detected,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "recall": recall,
        "recall_percentage": recall * 100,
    }


def run_benchmark(benchmark: Dict[str, Any], analyzer: CodeAnalyzer) -> Dict[str, Any]:
    """Run a single benchmark."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark['name']}")
    print(f"Description: {benchmark['description']}")
    print("=" * 60)

    # Load code
    code = load_sample_code(benchmark["file"])
    loc = count_lines_of_code(code)

    print(f"Lines of code: {loc}")

    # Run analysis
    start_time = time.perf_counter()
    result = analyzer.analyze(code)
    end_time = time.perf_counter()

    analysis_time = end_time - start_time

    # Calculate throughput
    loc_per_second = loc / analysis_time if analysis_time > 0 else float("inf")

    # Calculate accuracy
    accuracy = calculate_accuracy(result, benchmark["expected_dead_code"])

    # Print results
    print("\n--- Results ---")
    print(f"Analysis time: {analysis_time:.4f}s")
    print(f"Throughput: {loc_per_second:.0f} LOC/s")
    print(f"Cyclomatic complexity: {result.metrics.cyclomatic_complexity}")
    print(f"Functions: {result.metrics.num_functions}")
    print(f"Classes: {result.metrics.num_classes}")

    print("\n--- Dead Code Detection ---")
    print(f"Expected dead code items: {accuracy['expected']}")
    print(f"Total items detected: {accuracy['detected']}")
    print(f"True positives: {accuracy['true_positives']}")
    print(f"False negatives: {accuracy['false_negatives']}")
    print(f"Recall: {accuracy['recall_percentage']:.1f}%")

    # List detected dead code
    print("\nDetected dead code items:")
    for item in result.dead_code[:10]:  # Limit to first 10
        print(
            f"  - {item.code_type}: {item.name} (line {item.line_start}, {int(item.confidence*100)}% confidence)"
        )
    if len(result.dead_code) > 10:
        print(f"  ... and {len(result.dead_code) - 10} more items")

    # Check for expected items
    detected_names = {item.name for item in result.dead_code}
    print("\nExpected items detection:")
    for expected_name in benchmark["expected_dead_code"]:
        status = "✓ FOUND" if expected_name in detected_names else "✗ MISSED"
        print(f"  {status}: {expected_name}")

    if result.errors:
        print(f"\nErrors: {result.errors}")

    return {
        "name": benchmark["name"],
        "loc": loc,
        "analysis_time": analysis_time,
        "loc_per_second": loc_per_second,
        "accuracy": accuracy,
        "metrics": {
            "complexity": result.metrics.cyclomatic_complexity,
            "functions": result.metrics.num_functions,
            "classes": result.metrics.num_classes,
        },
        "dead_code_count": len(result.dead_code),
        "errors": result.errors,
    }


def run_all_benchmarks() -> Dict[str, Any]:
    """Run all benchmarks and aggregate results."""
    print("=" * 60)
    print("CodeAnalyzer Benchmark Suite")
    print("=" * 60)

    # Create analyzer
    analyzer = CodeAnalyzer(level=AnalysisLevel.STANDARD, cache_enabled=False)

    results = []
    total_loc = 0
    total_time = 0.0
    total_expected = 0
    total_found = 0

    for benchmark in BENCHMARKS:
        try:
            result = run_benchmark(benchmark, analyzer)
            results.append(result)

            total_loc += result["loc"]
            total_time += result["analysis_time"]
            total_expected += result["accuracy"]["expected"]
            total_found += result["accuracy"]["true_positives"]

        except Exception as e:
            print(f"\nError running benchmark {benchmark['name']}: {e}")
            results.append({"name": benchmark["name"], "error": str(e)})

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    print(f"\nTotal lines of code analyzed: {total_loc}")
    print(f"Total analysis time: {total_time:.4f}s")
    print(f"Average throughput: {total_loc / total_time:.0f} LOC/s")

    overall_recall = (total_found / total_expected * 100) if total_expected > 0 else 0
    print("\nOverall dead code detection:")
    print(f"  Expected items: {total_expected}")
    print(f"  Found items: {total_found}")
    print(f"  Recall: {overall_recall:.1f}%")

    # Check success criteria
    print("\n" + "-" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("-" * 60)

    # Target: 1k LOC in <5s (200 LOC/s minimum)
    target_throughput = 200  # LOC/s
    actual_throughput = total_loc / total_time if total_time > 0 else 0
    throughput_pass = actual_throughput >= target_throughput

    print(f"Performance target: 1k LOC in <5s ({target_throughput} LOC/s minimum)")
    print(f"Actual throughput: {actual_throughput:.0f} LOC/s")
    print(f"Status: {'✓ PASS' if throughput_pass else '✗ FAIL'}")

    # Target: 80% dead code accuracy
    target_accuracy = 80.0
    accuracy_pass = overall_recall >= target_accuracy

    print(f"\nAccuracy target: {target_accuracy}% dead code detection")
    print(f"Actual recall: {overall_recall:.1f}%")
    print(f"Status: {'✓ PASS' if accuracy_pass else '✗ FAIL'}")

    overall_pass = throughput_pass and accuracy_pass
    print(f"\n{'='*60}")
    print(
        f"OVERALL RESULT: {'✓ ALL CRITERIA MET' if overall_pass else '✗ SOME CRITERIA NOT MET'}"
    )
    print(f"{'='*60}")

    return {
        "results": results,
        "summary": {
            "total_loc": total_loc,
            "total_time": total_time,
            "throughput": actual_throughput,
            "recall": overall_recall,
            "throughput_pass": throughput_pass,
            "accuracy_pass": accuracy_pass,
            "overall_pass": overall_pass,
        },
    }


if __name__ == "__main__":
    results = run_all_benchmarks()

    # Log failure reasons if any
    if not results["summary"]["overall_pass"]:
        print("\nBenchmark failed due to:")
        if not results["summary"]["throughput_pass"]:
            print(
                f"  - Throughput below target: {results['summary']['throughput']:.0f} LOC/s (target: 200 LOC/s)"
            )
        if not results["summary"]["accuracy_pass"]:
            print(
                f"  - Accuracy below target: {results['summary']['recall']:.1f}% (target: 80%)"
            )

    # Exit with appropriate code
    sys.exit(0 if results["summary"]["overall_pass"] else 1)
