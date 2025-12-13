#!/usr/bin/env python3
"""
Code Scalpel Evidence Generation Suite
======================================

Master script to run all benchmarks and generate comprehensive evidence
for Code Scalpel's claims.

Usage:
    python run_all_benchmarks.py [--quick] [--verbose]
"""

import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


EVIDENCE_DIR = Path(__file__).parent


def run_benchmark(name: str, script_path: Path, output_file: str, verbose: bool = False) -> Dict[str, Any]:
    """Run a single benchmark and return results summary."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        cmd = [sys.executable, str(script_path), "-o", output_file]
        if verbose:
            cmd.append("-v")

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            cwd=script_path.parent
        )

        elapsed = time.time() - start_time

        output_path = script_path.parent / output_file
        if output_path.exists():
            with open(output_path) as f:
                results = json.load(f)
        else:
            results = {}

        return {
            "name": name,
            "status": "success" if result.returncode == 0 else "failed",
            "execution_time_seconds": round(elapsed, 2),
            "output_file": str(output_path),
            "summary": extract_summary(name, results),
            "error": result.stderr if result.returncode != 0 else None
        }

    except Exception as e:
        return {
            "name": name,
            "status": "error",
            "error": str(e),
            "execution_time_seconds": time.time() - start_time
        }


def extract_summary(benchmark_name: str, results: Dict) -> Dict[str, Any]:
    """Extract key summary metrics from benchmark results."""
    if not results:
        return {}

    if "vulnerability" in benchmark_name.lower():
        summary = results.get("summary", {})
        return {
            "accuracy": summary.get("accuracy_percentage"),
            "precision": summary.get("precision"),
            "recall": summary.get("recall"),
            "f1_score": summary.get("f1_score")
        }

    elif "token" in benchmark_name.lower():
        summary = results.get("summary", {})
        return {
            "average_token_savings_percentage": summary.get("average_token_savings_percentage"),
            "compression_factor": summary.get("compression_factor")
        }

    elif "cache" in benchmark_name.lower():
        summary = results.get("summary", {})
        return {
            "average_speedup": summary.get("average_speedup_ratio"),
            "max_speedup": summary.get("max_speedup_ratio")
        }

    elif "comparison" in benchmark_name.lower():
        summaries = results.get("tool_summaries", {})
        if "code_scalpel" in summaries:
            cs = summaries["code_scalpel"]
            return {
                "code_scalpel_accuracy": cs.get("accuracy_percentage"),
                "code_scalpel_detection_rate": cs.get("detection_rate_percentage"),
                "tools_compared": list(summaries.keys())
            }

    return {}


def run_demo(verbose: bool = False) -> Dict[str, Any]:
    """Run the refactoring demo."""
    print(f"\n{'='*60}")
    print("Running: Refactoring Demo")
    print(f"{'='*60}")

    demo_path = EVIDENCE_DIR / "demos" / "refactoring_demo.py"

    try:
        result = subprocess.run(
            [sys.executable, str(demo_path)],
            capture_output=not verbose,
            text=True,
            cwd=demo_path.parent
        )

        return {
            "name": "Refactoring Demo",
            "status": "success" if result.returncode == 0 else "failed",
            "output": result.stdout if not verbose else "See console output"
        }

    except Exception as e:
        return {
            "name": "Refactoring Demo",
            "status": "error",
            "error": str(e)
        }


def generate_summary_report(benchmark_results: list, demo_result: Dict) -> Dict[str, Any]:
    """Generate consolidated summary report."""
    return {
        "report_info": {
            "name": "Code Scalpel Evidence Report",
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0"
        },
        "overall_status": "pass" if all(r["status"] == "success" for r in benchmark_results) else "partial",
        "benchmarks_run": len(benchmark_results),
        "benchmarks_passed": sum(1 for r in benchmark_results if r["status"] == "success"),
        "total_execution_time_seconds": sum(r.get("execution_time_seconds", 0) for r in benchmark_results),
        "benchmark_results": benchmark_results,
        "demo_result": demo_result,
        "claims_summary": {
            "vulnerability_detection": next(
                (r["summary"] for r in benchmark_results if "vulnerability" in r["name"].lower()),
                {}
            ),
            "token_efficiency": next(
                (r["summary"] for r in benchmark_results if "token" in r["name"].lower()),
                {}
            ),
            "cache_performance": next(
                (r["summary"] for r in benchmark_results if "cache" in r["name"].lower()),
                {}
            ),
            "tool_comparison": next(
                (r["summary"] for r in benchmark_results if "comparison" in r["name"].lower()),
                {}
            )
        }
    }


def print_final_summary(report: Dict[str, Any]):
    """Print final summary to console."""
    print()
    print("=" * 70)
    print("EVIDENCE GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"Benchmarks run: {report['benchmarks_run']}")
    print(f"Benchmarks passed: {report['benchmarks_passed']}")
    print(f"Total time: {report['total_execution_time_seconds']:.1f} seconds")
    print()

    print("CLAIMS EVIDENCE SUMMARY:")
    print("-" * 70)

    claims = report["claims_summary"]

    if claims.get("vulnerability_detection"):
        vd = claims["vulnerability_detection"]
        print(f"Vulnerability Detection:")
        print(f"  Accuracy: {vd.get('accuracy', 'N/A')}%")
        print(f"  Precision: {vd.get('precision', 'N/A')}")
        print(f"  Recall: {vd.get('recall', 'N/A')}")
        print()

    if claims.get("token_efficiency"):
        te = claims["token_efficiency"]
        print(f"Token Efficiency:")
        print(f"  Average savings: {te.get('average_token_savings_percentage', 'N/A')}%")
        print(f"  Compression factor: {te.get('compression_factor', 'N/A')}x")
        print()

    if claims.get("cache_performance"):
        cp = claims["cache_performance"]
        print(f"Cache Performance:")
        print(f"  Average speedup: {cp.get('average_speedup', 'N/A')}x")
        print(f"  Max speedup: {cp.get('max_speedup', 'N/A')}x")
        print()

    if claims.get("tool_comparison"):
        tc = claims["tool_comparison"]
        print(f"Tool Comparison:")
        print(f"  Code Scalpel accuracy: {tc.get('code_scalpel_accuracy', 'N/A')}%")
        print(f"  Detection rate: {tc.get('code_scalpel_detection_rate', 'N/A')}%")
        if tc.get("tools_compared"):
            print(f"  Tools compared: {', '.join(tc['tools_compared'])}")
        print()

    print("-" * 70)
    print()
    print("Generated files:")
    for result in report["benchmark_results"]:
        if result.get("output_file"):
            print(f"  - {result['output_file']}")
    print(f"  - {EVIDENCE_DIR / 'evidence_report.json'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run all Code Scalpel evidence benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick version with fewer iterations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-comparison", action="store_true", help="Skip tool comparison (requires external tools)")
    args = parser.parse_args()

    print()
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "    CODE SCALPEL EVIDENCE GENERATION SUITE    ".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print()
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Evidence directory: {EVIDENCE_DIR}")
    print()

    benchmarks = [
        (
            "Vulnerability Detection Benchmark",
            EVIDENCE_DIR / "vulnerability-detection" / "run_benchmark.py",
            "results.json"
        ),
        (
            "Token Efficiency Benchmark",
            EVIDENCE_DIR / "token-efficiency" / "run_benchmark.py",
            "results.json"
        ),
        (
            "Cache Performance Benchmark",
            EVIDENCE_DIR / "performance" / "cache_benchmark.py",
            "results.json"
        ),
    ]

    if not args.skip_comparison:
        benchmarks.append((
            "Tool Comparison Benchmark",
            EVIDENCE_DIR / "comparisons" / "compare_tools.py",
            "comparison_results.json"
        ))

    # Run all benchmarks
    benchmark_results = []
    for name, script_path, output_file in benchmarks:
        if script_path.exists():
            result = run_benchmark(name, script_path, output_file, args.verbose)
            benchmark_results.append(result)
        else:
            print(f"Skipping {name}: script not found at {script_path}")

    # Run demo
    demo_result = run_demo(args.verbose)

    # Generate consolidated report
    report = generate_summary_report(benchmark_results, demo_result)

    # Save report
    report_path = EVIDENCE_DIR / "evidence_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print_final_summary(report)

    # Return exit code
    if report["overall_status"] == "pass":
        print("All benchmarks passed!")
        return 0
    else:
        print("Some benchmarks failed. Check individual results for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
