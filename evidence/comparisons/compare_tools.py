#!/usr/bin/env python3
"""
Code Scalpel Comparison Framework
=================================

Compares Code Scalpel's vulnerability detection against:
- Semgrep (if installed)
- Bandit (if installed)

This provides objective evidence of Code Scalpel's detection capabilities
relative to industry-standard tools.

Usage:
    python compare_tools.py [--output results.json]
"""

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.code_scalpel.code_analyzer import CodeAnalyzer


@dataclass
class ComparisonTestCase:
    """A test case for tool comparison."""
    id: str
    cwe_id: str
    name: str
    code: str
    is_vulnerable: bool
    vulnerability_type: str


# Focused test cases covering key vulnerability types
COMPARISON_TEST_CASES = [
    # SQL Injection
    ComparisonTestCase(
        id="SQL-001",
        cwe_id="CWE-89",
        name="SQL injection via string concatenation",
        code='''
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()
''',
        is_vulnerable=True,
        vulnerability_type="sql_injection"
    ),
    ComparisonTestCase(
        id="SQL-002",
        cwe_id="CWE-89",
        name="SQL injection via f-string",
        code='''
import sqlite3

def search_products(term):
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM products WHERE name LIKE '%{term}%'")
    return cursor.fetchall()
''',
        is_vulnerable=True,
        vulnerability_type="sql_injection"
    ),
    ComparisonTestCase(
        id="SQL-003",
        cwe_id="CWE-89",
        name="Safe parameterized query",
        code='''
import sqlite3

def get_user_safe(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name = ?", (username,))
    return cursor.fetchone()
''',
        is_vulnerable=False,
        vulnerability_type="sql_injection"
    ),

    # Command Injection
    ComparisonTestCase(
        id="CMD-001",
        cwe_id="CWE-78",
        name="Command injection via os.system",
        code='''
import os

def ping_host(hostname):
    os.system("ping -c 4 " + hostname)
''',
        is_vulnerable=True,
        vulnerability_type="command_injection"
    ),
    ComparisonTestCase(
        id="CMD-002",
        cwe_id="CWE-78",
        name="Command injection via subprocess shell=True",
        code='''
import subprocess

def run_command(user_input):
    subprocess.run(f"echo {user_input}", shell=True)
''',
        is_vulnerable=True,
        vulnerability_type="command_injection"
    ),
    ComparisonTestCase(
        id="CMD-003",
        cwe_id="CWE-78",
        name="Safe subprocess without shell",
        code='''
import subprocess

def run_command_safe(args):
    subprocess.run(["echo", args])
''',
        is_vulnerable=False,
        vulnerability_type="command_injection"
    ),

    # Code Injection
    ComparisonTestCase(
        id="CODE-001",
        cwe_id="CWE-94",
        name="Code injection via eval",
        code='''
def calculate(expression):
    return eval(expression)
''',
        is_vulnerable=True,
        vulnerability_type="code_injection"
    ),
    ComparisonTestCase(
        id="CODE-002",
        cwe_id="CWE-94",
        name="Code injection via exec",
        code='''
def run_code(code):
    exec(code)
''',
        is_vulnerable=True,
        vulnerability_type="code_injection"
    ),

    # Insecure Deserialization
    ComparisonTestCase(
        id="DESER-001",
        cwe_id="CWE-502",
        name="Insecure pickle deserialization",
        code='''
import pickle

def load_data(data):
    return pickle.loads(data)
''',
        is_vulnerable=True,
        vulnerability_type="insecure_deserialization"
    ),
    ComparisonTestCase(
        id="DESER-002",
        cwe_id="CWE-502",
        name="Insecure yaml.load",
        code='''
import yaml

def parse_yaml(content):
    return yaml.load(content)
''',
        is_vulnerable=True,
        vulnerability_type="insecure_deserialization"
    ),
    ComparisonTestCase(
        id="DESER-003",
        cwe_id="CWE-502",
        name="Safe yaml.safe_load",
        code='''
import yaml

def parse_yaml_safe(content):
    return yaml.safe_load(content)
''',
        is_vulnerable=False,
        vulnerability_type="insecure_deserialization"
    ),

    # Weak Cryptography
    ComparisonTestCase(
        id="CRYPTO-001",
        cwe_id="CWE-327",
        name="Weak MD5 hashing",
        code='''
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
''',
        is_vulnerable=True,
        vulnerability_type="weak_cryptography"
    ),
    ComparisonTestCase(
        id="CRYPTO-002",
        cwe_id="CWE-327",
        name="Weak SHA1 hashing",
        code='''
import hashlib

def create_token(data):
    return hashlib.sha1(data.encode()).hexdigest()
''',
        is_vulnerable=True,
        vulnerability_type="weak_cryptography"
    ),

    # Hardcoded Secrets
    ComparisonTestCase(
        id="SECRET-001",
        cwe_id="CWE-798",
        name="Hardcoded API key",
        code='''
API_KEY = "AKIAIOSFODNN7EXAMPLE"  # AWS-style key format

def make_request():
    return {"Authorization": f"Bearer {API_KEY}"}
''',
        is_vulnerable=True,
        vulnerability_type="hardcoded_secret"
    ),
    ComparisonTestCase(
        id="SECRET-002",
        cwe_id="CWE-798",
        name="Hardcoded password",
        code='''
DATABASE_PASSWORD = "super_secret_password123"

def connect():
    return {"password": DATABASE_PASSWORD}
''',
        is_vulnerable=True,
        vulnerability_type="hardcoded_secret"
    ),

    # Path Traversal
    ComparisonTestCase(
        id="PATH-001",
        cwe_id="CWE-22",
        name="Path traversal via file read",
        code='''
def read_file(filename):
    with open("/var/data/" + filename, 'r') as f:
        return f.read()
''',
        is_vulnerable=True,
        vulnerability_type="path_traversal"
    ),

    # SSRF
    ComparisonTestCase(
        id="SSRF-001",
        cwe_id="CWE-918",
        name="SSRF via requests.get",
        code='''
import requests

def fetch_url(url):
    return requests.get(url).text
''',
        is_vulnerable=True,
        vulnerability_type="ssrf"
    ),
]


@dataclass
class ToolResult:
    """Result from running a tool on a test case."""
    tool: str
    detected: bool
    findings: List[Dict[str, Any]]
    execution_time_ms: float
    error: Optional[str] = None


class ToolRunner:
    """Runs security analysis tools on code."""

    def __init__(self):
        from src.code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
        self.security_analyzer = SecurityAnalyzer()

    def check_tool_available(self, tool: str) -> bool:
        """Check if a tool is available."""
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def run_code_scalpel(self, code: str) -> ToolResult:
        """Run Code Scalpel analysis."""
        import time
        start = time.time()

        try:
            result = self.security_analyzer.analyze(code)
            vulnerabilities = result.vulnerabilities if result else []

            return ToolResult(
                tool="code_scalpel",
                detected=len(vulnerabilities) > 0,
                findings=[
                    {
                        "type": getattr(v, "vulnerability_type", "unknown"),
                        "line": getattr(v, "line_number", None),
                        "message": getattr(v, "description", "")
                    }
                    for v in vulnerabilities
                ],
                execution_time_ms=round((time.time() - start) * 1000, 2)
            )
        except Exception as e:
            return ToolResult(
                tool="code_scalpel",
                detected=False,
                findings=[],
                execution_time_ms=round((time.time() - start) * 1000, 2),
                error=str(e)
            )

    def run_bandit(self, code: str) -> ToolResult:
        """Run Bandit analysis."""
        import time

        if not self.check_tool_available("bandit"):
            return ToolResult(
                tool="bandit",
                detected=False,
                findings=[],
                execution_time_ms=0,
                error="Bandit not installed"
            )

        start = time.time()

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ["bandit", "-f", "json", temp_file],
                capture_output=True,
                text=True
            )

            os.unlink(temp_file)

            findings = []
            if result.stdout:
                try:
                    bandit_result = json.loads(result.stdout)
                    findings = [
                        {"type": r.get("test_id"), "line": r.get("line_number"), "message": r.get("issue_text")}
                        for r in bandit_result.get("results", [])
                    ]
                except json.JSONDecodeError:
                    pass

            return ToolResult(
                tool="bandit",
                detected=len(findings) > 0,
                findings=findings,
                execution_time_ms=round((time.time() - start) * 1000, 2)
            )

        except Exception as e:
            return ToolResult(
                tool="bandit",
                detected=False,
                findings=[],
                execution_time_ms=round((time.time() - start) * 1000, 2),
                error=str(e)
            )

    def run_semgrep(self, code: str) -> ToolResult:
        """Run Semgrep analysis."""
        import time

        if not self.check_tool_available("semgrep"):
            return ToolResult(
                tool="semgrep",
                detected=False,
                findings=[],
                execution_time_ms=0,
                error="Semgrep not installed"
            )

        start = time.time()

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            result = subprocess.run(
                ["semgrep", "--config", "auto", "--json", temp_file],
                capture_output=True,
                text=True
            )

            os.unlink(temp_file)

            findings = []
            if result.stdout:
                try:
                    semgrep_result = json.loads(result.stdout)
                    findings = [
                        {
                            "type": r.get("check_id"),
                            "line": r.get("start", {}).get("line"),
                            "message": r.get("extra", {}).get("message", "")
                        }
                        for r in semgrep_result.get("results", [])
                    ]
                except json.JSONDecodeError:
                    pass

            return ToolResult(
                tool="semgrep",
                detected=len(findings) > 0,
                findings=findings,
                execution_time_ms=round((time.time() - start) * 1000, 2)
            )

        except Exception as e:
            return ToolResult(
                tool="semgrep",
                detected=False,
                findings=[],
                execution_time_ms=round((time.time() - start) * 1000, 2),
                error=str(e)
            )


class ComparisonBenchmark:
    """Compares Code Scalpel against other security tools."""

    def __init__(self):
        self.runner = ToolRunner()
        self.results: Dict[str, List[Dict]] = {}

    def run_comparison(self) -> Dict[str, Any]:
        """Run comparison across all test cases."""
        print(f"\n{'='*70}")
        print("CODE SCALPEL TOOL COMPARISON BENCHMARK")
        print(f"{'='*70}")
        print(f"Test cases: {len(COMPARISON_TEST_CASES)}")
        print(f"Started: {datetime.now().isoformat()}")

        # Check tool availability
        tools = ["code_scalpel"]
        if self.runner.check_tool_available("bandit"):
            tools.append("bandit")
            print("Bandit: Available")
        else:
            print("Bandit: Not installed (will skip)")

        if self.runner.check_tool_available("semgrep"):
            tools.append("semgrep")
            print("Semgrep: Available")
        else:
            print("Semgrep: Not installed (will skip)")

        print()

        self.results = {tool: [] for tool in tools}

        for i, test_case in enumerate(COMPARISON_TEST_CASES, 1):
            print(f"[{i}/{len(COMPARISON_TEST_CASES)}] {test_case.id}: {test_case.name}")

            for tool in tools:
                if tool == "code_scalpel":
                    result = self.runner.run_code_scalpel(test_case.code)
                elif tool == "bandit":
                    result = self.runner.run_bandit(test_case.code)
                elif tool == "semgrep":
                    result = self.runner.run_semgrep(test_case.code)
                else:
                    continue

                # Determine correctness
                if test_case.is_vulnerable:
                    correct = result.detected  # Should detect
                else:
                    correct = not result.detected  # Should not detect (no false positive)

                self.results[tool].append({
                    "test_id": test_case.id,
                    "cwe_id": test_case.cwe_id,
                    "is_vulnerable": test_case.is_vulnerable,
                    "detected": result.detected,
                    "correct": correct,
                    "findings_count": len(result.findings),
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error
                })

            # Print quick status
            cs_result = self.results["code_scalpel"][-1]
            status = "PASS" if cs_result["correct"] else "FAIL"
            print(f"  Code Scalpel: {status}")

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate comparison report."""
        tool_summaries = {}

        for tool, results in self.results.items():
            # Skip if tool wasn't available
            if not results:
                continue

            total = len(results)
            correct = sum(1 for r in results if r["correct"])
            vulnerable_cases = [r for r in results if r["is_vulnerable"]]
            safe_cases = [r for r in results if not r["is_vulnerable"]]

            true_positives = sum(1 for r in vulnerable_cases if r["detected"])
            false_negatives = sum(1 for r in vulnerable_cases if not r["detected"])
            true_negatives = sum(1 for r in safe_cases if not r["detected"])
            false_positives = sum(1 for r in safe_cases if r["detected"])

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            avg_time = sum(r["execution_time_ms"] for r in results) / total

            tool_summaries[tool] = {
                "total_tests": total,
                "correct": correct,
                "accuracy_percentage": round(correct / total * 100, 1),
                "true_positives": true_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(f1, 3),
                "detection_rate_percentage": round(true_positives / len(vulnerable_cases) * 100, 1) if vulnerable_cases else 100,
                "false_positive_rate_percentage": round(false_positives / len(safe_cases) * 100, 1) if safe_cases else 0,
                "avg_execution_time_ms": round(avg_time, 2)
            }

        # Create comparison table
        comparison_table = []
        cwe_types = set(t.cwe_id for t in COMPARISON_TEST_CASES)

        for cwe in sorted(cwe_types):
            row = {"cwe_id": cwe}
            cwe_cases = [t for t in COMPARISON_TEST_CASES if t.cwe_id == cwe]
            cwe_vulnerable = [t for t in cwe_cases if t.is_vulnerable]

            for tool in self.results.keys():
                tool_cwe_results = [r for r in self.results[tool] if r["cwe_id"] == cwe and r["is_vulnerable"]]
                detected = sum(1 for r in tool_cwe_results if r["detected"])
                row[tool] = f"{detected}/{len(tool_cwe_results)}"

            comparison_table.append(row)

        report = {
            "benchmark_info": {
                "name": "Code Scalpel Tool Comparison Benchmark",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "total_test_cases": len(COMPARISON_TEST_CASES),
                "tools_compared": list(self.results.keys())
            },
            "tool_summaries": tool_summaries,
            "comparison_by_cwe": comparison_table,
            "detailed_results": self.results,
            "conclusion": self._generate_conclusion(tool_summaries)
        }

        return report

    def _generate_conclusion(self, summaries: Dict) -> str:
        """Generate conclusion based on results."""
        if "code_scalpel" not in summaries:
            return "Code Scalpel results not available"

        cs = summaries["code_scalpel"]
        conclusions = [f"Code Scalpel achieved {cs['accuracy_percentage']}% accuracy with {cs['detection_rate_percentage']}% detection rate."]

        for tool in ["bandit", "semgrep"]:
            if tool in summaries:
                t = summaries[tool]
                if cs["detection_rate_percentage"] > t["detection_rate_percentage"]:
                    conclusions.append(f"Code Scalpel outperformed {tool.capitalize()} in detection rate ({cs['detection_rate_percentage']}% vs {t['detection_rate_percentage']}%).")
                elif cs["detection_rate_percentage"] < t["detection_rate_percentage"]:
                    conclusions.append(f"{tool.capitalize()} had higher detection rate ({t['detection_rate_percentage']}% vs {cs['detection_rate_percentage']}%).")
                else:
                    conclusions.append(f"Code Scalpel matched {tool.capitalize()} in detection rate.")

        return " ".join(conclusions)

    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable summary."""
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}\n")

        # Tool comparison table
        print(f"{'Tool':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Detect%':<10} {'FP%':<8}")
        print("-" * 73)

        for tool, summary in report["tool_summaries"].items():
            print(f"{tool:<15} {summary['accuracy_percentage']:<10} {summary['precision']:<10} {summary['recall']:<10} {summary['f1_score']:<10} {summary['detection_rate_percentage']:<10} {summary['false_positive_rate_percentage']:<8}")

        print(f"\n{'='*70}")
        print("DETECTION BY CWE TYPE")
        print(f"{'='*70}\n")

        tools = list(report["tool_summaries"].keys())
        header = f"{'CWE':<10}" + "".join(f"{t:<15}" for t in tools)
        print(header)
        print("-" * (10 + 15 * len(tools)))

        for row in report["comparison_by_cwe"]:
            line = f"{row['cwe_id']:<10}"
            for tool in tools:
                line += f"{row.get(tool, 'N/A'):<15}"
            print(line)

        print(f"\n{'='*70}")
        print("CONCLUSION")
        print(f"{'='*70}\n")
        print(report["conclusion"])
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Code Scalpel with other security tools")
    parser.add_argument("--output", "-o", default="comparison_results.json", help="Output file")
    args = parser.parse_args()

    benchmark = ComparisonBenchmark()
    report = benchmark.run_comparison()
    benchmark.print_summary(report)

    # Save results
    output_path = Path(__file__).parent / args.output
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Detailed results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
