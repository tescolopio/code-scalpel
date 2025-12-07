"""Refactor Simulator - Verify Code Changes Before Applying.

This module allows AI Agents to "dry run" their code changes,
verifying that a patch doesn't introduce security vulnerabilities
or break existing functionality.

Example:
    >>> from code_scalpel.generators import RefactorSimulator
    >>> simulator = RefactorSimulator()
    >>> result = simulator.simulate(original_code, patch)
    >>> if result.is_safe:
    ...     print("Safe to apply!")
    ... else:
    ...     print(f"UNSAFE: {result.reason}")
"""

import ast
import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RefactorStatus(Enum):
    """Status of a refactor simulation."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SecurityIssue:
    """A security issue detected in refactored code."""

    type: str
    severity: str
    line: int | None
    description: str
    cwe: str | None = None


@dataclass
class RefactorResult:
    """Result of simulating a refactor."""

    status: RefactorStatus
    is_safe: bool
    patched_code: str
    reason: str | None = None
    security_issues: list[SecurityIssue] = field(default_factory=list)
    structural_changes: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "is_safe": self.is_safe,
            "reason": self.reason,
            "security_issues": [
                {
                    "type": issue.type,
                    "severity": issue.severity,
                    "line": issue.line,
                    "description": issue.description,
                    "cwe": issue.cwe,
                }
                for issue in self.security_issues
            ],
            "structural_changes": self.structural_changes,
            "warnings": self.warnings,
        }


class RefactorSimulator:
    """Simulate code refactors and verify safety.

    This class allows AI Agents to test their code changes in a
    sandboxed environment before actually applying them. It:

    1. Applies the patch to get the new code
    2. Runs security analysis on the patched code
    3. Compares structural changes (added/removed functions, etc.)
    4. Returns a verdict: SAFE, UNSAFE, WARNING, or ERROR

    Example:
        >>> simulator = RefactorSimulator()
        >>> result = simulator.simulate(
        ...     original_code='def foo(x): return x',
        ...     patch='@@ -1 +1 @@\\n-def foo(x): return x\\n+def foo(x): return eval(x)'
        ... )
        >>> result.is_safe
        False
        >>> result.reason
        'Introduced Code Injection vulnerability'
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize the refactor simulator.

        Args:
            strict_mode: If True, treat warnings as unsafe
        """
        self.strict_mode = strict_mode

    def simulate(
        self,
        original_code: str,
        patch: str | None = None,
        new_code: str | None = None,
        language: str = "python",
    ) -> RefactorResult:
        """Simulate applying a patch and check for safety.

        Provide either a unified diff patch OR the new code directly.

        Args:
            original_code: The original source code
            patch: Unified diff patch to apply (optional)
            new_code: New code to compare against (optional)
            language: Source language ("python", "javascript", "java")

        Returns:
            RefactorResult with safety verdict and details

        Raises:
            ValueError: If neither patch nor new_code provided
        """
        if patch is None and new_code is None:
            raise ValueError("Must provide either 'patch' or 'new_code'")

        # Apply patch to get new code
        if new_code is None:
            try:
                new_code = self._apply_patch(original_code, patch)
            except Exception as e:
                return RefactorResult(
                    status=RefactorStatus.ERROR,
                    is_safe=False,
                    patched_code="",
                    reason=f"Failed to apply patch: {str(e)}",
                )

        # Validate syntax
        syntax_error = self._check_syntax(new_code, language)
        if syntax_error:
            return RefactorResult(
                status=RefactorStatus.ERROR,
                is_safe=False,
                patched_code=new_code,
                reason=f"Syntax error in patched code: {syntax_error}",
            )

        # Run security scan
        security_issues = self._scan_security(new_code, original_code, language)

        # Check structural changes
        structural_changes = self._analyze_structural_changes(
            original_code, new_code, language
        )

        # Generate warnings
        warnings = self._generate_warnings(structural_changes)

        # Determine verdict
        status, is_safe, reason = self._determine_verdict(
            security_issues, structural_changes, warnings
        )

        return RefactorResult(
            status=status,
            is_safe=is_safe,
            patched_code=new_code,
            reason=reason,
            security_issues=security_issues,
            structural_changes=structural_changes,
            warnings=warnings,
        )

    def simulate_inline(
        self,
        original_code: str,
        new_code: str,
        language: str = "python",
    ) -> RefactorResult:
        """Simulate by comparing original and new code directly.

        This is a convenience method when you have both versions.

        Args:
            original_code: Original source code
            new_code: New/modified source code
            language: Source language

        Returns:
            RefactorResult with safety analysis
        """
        return self.simulate(
            original_code=original_code,
            new_code=new_code,
            language=language,
        )

    def _apply_patch(self, original: str, patch: str) -> str:
        """Apply a unified diff patch to the original code."""
        # Parse the patch
        original_lines = original.splitlines(keepends=True)
        patched_lines = list(original_lines)

        # Simple unified diff parser
        hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

        lines = patch.splitlines(keepends=True)
        i = 0
        offset = 0

        while i < len(lines):
            line = lines[i]

            # Find hunk header
            match = hunk_pattern.match(line)
            if match:
                old_start = int(match.group(1)) - 1  # 0-indexed
                new_start = int(match.group(3)) - 1
                i += 1

                # Process hunk
                current_old = old_start + offset
                deletions = []
                additions = []

                while i < len(lines) and not lines[i].startswith("@@"):
                    hunk_line = lines[i]
                    if hunk_line.startswith("-") and not hunk_line.startswith("---"):
                        deletions.append(current_old)
                        current_old += 1
                    elif hunk_line.startswith("+") and not hunk_line.startswith("+++"):
                        additions.append(hunk_line[1:])
                    elif hunk_line.startswith(" ") or hunk_line == "\n":
                        current_old += 1
                    i += 1

                # Apply deletions (in reverse to preserve indices)
                for del_idx in reversed(deletions):
                    if del_idx < len(patched_lines):
                        patched_lines.pop(del_idx)

                # Apply additions
                insert_point = old_start + offset
                for j, add_line in enumerate(additions):
                    patched_lines.insert(insert_point + j, add_line)

                offset += len(additions) - len(deletions)
            else:
                i += 1

        return "".join(patched_lines)

    def _check_syntax(self, code: str, language: str) -> str | None:
        """Check if code has valid syntax. Returns error message or None."""
        if language == "python":
            try:
                ast.parse(code)
                return None
            except SyntaxError as e:
                return f"Line {e.lineno}: {e.msg}"
        # For other languages, we'd need tree-sitter
        return None

    def _scan_security(
        self, new_code: str, original_code: str, language: str
    ) -> list[SecurityIssue]:
        """Scan for security vulnerabilities in the new code."""
        issues = []

        # Try using the SecurityAnalyzer
        try:
            from code_scalpel.security import SecurityAnalyzer

            analyzer = SecurityAnalyzer()

            # Scan new code
            new_result = analyzer.analyze(new_code)
            old_result = analyzer.analyze(original_code)

            # Find NEW vulnerabilities (not in original)
            old_vulns = {
                (v.get("type"), v.get("line"))
                for v in old_result.get("vulnerabilities", [])
            }

            for vuln in new_result.get("vulnerabilities", []):
                vuln_key = (vuln.get("type"), vuln.get("line"))
                if vuln_key not in old_vulns:
                    issues.append(SecurityIssue(
                        type=vuln.get("type", "Unknown"),
                        severity=vuln.get("severity", "medium"),
                        line=vuln.get("line"),
                        description=vuln.get("description", "Security vulnerability"),
                        cwe=vuln.get("cwe"),
                    ))

        except ImportError:
            # Fallback to pattern-based detection
            issues.extend(self._pattern_security_scan(new_code, original_code))

        return issues

    def _pattern_security_scan(
        self, new_code: str, original_code: str
    ) -> list[SecurityIssue]:
        """Pattern-based security scan fallback."""
        issues = []

        dangerous_patterns = [
            ("eval(", "Code Injection", "CWE-94", "high", "eval() can execute arbitrary code"),
            ("exec(", "Code Injection", "CWE-94", "high", "exec() can execute arbitrary code"),
            ("os.system(", "Command Injection", "CWE-78", "high", "os.system() can execute shell commands"),
            ("subprocess.call(", "Command Injection", "CWE-78", "medium", "subprocess with shell=True is dangerous"),
            ("cursor.execute(", "SQL Injection", "CWE-89", "high", "SQL query may be injectable"),
            ("render_template_string(", "XSS", "CWE-79", "medium", "Template injection risk"),
            ("pickle.loads(", "Deserialization", "CWE-502", "high", "Insecure deserialization"),
            ("yaml.load(", "Deserialization", "CWE-502", "medium", "yaml.load() is unsafe, use safe_load()"),
        ]

        for line_num, line in enumerate(new_code.splitlines(), 1):
            for pattern, vuln_type, cwe, severity, desc in dangerous_patterns:
                if pattern in line and pattern not in original_code:
                    issues.append(SecurityIssue(
                        type=vuln_type,
                        severity=severity,
                        line=line_num,
                        description=f"Introduced {desc}",
                        cwe=cwe,
                    ))

        return issues

    def _analyze_structural_changes(
        self, original: str, new_code: str, language: str
    ) -> dict[str, Any]:
        """Analyze structural changes between versions."""
        changes = {
            "functions_added": [],
            "functions_removed": [],
            "functions_modified": [],
            "classes_added": [],
            "classes_removed": [],
            "imports_added": [],
            "imports_removed": [],
            "lines_added": 0,
            "lines_removed": 0,
        }

        if language == "python":
            try:
                old_tree = ast.parse(original)
                new_tree = ast.parse(new_code)

                old_funcs = {n.name for n in ast.walk(old_tree) if isinstance(n, ast.FunctionDef)}
                new_funcs = {n.name for n in ast.walk(new_tree) if isinstance(n, ast.FunctionDef)}

                old_classes = {n.name for n in ast.walk(old_tree) if isinstance(n, ast.ClassDef)}
                new_classes = {n.name for n in ast.walk(new_tree) if isinstance(n, ast.ClassDef)}

                old_imports = set()
                new_imports = set()

                for node in ast.walk(old_tree):
                    if isinstance(node, ast.Import):
                        old_imports.update(a.name for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        old_imports.update(f"{node.module}.{a.name}" for a in node.names)

                for node in ast.walk(new_tree):
                    if isinstance(node, ast.Import):
                        new_imports.update(a.name for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        new_imports.update(f"{node.module}.{a.name}" for a in node.names)

                changes["functions_added"] = list(new_funcs - old_funcs)
                changes["functions_removed"] = list(old_funcs - new_funcs)
                changes["classes_added"] = list(new_classes - old_classes)
                changes["classes_removed"] = list(old_classes - new_classes)
                changes["imports_added"] = list(new_imports - old_imports)
                changes["imports_removed"] = list(old_imports - new_imports)

            except SyntaxError:
                pass

        # Count line changes
        diff = list(difflib.unified_diff(
            original.splitlines(),
            new_code.splitlines(),
            lineterm=""
        ))

        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                changes["lines_added"] += 1
            elif line.startswith("-") and not line.startswith("---"):
                changes["lines_removed"] += 1

        return changes

    def _generate_warnings(self, structural_changes: dict[str, Any]) -> list[str]:
        """Generate warnings based on structural changes."""
        warnings = []

        if structural_changes.get("functions_removed"):
            funcs = ", ".join(structural_changes["functions_removed"])
            warnings.append(f"Removed functions: {funcs}")

        if structural_changes.get("classes_removed"):
            classes = ", ".join(structural_changes["classes_removed"])
            warnings.append(f"Removed classes: {classes}")

        lines_added = structural_changes.get("lines_added", 0)
        lines_removed = structural_changes.get("lines_removed", 0)

        if lines_removed > lines_added * 2:
            warnings.append(f"Large deletion: {lines_removed} lines removed vs {lines_added} added")

        if lines_added > 100:
            warnings.append(f"Large addition: {lines_added} new lines")

        return warnings

    def _determine_verdict(
        self,
        security_issues: list[SecurityIssue],
        structural_changes: dict[str, Any],
        warnings: list[str],
    ) -> tuple[RefactorStatus, bool, str | None]:
        """Determine the final verdict."""
        # High severity security issues = UNSAFE
        high_severity = [i for i in security_issues if i.severity == "high"]
        if high_severity:
            issue = high_severity[0]
            return (
                RefactorStatus.UNSAFE,
                False,
                f"Introduced {issue.type} vulnerability: {issue.description}",
            )

        # Medium severity = UNSAFE in strict mode, WARNING otherwise
        medium_severity = [i for i in security_issues if i.severity == "medium"]
        if medium_severity:
            issue = medium_severity[0]
            if self.strict_mode:
                return (
                    RefactorStatus.UNSAFE,
                    False,
                    f"Introduced {issue.type} vulnerability: {issue.description}",
                )
            return (
                RefactorStatus.WARNING,
                True,  # Still safe but with warning
                f"Potential {issue.type} issue: {issue.description}",
            )

        # Warnings without security issues
        if warnings and self.strict_mode:
            return (
                RefactorStatus.WARNING,
                True,
                warnings[0],
            )

        # All clear
        return (RefactorStatus.SAFE, True, None)
