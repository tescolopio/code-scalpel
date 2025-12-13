"""
Secret Scanner - Hardcoded Secret Detection (v1.3.0 Enhanced).

Detects hardcoded secrets in Python code using comprehensive regex patterns:
- AWS keys (access keys, secret keys)
- GitHub tokens (PAT, OAuth, App tokens)
- Stripe keys (live, test, restricted)
- Slack tokens and webhooks
- Google API keys
- Firebase credentials
- Twilio SID/tokens
- SendGrid API keys
- Private keys (RSA, EC, DSA, OpenSSH)
- Generic API keys and secrets
"""

import ast
import re
from typing import Dict, List, Pattern

from .taint_tracker import (
    HARDCODED_SECRET_PATTERNS,
    Vulnerability,
    SecuritySink,
    TaintSource,
)


class SecretScanner(ast.NodeVisitor):
    """
    Scans AST for hardcoded secrets using comprehensive pattern matching.
    """

    def __init__(self) -> None:
        self.vulnerabilities: List[Vulnerability] = []

        # Compile all patterns from HARDCODED_SECRET_PATTERNS
        self.compiled_patterns: Dict[str, Pattern] = {
            name: re.compile(pattern)
            for name, pattern in HARDCODED_SECRET_PATTERNS.items()
        }

        # Placeholder patterns to ignore (not real secrets)
        # These match whole words/patterns, not substrings within keys
        self.placeholder_patterns = [
            r"^your[-_]?api[-_]?key[-_]?here$",
            r"^your[-_]?secret[-_]?here$",
            r"^replace[-_]?me$",
            r"^example[-_]?key$",
            r"^example[-_]?secret$",
            r"^test[-_]?key$",
            r"^dummy[-_]?",
            r"^placeholder",
            r"^xxxx+$",
            r"^\*+$",
            r"^todo$",
            r"^changeme$",
            r"^insert[-_]?",
            r"^fake[-_]?",
        ]
        self._compiled_placeholders = [
            re.compile(p, re.IGNORECASE) for p in self.placeholder_patterns
        ]

    @property
    def string_patterns(self) -> List[tuple[str, Pattern[str]]]:
        """
        Backward compatibility property for tests.
        
        Returns:
            List of (name, pattern) tuples
        """
        return [(name, pattern) for name, pattern in self.compiled_patterns.items()]

    def scan(self, tree: ast.AST) -> List[Vulnerability]:
        """
        Scan an AST for hardcoded secrets.

        Args:
            tree: The AST to scan

        Returns:
            List of detected vulnerabilities
        """
        self.vulnerabilities = []
        self.visit(tree)
        return self.vulnerabilities

    def visit_Constant(self, node: ast.Constant) -> None:
        """Check string literals against all secret patterns."""
        value = node.value
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")

        if isinstance(value, str):
            self._check_string_for_secrets(value, node)

        self.generic_visit(node)

    def visit_Str(self, node: ast.Str) -> None:
        """Support for deprecated ast.Str nodes (Python < 3.8)."""
        self._check_string_for_secrets(node.s, node)
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """Check f-strings for hardcoded secrets."""
        # Concatenate all constant parts
        full_string = ""
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                full_string += value.value

        if full_string:
            self._check_string_for_secrets(full_string, node)

        self.generic_visit(node)

    def _is_placeholder(self, value: str) -> bool:
        """
        Check if a string is a placeholder value that should be ignored.

        Args:
            value: String value to check

        Returns:
            True if the string appears to be a placeholder
        """
        # Check against compiled placeholder patterns
        for pattern in self._compiled_placeholders:
            if pattern.search(value):
                return True
        return False

    def _check_string_for_secrets(self, string_value: str, node: ast.AST) -> None:
        """
        Check a string for secret patterns.

        Args:
            string_value: String to check
            node: AST node for location info
        """
        # Skip obvious placeholders (check the whole string as a placeholder)
        if self._is_placeholder(string_value):
            return

        # Check each pattern
        for secret_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(string_value)
            for match in matches:
                self._add_vuln(secret_type, match.group(), node)

    def _add_vuln(self, secret_type: str, matched_value: str, node: ast.AST) -> None:
        """
        Add a vulnerability to the list.

        Args:
            secret_type: Type of secret detected (e.g., "aws_access_key")
            matched_value: The actual matched string (masked in output)
            node: AST node where the secret was found
        """
        loc = (node.lineno, node.col_offset) if hasattr(node, "lineno") else (0, 0)

        # Check for duplicates
        for v in self.vulnerabilities:
            if v.sink_location == loc and v.sink_type == SecuritySink.HARDCODED_SECRET:
                return

        # Mask the secret in the description
        masked = self._mask_secret(matched_value)
        
        # Get human-readable name and recommendation
        display_name, recommendation = self._get_recommendation(secret_type)

        vuln = Vulnerability(
            sink_type=SecuritySink.HARDCODED_SECRET,
            taint_source=TaintSource.HARDCODED,
            taint_path=[f"{display_name}: {masked}"],
            sink_location=loc,
            source_location=loc,
            sanitizers_applied=set(),
        )
        
        # Add recommendation to the vulnerability (stored in taint_path for now)
        vuln.taint_path.append(f"Recommendation: {recommendation}")

        self.vulnerabilities.append(vuln)

    def _mask_secret(self, value: str, show_chars: int = 4) -> str:
        """
        Mask a secret value, showing only the first few characters.

        Args:
            value: Secret value to mask
            show_chars: Number of characters to show (default 4)

        Returns:
            Masked string (e.g., "AKIA********")
        """
        if len(value) <= show_chars:
            return "*" * len(value)
        return value[:show_chars] + "*" * (len(value) - show_chars)

    def _get_recommendation(self, secret_type: str) -> tuple[str, str]:
        """
        Get display name and recommendation for a secret type.

        Args:
            secret_type: Internal secret type identifier

        Returns:
            Tuple of (display_name, recommendation)
        """
        recommendations = {
            "aws_access_key": (
                "AWS Access Key",
                "Use AWS IAM roles or environment variables with AWS Secrets Manager"
            ),
            "aws_secret_key": (
                "AWS Secret Key",
                "Use AWS IAM roles or environment variables with AWS Secrets Manager"
            ),
            "github_token": (
                "GitHub Personal Access Token",
                "Use GitHub Actions secrets or environment variables"
            ),
            "github_oauth": (
                "GitHub OAuth Token",
                "Use GitHub App authentication or OAuth flows with secure token storage"
            ),
            "github_app": (
                "GitHub App Token",
                "Store GitHub App private keys in secure key management systems"
            ),
            "github_fine_grained": (
                "GitHub Fine-Grained Token",
                "Use GitHub Actions secrets or environment variables with minimal permissions"
            ),
            "gitlab_token": (
                "GitLab Personal Access Token",
                "Use GitLab CI/CD variables or environment variables"
            ),
            "stripe_live": (
                "Stripe Live Key",
                "Use environment variables and never commit live keys to version control"
            ),
            "stripe_test": (
                "Stripe Test Key",
                "Use environment variables even for test keys to maintain security practices"
            ),
            "slack_token": (
                "Slack Bot Token",
                "Use environment variables or secure secret management"
            ),
            "slack_webhook": (
                "Slack Webhook URL",
                "Store webhook URLs in environment variables or secret management systems"
            ),
            "google_api": (
                "Google API Key",
                "Use Google Cloud Secret Manager or environment variables"
            ),
            "firebase": (
                "Firebase Key",
                "Use Firebase environment config and restrict API key permissions"
            ),
            "twilio_sid": (
                "Twilio Account SID",
                "Use environment variables for Twilio credentials"
            ),
            "twilio_token": (
                "Twilio Auth Token",
                "Use environment variables and rotate tokens regularly"
            ),
            "sendgrid": (
                "SendGrid API Key",
                "Use environment variables and restrict API key permissions"
            ),
            "mailgun": (
                "Mailgun API Key",
                "Use environment variables and secure key storage"
            ),
            "square_token": (
                "Square Access Token",
                "Use environment variables and OAuth for production applications"
            ),
            "private_key_rsa": (
                "RSA Private Key",
                "Store private keys in secure key management systems, never in code"
            ),
            "private_key_ec": (
                "EC Private Key",
                "Store private keys in secure key management systems, never in code"
            ),
            "private_key_dsa": (
                "DSA Private Key",
                "Store private keys in secure key management systems, never in code"
            ),
            "private_key_openssh": (
                "OpenSSH Private Key",
                "Store SSH keys in ~/.ssh with proper permissions (600), never in code"
            ),
            "private_key_generic": (
                "Private Key",
                "Store private keys in secure key management systems, never in code"
            ),
            "generic_api_key": (
                "Generic API Key",
                "Use environment variables or secure secret management systems"
            ),
            "generic_secret": (
                "Generic Secret",
                "Use environment variables or secure secret management systems"
            ),
        }
        
        return recommendations.get(
            secret_type,
            ("Hardcoded Secret", "Use environment variables or secure secret management")
        )
