"""
Test Suite for New Vulnerability Detection: WEAK_CRYPTO and SSRF

These tests verify that the security analyzer correctly detects:
1. Weak Cryptography (CWE-327): MD5, SHA1, DES usage with tainted data
2. SSRF (CWE-918): Server-Side Request Forgery with user-controlled URLs

Each test follows the pattern:
- Create code with tainted input flowing to vulnerable sink
- Run security analyzer
- Verify vulnerability is detected with correct type
"""


class TestWeakCryptographyDetection:
    """Test detection of weak cryptographic algorithms (CWE-327)."""

    def test_md5_with_user_input(self):
        """Detect MD5 hash of user-controlled data."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import hashlib
user_data = input("Enter data to hash: ")
digest = hashlib.md5(user_data.encode()).hexdigest()
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        vuln_types = [v.vulnerability_type for v in result.vulnerabilities]
        assert any(
            "Weak" in str(vt) or "Crypto" in str(vt) or "WEAK_CRYPTO" in str(vt)
            for vt in vuln_types
        )

    def test_sha1_with_request_data(self):
        """Detect SHA1 hash of request data."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import hashlib
password = request.form.get("password")
hashed = hashlib.sha1(password.encode()).hexdigest()
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_des_cipher_with_tainted_key(self):
        """Detect DES encryption with tainted key."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from cryptography.hazmat.primitives.ciphers.algorithms import DES
key = request.args.get("key")
cipher = DES(key.encode())
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_pycryptodome_md5(self):
        """Detect PyCryptodome MD5 usage."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from Crypto.Hash import MD5
user_input = input()
h = MD5.new(user_input.encode())
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_safe_sha256_no_vulnerability(self):
        """SHA-256 should NOT be flagged as weak crypto."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import hashlib
user_data = input("Enter data: ")
digest = hashlib.sha256(user_data.encode()).hexdigest()
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        # SHA-256 is NOT weak crypto - should not flag WEAK_CRYPTO
        weak_crypto_vulns = [
            v
            for v in result.vulnerabilities
            if "WEAK_CRYPTO" in str(v.vulnerability_type)
        ]
        assert len(weak_crypto_vulns) == 0


class TestSSRFDetection:
    """Test detection of Server-Side Request Forgery (CWE-918)."""

    def test_requests_get_with_user_url(self):
        """Detect SSRF via requests.get with user-controlled URL."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import requests
url = request.args.get("url")
response = requests.get(url)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        vuln_types = [v.vulnerability_type for v in result.vulnerabilities]
        assert any("SSRF" in str(vt) or "CWE-918" in str(vt) for vt in vuln_types)

    def test_requests_post_with_user_url(self):
        """Detect SSRF via requests.post."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import requests
target = request.form.get("target")
requests.post(target, data={"key": "value"})
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_urllib_urlopen_with_tainted_url(self):
        """Detect SSRF via urllib.request.urlopen."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
from urllib.request import urlopen
user_url = input("Enter URL: ")
response = urlopen(user_url)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_httpx_get_with_user_controlled_url(self):
        """Detect SSRF via httpx.get."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import httpx
endpoint = request.args.get("endpoint")
r = httpx.get(endpoint)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_url_concatenation_ssrf(self):
        """Detect SSRF with URL constructed from user input."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import requests
base_url = "https://api.example.com/"
user_path = request.args.get("path")
full_url = base_url + user_path
response = requests.get(full_url)
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities

    def test_safe_hardcoded_url_no_ssrf(self):
        """Hardcoded URL should NOT be flagged as SSRF."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import requests
response = requests.get("https://api.example.com/status")
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        # No tainted data = no SSRF
        ssrf_vulns = [
            v for v in result.vulnerabilities if "SSRF" in str(v.vulnerability_type)
        ]
        assert len(ssrf_vulns) == 0


class TestCombinedVulnerabilities:
    """Test detection of multiple vulnerability types in same code."""

    def test_ssrf_and_weak_crypto_together(self):
        """Detect both SSRF and weak crypto in the same code."""
        from code_scalpel.symbolic_execution_tools.security_analyzer import (
            SecurityAnalyzer,
        )

        code = """
import requests
import hashlib

user_url = request.args.get("url")
response = requests.get(user_url)  # SSRF

user_data = request.form.get("data")
digest = hashlib.md5(user_data.encode())  # Weak Crypto
"""

        analyzer = SecurityAnalyzer()
        result = analyzer.analyze(code)

        assert result.has_vulnerabilities
        # Should have at least 2 vulnerabilities
        assert len(result.vulnerabilities) >= 2


class TestTaintTrackerIntegration:
    """Test that TaintTracker correctly handles new sink types."""

    def test_taint_is_dangerous_for_weak_crypto(self):
        """Verify taint is marked dangerous for WEAK_CRYPTO sink."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            TaintInfo,
            TaintSource,
            TaintLevel,
            SecuritySink,
        )

        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH,
            source_location=(1, 0),
            propagation_path=[],
        )

        assert taint.is_dangerous_for(SecuritySink.WEAK_CRYPTO)

    def test_taint_is_dangerous_for_ssrf(self):
        """Verify taint is marked dangerous for SSRF sink."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            TaintInfo,
            TaintSource,
            TaintLevel,
            SecuritySink,
        )

        taint = TaintInfo(
            source=TaintSource.USER_INPUT,
            level=TaintLevel.HIGH,
            source_location=(1, 0),
            propagation_path=[],
        )

        assert taint.is_dangerous_for(SecuritySink.SSRF)

    def test_sink_patterns_registered(self):
        """Verify all new sink patterns are registered."""
        from code_scalpel.symbolic_execution_tools.taint_tracker import (
            SecuritySink,
            SINK_PATTERNS,
        )

        # Weak crypto patterns
        assert SINK_PATTERNS.get("hashlib.md5") == SecuritySink.WEAK_CRYPTO
        assert SINK_PATTERNS.get("hashlib.sha1") == SecuritySink.WEAK_CRYPTO

        # SSRF patterns
        assert SINK_PATTERNS.get("requests.get") == SecuritySink.SSRF
        assert SINK_PATTERNS.get("requests.post") == SecuritySink.SSRF
        assert SINK_PATTERNS.get("urllib.request.urlopen") == SecuritySink.SSRF
        assert SINK_PATTERNS.get("httpx.get") == SecuritySink.SSRF
