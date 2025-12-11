"""
Secret Scanner Tests - Using Safe Test Fixtures

These tests use obviously fake/example credentials that are documented as safe:
- AWS EXAMPLE keys (from AWS documentation)
- Clearly fake patterns that won't trigger secret scanners
"""

from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink


def test_aws_key_detection():
    """Test detection of AWS access keys using AWS's official EXAMPLE keys."""
    analyzer = SecurityAnalyzer()
    # These are AWS's documented example keys - safe to use in tests
    code = """
def connect_aws():
    access_key = "AKIAIOSFODNN7EXAMPLE"
    secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    client = boto3.client('s3', aws_access_key_id=access_key)
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) >= 1
    assert "AWS Access Key" in vulns[0].taint_path


def test_stripe_secret_detection():
    """Test detection of Stripe-like keys.
    
    Note: We test the pattern matching directly rather than using a real-looking
    key to avoid triggering GitHub's secret scanner.
    """
    import re
    from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
    
    scanner = SecretScanner()
    
    # Find the Stripe pattern
    stripe_pattern = None
    for name, pattern in scanner.string_patterns:
        if "Stripe" in name:
            stripe_pattern = pattern
            break
    
    assert stripe_pattern is not None, "Stripe pattern should exist"
    
    # Test that the pattern matches correctly formatted keys
    # The pattern is: sk_live_[0-9a-zA-Z]{24}
    test_key = "sk_" + "live_" + "a" * 24  # Build key in parts to avoid scanner
    assert stripe_pattern.search(test_key), f"Pattern should match: {test_key}"
    
    # Test that it doesn't match incorrect formats
    assert not stripe_pattern.search("sk_live_tooshort")
    assert not stripe_pattern.search("sk_test_" + "a" * 24)  # test, not live


def test_private_key_detection():
    """Test detection of private key headers."""
    analyzer = SecurityAnalyzer()
    code = 'key = "-----BEGIN RSA PRIVATE KEY-----"'
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) == 1
    assert "Private Key" in vulns[0].taint_path


def test_generic_api_key_detection():
    """Test detection of generic API key patterns."""
    analyzer = SecurityAnalyzer()
    code = """
api_key = "abcdefghijklmnopqrstuvwxyz123456"
access_token = "789012345678901234567890123456"
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) == 2
    # Order is not guaranteed, check existence
    paths = [v.taint_path[0] for v in vulns]
    assert "Generic API Key" in paths


def test_generic_api_key_false_positive():
    """Test that short strings don't trigger false positives."""
    analyzer = SecurityAnalyzer()
    code = """
api_key = "short"
other_var = "abcdefghijklmnopqrstuvwxyz123456"
"""
    result = analyzer.analyze(code)
    # Should be 0 because api_key is too short, and other_var is not a target key
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) == 0


def test_mixed_vulnerabilities():
    """Test detection of mixed vulnerability types."""
    analyzer = SecurityAnalyzer()
    code = """
def unsafe():
    key = "AKIAIOSFODNN7EXAMPLE"
    user_id = request.args.get("id")
    query = "SELECT * FROM users WHERE id=" + user_id
    cursor.execute(query)
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities

    secrets = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    sqli = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.SQL_QUERY]

    assert len(secrets) == 1
    assert len(sqli) == 1


def test_bytes_secret_detection():
    """Test detection of secrets in bytes literals."""
    analyzer = SecurityAnalyzer()
    code = """
def connect_aws():
    access_key = b"AKIAIOSFODNN7EXAMPLE"
    client = boto3.client('s3', aws_access_key_id=access_key)
"""
    result = analyzer.analyze(code)
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) >= 1
    assert "AWS Access Key" in vulns[0].taint_path


def test_fstring_secret_detection():
    """Test detection of secrets embedded in f-strings."""
    analyzer = SecurityAnalyzer()
    code = """
msg = f"Your key is AKIAIOSFODNN7EXAMPLE for now"
"""
    result = analyzer.analyze(code)
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) >= 1
    assert "AWS Access Key" in vulns[0].taint_path


def test_generic_api_key_tuple_unpacking():
    """Test handling of tuple unpacking (known limitation)."""
    analyzer = SecurityAnalyzer()
    code = """
api_key, other = "abcdefghijklmnopqrstuvwxyz123456", "val"
"""
    result = analyzer.analyze(code)
    # It will likely miss it because node.value is a Tuple, not a Constant string.
    # We just want to ensure no crash.
    assert result is not None  # Just ensure no exception
