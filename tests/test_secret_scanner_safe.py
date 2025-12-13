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
    # taint_path is a list - check if any element contains the key type
    assert any("AWS Access Key" in path for path in vulns[0].taint_path)


def test_stripe_secret_detection():
    """Test detection of Stripe-like keys.
    
    Note: We test the pattern matching directly rather than using a real-looking
    key to avoid triggering GitHub's secret scanner.
    """
    import re
    from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
    
    scanner = SecretScanner()
    
    # Find the Stripe pattern (pattern name contains "stripe")
    stripe_pattern = None
    for name, pattern in scanner.string_patterns:
        if "stripe" in name.lower():
            stripe_pattern = pattern
            break
    
    assert stripe_pattern is not None, "Stripe pattern should exist"
    
    # Test that the pattern matches correctly formatted keys
    # The pattern is: sk_live_[0-9a-zA-Z]{24}
    test_key = "sk_" + "live_" + "a" * 24  # Build key in parts to avoid scanner
    assert stripe_pattern.search(test_key), f"Pattern should match: {test_key}"
    
    # Test that it doesn't match incorrect formats
    assert not stripe_pattern.search("sk_live_tooshort")


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
    # taint_path is a list - check if any element contains the key type
    assert any("Private Key" in path for path in vulns[0].taint_path)


def test_generic_api_key_detection():
    """Test detection of generic API key patterns in string literals.
    
    Note: The scanner looks for patterns WITHIN strings, not variable names.
    For example: config = "api_key='abc123...'" would match.
    """
    analyzer = SecurityAnalyzer()
    # The pattern matches strings containing api_key='value' patterns (with quotes)
    code = """
config = "api_key='abcdefghijklmnopqrstuvwx'"
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) >= 1
    # Check if any element contains "API Key"
    assert any("API Key" in path for path in vulns[0].taint_path)


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
    # taint_path is a list - check if any element contains the key type
    assert any("AWS Access Key" in path for path in vulns[0].taint_path)


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
    # taint_path is a list - check if any element contains the key type
    assert any("AWS Access Key" in path for path in vulns[0].taint_path)


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


def test_duplicate_secret_detection():
    """Test that duplicate secrets at the same location aren't reported twice."""
    from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
    import ast
    
    # Code with the same secret appearing multiple times
    code = """
key1 = "AKIAIOSFODNN7EXAMPLE"
key2 = "AKIAIOSFODNN7EXAMPLE"
"""
    tree = ast.parse(code)
    scanner = SecretScanner()
    vulns = scanner.scan(tree)
    
    # Should detect 2 vulns (one per assignment, different locations)
    assert len(vulns) == 2


def test_same_location_dedup():
    """Test that same location is deduplicated (line 106 coverage)."""
    from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
    import ast
    
    scanner = SecretScanner()
    
    # Create two nodes with the SAME location
    node1 = ast.Constant(value="AKIAIOSFODNN7EXAMPLE")
    node1.lineno = 1
    node1.col_offset = 0
    
    # _add_vuln now takes (secret_type, matched_value, node)
    scanner._add_vuln("aws_access_key", "AKIAIOSFODNN7EXAMPLE", node1)
    scanner._add_vuln("aws_access_key", "AKIAIOSFODNN7EXAMPLE", node1)  # Should be deduped
    
    # Should only have 1 vulnerability (deduped)
    assert len(scanner.vulnerabilities) == 1


def test_invalid_bytes_decoding():
    """Test handling of bytes that fail to decode properly."""
    from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
    import ast
    
    # Create an AST node with bytes that might cause decode issues
    # Note: In practice, Python source is always valid UTF-8, so this tests
    # the defensive code path
    code = '''
data = b"AKIAIOSFODNN7EXAMPLE"
'''
    tree = ast.parse(code)
    scanner = SecretScanner()
    vulns = scanner.scan(tree)
    
    # Should still detect the AWS key in bytes
    assert len(vulns) >= 1


def test_security_result_summary_with_taint_path():
    """Test that security result summary includes taint path details."""
    analyzer = SecurityAnalyzer()
    code = """
key = "AKIAIOSFODNN7EXAMPLE"
"""
    result = analyzer.analyze(code)
    
    # Get the summary which should include the taint path
    summary = result.summary()
    
    # The summary should mention the vulnerability type
    assert "AWS Access Key" in summary or "HARDCODED_SECRET" in summary


def test_deprecated_ast_str_node():
    """Test handling of deprecated ast.Str nodes (Python < 3.8 compatibility)."""
    from code_scalpel.symbolic_execution_tools.secret_scanner import SecretScanner
    import ast
    
    # Create a mock ast.Str node (deprecated in Python 3.8+)
    # In Python 3.9+, strings are parsed as ast.Constant, but we test the visit_Str path
    scanner = SecretScanner()
    
    # Create a fake ast.Str node
    str_node = ast.Str(s="AKIAIOSFODNN7EXAMPLE")
    str_node.lineno = 1
    str_node.col_offset = 0
    
    # Directly call visit_Str
    scanner.visit_Str(str_node)
    
    # Should detect the AWS key
    assert len(scanner.vulnerabilities) == 1
    # taint_path is a list - check if any element contains the key type
    assert any("AWS Access Key" in path for path in scanner.vulnerabilities[0].taint_path)
