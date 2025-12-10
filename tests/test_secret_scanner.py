import pytest
from src.code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
from src.code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink

def test_aws_key_detection():
    analyzer = SecurityAnalyzer()
    code = """
def connect_aws():
    access_key = "AKIAIOSFODNN7EXAMPLE"
    secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
    client = boto3.client('s3', aws_access_key_id=access_key)
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    assert len(vulns) >= 1
    assert "AWS Access Key" in vulns[0].taint_path

def test_stripe_secret_detection():
    analyzer = SecurityAnalyzer()
    code = """
stripe.api_key = "sk_live_51Hs1234567890abcdefghij"
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    assert len(vulns) == 1
    assert "Stripe Secret" in vulns[0].taint_path

def test_private_key_detection():
    analyzer = SecurityAnalyzer()
    code = 'key = "-----BEGIN RSA PRIVATE KEY-----"'
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    assert len(vulns) == 1
    assert "Private Key" in vulns[0].taint_path

def test_generic_api_key_detection():
    analyzer = SecurityAnalyzer()
    code = """
api_key = "abcdefghijklmnopqrstuvwxyz123456"
access_token = "789012345678901234567890123456"
"""
    result = analyzer.analyze(code)
    assert result.has_vulnerabilities
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    assert len(vulns) == 2
    # Order is not guaranteed, check existence
    paths = [v.taint_path[0] for v in vulns]
    assert "Generic API Key" in paths

def test_generic_api_key_false_positive():
    analyzer = SecurityAnalyzer()
    code = """
api_key = "short"
other_var = "abcdefghijklmnopqrstuvwxyz123456"
"""
    result = analyzer.analyze(code)
    # Should be 0 because api_key is too short, and other_var is not a target key
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    assert len(vulns) == 0

def test_mixed_vulnerabilities():
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
    
    secrets = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    sqli = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.SQL_QUERY]
    
    assert len(secrets) == 1
    assert len(sqli) == 1

def test_bytes_secret_detection():
    analyzer = SecurityAnalyzer()
    code = """
def connect_aws():
    access_key = b"AKIAIOSFODNN7EXAMPLE"
    client = boto3.client('s3', aws_access_key_id=access_key)
"""
    result = analyzer.analyze(code)
    # Currently expected to fail if bytes are not handled
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    # We want this to pass eventually
    assert len(vulns) >= 1
    assert "AWS Access Key" in vulns[0].taint_path

def test_fstring_secret_detection():
    analyzer = SecurityAnalyzer()
    # If the secret is split, it might not be detected.
    # But if it's a full string inside an f-string part, it might be.
    # Actually, f"prefix{var}" is parsed as JoinedStr(values=[Constant("prefix"), FormattedValue(var)])
    # So "prefix" is a Constant.
    # If the secret is "AKIA..." and it's a literal part of f-string:
    code = """
msg = f"Your key is AKIAIOSFODNN7EXAMPLE for now"
"""
    result = analyzer.analyze(code)
    vulns = [v for v in result.vulnerabilities if v.sink_type == SecuritySink.HARDCODED_SECRET]
    assert len(vulns) >= 1
    assert "AWS Access Key" in vulns[0].taint_path

def test_generic_api_key_tuple_unpacking():
    analyzer = SecurityAnalyzer()
    # This is a known limitation, but let's see if it crashes or just misses it.
    code = """
api_key, other = "abcdefghijklmnopqrstuvwxyz123456", "val"
"""
    result = analyzer.analyze(code)
    # It will likely miss it because node.value is a Tuple, not a Constant string.
    # We just want to ensure no crash.
    assert result is not None
