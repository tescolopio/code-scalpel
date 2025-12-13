from code_scalpel.symbolic_execution_tools.security_analyzer import SecurityAnalyzer
from code_scalpel.symbolic_execution_tools.taint_tracker import SecuritySink


def test_bytes_secret_detection():
    analyzer = SecurityAnalyzer()
    code = """
def connect_aws():
    access_key = b"AKIAIOSFODNN7EXAMPLE"
    client = boto3.client('s3', aws_access_key_id=access_key)
"""
    result = analyzer.analyze(code)
    # Currently expected to fail if bytes are not handled
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    # We want this to pass eventually
    assert len(vulns) >= 1
    # taint_path is a list - check if any element contains the key type
    assert any("AWS Access Key" in path for path in vulns[0].taint_path)


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
    vulns = [
        v
        for v in result.vulnerabilities
        if v.sink_type == SecuritySink.HARDCODED_SECRET
    ]
    assert len(vulns) >= 1
    # taint_path is a list - check if any element contains the key type
    assert any("AWS Access Key" in path for path in vulns[0].taint_path)


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
