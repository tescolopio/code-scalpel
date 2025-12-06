"""
Security Analysis Example - Code Scalpel v0.3.0

This example demonstrates how to use the SecurityAnalyzer to detect
common vulnerabilities in Python code:
- SQL Injection (CWE-89)
- Cross-Site Scripting (CWE-79)
- Command Injection (CWE-78)
- Path Traversal (CWE-22)

Usage:
    python security_analysis_example.py

    Or scan this file directly:
    code-scalpel scan examples/security_analysis_example.py
"""

import warnings
warnings.filterwarnings("ignore", message="symbolic_execution_tools")


# =============================================================================
# DELIBERATELY VULNERABLE CODE (for demonstration)
# =============================================================================
# The following function contains a REAL SQL injection vulnerability.
# Run `code-scalpel scan examples/security_analysis_example.py` to detect it.

def vulnerable_login(username):
    """
    WARNING: This function is intentionally vulnerable for demonstration.
    DO NOT use this pattern in production code.
    """
    user_input = request.args.get("username")  # Taint source
    query = "SELECT * FROM users WHERE name='" + user_input + "'"  # Vulnerable
    cursor.execute(query)  # Taint sink - SQL Injection!
    return cursor.fetchone()


# =============================================================================
# END OF VULNERABLE CODE
# =============================================================================

from code_scalpel.symbolic_execution_tools import (
    SecurityAnalyzer,
    analyze_security,
    find_sql_injections,
    find_command_injections,
    TaintSource,
    SecuritySink,
)


def example_sql_injection():
    """Demonstrate SQL Injection detection."""
    print("=" * 60)
    print("EXAMPLE 1: SQL Injection Detection")
    print("=" * 60)
    
    vulnerable_code = '''
def get_user(user_id):
    # Taint source: user input from request
    user_id = request.args.get("id")
    
    # Vulnerable: string concatenation with user input
    query = "SELECT * FROM users WHERE id=" + user_id
    
    # Taint sink: database execution
    cursor.execute(query)
    
    return cursor.fetchone()
'''
    
    print("\nVulnerable Code:")
    print(vulnerable_code)
    
    result = analyze_security(vulnerable_code)
    
    print("\nAnalysis Result:")
    print(result.summary())
    
    if result.has_vulnerabilities:
        for vuln in result.vulnerabilities:
            print(f"\n  [VULN] {vuln.vulnerability_type} ({vuln.cwe_id})")
            print(f"     Source: {vuln.taint_source.name}")
            print(f"     Flow: {' → '.join(vuln.taint_path)}")
    
    print("\n" + "-" * 60)
    print("SAFE VERSION (parameterized query):")
    
    safe_code = '''
def get_user_safe(user_id):
    user_id = request.args.get("id")
    
    # Safe: parameterized query
    query = "SELECT * FROM users WHERE id = %s"
    cursor.execute(query, (user_id,))
    
    return cursor.fetchone()
'''
    print(safe_code)


def example_command_injection():
    """Demonstrate Command Injection detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Command Injection Detection")
    print("=" * 60)
    
    vulnerable_code = '''
def process_file(filename):
    # Taint source: user input
    filename = request.args.get("file")
    
    # Vulnerable: direct use in shell command
    cmd = "cat " + filename
    os.system(cmd)
'''
    
    print("\nVulnerable Code:")
    print(vulnerable_code)
    
    result = analyze_security(vulnerable_code)
    
    print("\nAnalysis Result:")
    print(result.summary())
    
    if result.has_vulnerabilities:
        for vuln in result.vulnerabilities:
            print(f"\n  [VULN] {vuln.vulnerability_type} ({vuln.cwe_id})")
    
    print("\n" + "-" * 60)
    print("SAFE VERSION (subprocess with list args):")
    
    safe_code = '''
def process_file_safe(filename):
    filename = request.args.get("file")
    
    # Safe: using list arguments (no shell interpretation)
    # Plus: validate filename
    import os.path
    safe_name = os.path.basename(filename)
    subprocess.run(["cat", safe_name], check=True)
'''
    print(safe_code)


def example_path_traversal():
    """Demonstrate Path Traversal detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Path Traversal Detection")
    print("=" * 60)
    
    vulnerable_code = '''
def download_file():
    # Taint source: user-controlled filename
    filename = request.args.get("filename")
    
    # Vulnerable: directly using user input in file path
    filepath = "/uploads/" + filename
    
    # Sink: opening file
    with open(filepath, "rb") as f:
        return f.read()
'''
    
    print("\nVulnerable Code:")
    print(vulnerable_code)
    
    result = analyze_security(vulnerable_code)
    
    print("\nAnalysis Result:")
    print(result.summary())
    
    print("\n" + "-" * 60)
    print("Attack Vector:")
    print("  GET /download?filename=../../../etc/passwd")
    
    print("\nSAFE VERSION:")
    safe_code = '''
def download_file_safe():
    filename = request.args.get("filename")
    
    # Safe: use secure_filename + validate path
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(filename)
    
    filepath = os.path.join("/uploads/", safe_name)
    
    # Extra safety: ensure it's within allowed directory
    if not filepath.startswith("/uploads/"):
        abort(403)
    
    with open(filepath, "rb") as f:
        return f.read()
'''
    print(safe_code)


def example_taint_flow():
    """Demonstrate taint flow tracking."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Taint Flow Tracking")
    print("=" * 60)
    
    code = '''
def build_query():
    # Level 0: Taint source
    user_input = request.args.get("search")
    
    # Level 1: Taint propagates through assignment
    search_term = user_input
    
    # Level 2: Taint propagates through string concatenation
    where_clause = "name LIKE '%" + search_term + "%'"
    
    # Level 3: More concatenation
    query = "SELECT * FROM products WHERE " + where_clause
    
    # Level 4: Sink - tainted query executed
    cursor.execute(query)
'''
    
    print("\nCode with Multi-Level Taint Flow:")
    print(code)
    
    result = analyze_security(code)
    
    print("\nTaint Flow Analysis:")
    if result.taint_flows:
        for var_name, taint_info in result.taint_flows.items():
            print(f"  {var_name}: {taint_info.source.name} → {taint_info.propagation_path}")
    
    print("\nDetected Vulnerabilities:")
    print(result.summary())


def example_convenience_functions():
    """Demonstrate convenience functions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Using Convenience Functions")
    print("=" * 60)
    
    code = '''
def vulnerable_app():
    user_id = request.args.get("id")
    name = request.args.get("name")
    file = request.args.get("file")
    
    # SQL Injection
    cursor.execute("SELECT * FROM users WHERE id=" + user_id)
    
    # Command Injection  
    os.system("echo " + name)
'''
    
    print("\nCode with Multiple Vulnerabilities:")
    print(code)
    
    print("\nFinding SQL Injections:")
    sqli = find_sql_injections(code)
    for v in sqli:
        print(f"  Found: {v.vulnerability_type}")
    
    print("\nFinding Command Injections:")
    cmdi = find_command_injections(code)
    for v in cmdi:
        print(f"  Found: {v.vulnerability_type}")


def example_json_output():
    """Demonstrate JSON serialization for integration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: JSON Output for CI/CD Integration")
    print("=" * 60)
    
    code = '''
user_input = request.args.get("q")
cursor.execute("SELECT * FROM t WHERE x=" + user_input)
'''
    
    result = analyze_security(code)
    
    import json
    output = result.to_dict()
    
    print("\nJSON Output:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    print("\n🔒 Code Scalpel v0.3.0 - Security Analysis Demo\n")
    
    example_sql_injection()
    example_command_injection()
    example_path_traversal()
    example_taint_flow()
    example_convenience_functions()
    example_json_output()
    
    print("\n" + "=" * 60)
    print("Demo complete! See docs for more details.")
    print("=" * 60)
