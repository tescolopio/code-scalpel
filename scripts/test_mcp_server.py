#!/usr/bin/env python3
"""
Test script for Code Scalpel MCP Server.

This script validates the MCP server is working correctly by:
1. Connecting to the server
2. Initializing a session
3. Listing available tools
4. Calling each tool with test data
"""

import json
import sys
import requests
from typing import Optional


class MCPClient:
    """Simple MCP client for testing."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8593"):
        self.base_url = base_url
        self.session_id: Optional[str] = None
        self.request_id = 0
    
    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id
    
    def _request(self, method: str, params: dict = None) -> dict:
        """Send JSON-RPC request to MCP server."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["mcp-session-id"] = self.session_id
        
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params:
            payload["params"] = params
        
        response = requests.post(
            f"{self.base_url}/mcp",
            headers=headers,
            json=payload,
            timeout=30,
        )
        
        # Extract session ID from response
        if "mcp-session-id" in response.headers:
            self.session_id = response.headers["mcp-session-id"]
        
        # Parse SSE response
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                return json.loads(line[6:])
        
        return {"error": "No data in response", "raw": response.text}
    
    def initialize(self) -> dict:
        """Initialize MCP session."""
        return self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        })
    
    def list_tools(self) -> dict:
        """List available tools."""
        return self._request("tools/list", {})
    
    def call_tool(self, name: str, arguments: dict) -> dict:
        """Call an MCP tool."""
        return self._request("tools/call", {
            "name": name,
            "arguments": arguments,
        })


def test_server(base_url: str = "http://127.0.0.1:8593"):
    """Run all tests against the MCP server."""
    print(f"Testing MCP Server at {base_url}")
    print("=" * 60)
    
    client = MCPClient(base_url)
    
    # Test 1: Initialize
    print("\n[1] Testing initialize...")
    result = client.initialize()
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    
    server_info = result.get("result", {}).get("serverInfo", {})
    print(f"    Server: {server_info.get('name')} v{server_info.get('version')}")
    print(f"    Session ID: {client.session_id[:16]}...")
    print("    PASSED")
    
    # Test 2: List tools
    print("\n[2] Testing tools/list...")
    result = client.list_tools()
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    
    tools = result.get("result", {}).get("tools", [])
    tool_names = [t["name"] for t in tools]
    print(f"    Available tools: {tool_names}")
    
    expected_tools = ["analyze_code", "security_scan", "symbolic_execute"]
    for tool in expected_tools:
        if tool not in tool_names:
            print(f"    FAILED: Missing tool '{tool}'")
            return False
    print("    PASSED")
    
    # Test 3: Call analyze_code
    print("\n[3] Testing analyze_code tool...")
    test_code = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def add(self, a, b):
        return a + b
'''
    result = client.call_tool("analyze_code", {"code": test_code})
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    
    content = result.get("result", {}).get("structuredContent", {})
    print(f"    Functions: {content.get('functions')}")
    print(f"    Classes: {content.get('classes')}")
    print(f"    Complexity: {content.get('complexity')}")
    
    if not content.get("success"):
        print(f"    FAILED: {content.get('error')}")
        return False
    print("    PASSED")
    
    # Test 4: Call security_scan
    print("\n[4] Testing security_scan tool...")
    vuln_code = '''
import os
def dangerous(user_input):
    os.system("echo " + user_input)
'''
    result = client.call_tool("security_scan", {"code": vuln_code})
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    
    content = result.get("result", {}).get("structuredContent", {})
    print(f"    Has vulnerabilities: {content.get('has_vulnerabilities')}")
    print(f"    Vulnerability count: {content.get('vulnerability_count')}")
    print(f"    Risk level: {content.get('risk_level')}")
    
    if not content.get("success"):
        print(f"    FAILED: {content.get('error')}")
        return False
    print("    PASSED")
    
    # Test 5: Call symbolic_execute
    print("\n[5] Testing symbolic_execute tool...")
    branch_code = '''
def classify(x):
    if x > 0:
        return "positive"
    elif x < 0:
        return "negative"
    else:
        return "zero"
'''
    result = client.call_tool("symbolic_execute", {"code": branch_code, "max_paths": 5})
    if "error" in result:
        print(f"    FAILED: {result['error']}")
        return False
    
    content = result.get("result", {}).get("structuredContent", {})
    print(f"    Paths explored: {content.get('paths_explored')}")
    print(f"    Symbolic variables: {content.get('symbolic_variables')}")
    
    if not content.get("success"):
        print(f"    FAILED: {content.get('error')}")
        return False
    print("    PASSED")
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8593"
    success = test_server(url)
    sys.exit(0 if success else 1)
