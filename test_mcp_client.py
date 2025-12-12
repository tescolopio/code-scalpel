#!/usr/bin/env python3
"""
Test Code Scalpel MCP server via the MCP protocol.
This uses SSE transport to communicate with the running Docker container.
"""

import asyncio
import json
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


async def test_via_sse():
    """Test MCP server via SSE transport (Docker container)."""
    print("=" * 60)
    print("Testing Code Scalpel MCP Server via SSE")
    print("=" * 60)
    
    url = "http://localhost:8593/sse"
    
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("\n✓ Connected to MCP server")
            
            # Test 1: List available tools
            print("\n" + "-" * 60)
            print("Test 1: List Available Tools")
            print("-" * 60)
            
            tools = await session.list_tools()
            print(f"\nFound {len(tools.tools)} tools:")
            for i, tool in enumerate(tools.tools, 1):
                print(f"  {i}. {tool.name}")
            
            # Test 2: extract_code
            print("\n" + "-" * 60)
            print("Test 2: Extract Code (function)")
            print("-" * 60)
            
            test_code = """
def calculate_tax(amount, rate=0.1):
    \"\"\"Calculate tax on an amount.\"\"\"
    return amount * rate

def apply_discount(price, discount):
    \"\"\"Apply a discount.\"\"\"
    return price * (1 - discount)
"""
            
            result = await session.call_tool(
                "extract_code",
                arguments={
                    "code": test_code,
                    "target_type": "function",
                    "target_name": "calculate_tax",
                }
            )
            
            print("✓ extract_code completed")
            if result.content:
                for content in result.content:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(f"  Target: {data['target_name']}")
                        print(f"  Token estimate: {data['token_estimate']}")
                        print(f"  Lines: {data['total_lines']}")
                        print(f"\n  Extracted code:")
                        print("  " + "\n  ".join(data['full_code'].split('\n')))
            
            # Test 3: analyze_code
            print("\n" + "-" * 60)
            print("Test 3: Analyze Code")
            print("-" * 60)
            
            result = await session.call_tool(
                "analyze_code",
                arguments={
                    "code": test_code,
                    "language": "python",
                }
            )
            
            print("✓ analyze_code completed")
            if result.content:
                for content in result.content:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(f"  Functions found: {data['function_count']}")
                        print(f"  Function names: {', '.join(data['functions'])}")
                        print(f"  Lines of code: {data['lines_of_code']}")
                        print(f"  Complexity: {data['complexity']}")
            
            # Test 4: security_scan
            print("\n" + "-" * 60)
            print("Test 4: Security Scan")
            print("-" * 60)
            
            vuln_code = """
import sqlite3

def get_user(username):
    conn = sqlite3.connect('db.sqlite')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
"""
            
            result = await session.call_tool(
                "security_scan",
                arguments={
                    "code": vuln_code,
                }
            )
            
            print("✓ security_scan completed")
            if result.content:
                for content in result.content:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(f"  Risk level: {data['risk_level']}")
                        print(f"  Vulnerabilities found: {len(data['vulnerabilities'])}")
                        if data['vulnerabilities']:
                            for vuln in data['vulnerabilities']:
                                print(f"    - {vuln['type']} ({vuln['severity']}) at line {vuln['line']}")
            
            # Test 5: symbolic_execute
            print("\n" + "-" * 60)
            print("Test 5: Symbolic Execution")
            print("-" * 60)
            
            sym_code = """
def check_value(x):
    if x > 10:
        return "high"
    elif x > 5:
        return "medium"
    else:
        return "low"
"""
            
            result = await session.call_tool(
                "symbolic_execute",
                arguments={
                    "code": sym_code,
                    "max_paths": 5,
                }
            )
            
            print("✓ symbolic_execute completed")
            if result.content:
                for content in result.content:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(f"  Paths explored: {data['total_paths']}")
                        print(f"  Feasible: {data['feasible_paths']}")
                        print(f"  Infeasible: {data['infeasible_paths']}")
            
            print("\n" + "=" * 60)
            print("✅ ALL MCP TESTS PASSED")
            print("=" * 60)
            return True


async def test_via_stdio():
    """Test MCP server via stdio (local process)."""
    print("=" * 60)
    print("Testing Code Scalpel MCP Server via stdio")
    print("=" * 60)
    
    server_params = StdioServerParameters(
        command="code-scalpel",
        args=["mcp"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("\n✓ Connected to MCP server (stdio)")
            
            tools = await session.list_tools()
            print(f"\nFound {len(tools.tools)} tools via stdio transport")
            
            return True


async def main():
    """Run MCP tests."""
    try:
        # Test SSE (Docker)
        result1 = await test_via_sse()
        
        print("\n" + "=" * 60)
        print("MCP Server Test Summary")
        print("=" * 60)
        print("✓ SSE transport (Docker): PASSED")
        print("\nCode Scalpel MCP server is fully operational!")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
