#!/usr/bin/env python3
"""
Quick smoke test for MCP server tool registration.
Verifies that extract_code and crawl_project are available.
"""

import sys
import asyncio
from code_scalpel.mcp.server import mcp


async def test_tool_registration():
    """Check that all expected tools are registered."""
    print("Code Scalpel MCP Server Tool Registration Test")
    print("=" * 60)
    
    # Get all registered tools
    tools = await mcp.list_tools()
    tool_names = [tool.name for tool in tools]
    
    print(f"\nTotal tools registered: {len(tool_names)}")
    print("\nRegistered tools:")
    for i, name in enumerate(sorted(tool_names), 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 60)
    
    # Check for critical tools
    required_tools = [
        "extract_code",
        "crawl_project",
        "analyze_code",
        "security_scan",
        "symbolic_execute",
        "generate_unit_tests",
        "update_symbol",
        "simulate_refactor",
    ]
    
    missing = []
    found = []
    for tool in required_tools:
        if tool in tool_names:
            found.append(tool)
        else:
            missing.append(tool)
    
    print(f"\nRequired tools found: {len(found)}/{len(required_tools)}")
    for tool in found:
        print(f"  ✓ {tool}")
    
    if missing:
        print(f"\nMISSING TOOLS ({len(missing)}):")
        for tool in missing:
            print(f"  ✗ {tool}")
        return False
    
    print("\n✓ All required tools are registered!")
    return True


async def test_extract_code_tool():
    """Quick validation that extract_code tool has the right structure."""
    print("\n" + "=" * 60)
    print("Testing extract_code tool definition...")
    print("=" * 60)
    
    tool_list = await mcp.list_tools()
    tools = {tool.name: tool for tool in tool_list}
    
    if "extract_code" not in tools:
        print("✗ extract_code tool not found!")
        return False
    
    tool = tools["extract_code"]
    print(f"\nTool: {tool.name}")
    print(f"Description: {tool.description[:100]}...")
    
    # Check parameters
    schema = tool.inputSchema
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    print(f"\nRequired parameters: {required}")
    print(f"Optional parameters: {[k for k in properties.keys() if k not in required]}")
    
    # Verify key parameters exist
    expected_params = ["target_type", "target_name"]
    for param in expected_params:
        if param not in properties:
            print(f"✗ Missing parameter: {param}")
            return False
    
    print("\n✓ extract_code tool structure is valid!")
    return True


async def main():
    """Main test runner."""
    result1 = await test_tool_registration()
    result2 = await test_extract_code_tool()
    return result1 and result2


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
