import asyncio
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_client_test():
    # Define server parameters
    # We run the module as a subprocess
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "code_scalpel.mcp.server"],
        env=os.environ.copy(),
    )

    print(f"Starting MCP client test against: {server_params.command} {' '.join(server_params.args)}")

    async with AsyncExitStack() as stack:
        # Connect to the server via stdio
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))
        
        await session.initialize()
        print("\n--- Server Initialized ---")

        # 1. List Tools
        print("\n[1] Listing Tools...")
        tools_result = await session.list_tools()
        tool_names = [t.name for t in tools_result.tools]
        print(f"Available tools: {tool_names}")
        
        assert "analyze_code" in tool_names
        assert "security_scan" in tool_names
        assert "symbolic_execute" in tool_names

        # 2. Test analyze_code
        print("\n[2] Testing analyze_code...")
        code_sample = """
class TestClass:
    def test_method(self):
        return "hello"
"""
        result = await session.call_tool(
            "analyze_code",
            arguments={"code": code_sample}
        )
        # FastMCP returns a list of TextContent or ImageContent
        content = result.content[0].text
        print(f"Analysis Result: {content[:100]}...")
        assert "TestClass" in content
        assert "test_method" in content

        # 3. Test security_scan (Secret Detection)
        print("\n[3] Testing security_scan (Secret Detection)...")
        secret_code = """
def get_creds():
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    return aws_key
"""
        result = await session.call_tool(
            "security_scan",
            arguments={"code": secret_code}
        )
        content = result.content[0].text
        print(f"Security Scan Result: {content[:100]}...")
        
        # Check for secret detection in the JSON output
        assert "AKIA" in content or "Secret" in content or "Hardcoded" in content
        assert "true" in content.lower() or "True" in content # has_vulnerabilities

        print("\n--- All Tests Passed Successfully ---")

if __name__ == "__main__":
    asyncio.run(run_client_test())
