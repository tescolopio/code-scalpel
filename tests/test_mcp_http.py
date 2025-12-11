import pytest
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client


@pytest.mark.asyncio
async def test_mcp_http_connection():
    """Test connection to the running MCP server via HTTP SSE."""
    url = "http://localhost:8593/sse"
    print(f"Connecting to {url}...")

    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                tool_names = [t.name for t in tools.tools]
                print(f"Tools: {tool_names}")

                assert "analyze_code" in tool_names
                assert "security_scan" in tool_names

    except Exception as e:
        pytest.fail(f"Failed to connect to MCP server: {e}")


if __name__ == "__main__":
    asyncio.run(test_mcp_http_connection())
