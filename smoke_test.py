#!/usr/bin/env python3
"""
Smoke test for extract_code functionality.
Tests the actual execution of the tool with sample code.
"""

import sys
import asyncio
from code_scalpel.mcp.server import mcp


async def test_extract_code_basic():
    """Test basic extraction functionality."""
    print("Smoke Test: extract_code Tool")
    print("=" * 60)
    
    # Sample test code
    test_code = """
def calculate_tax(amount, rate=0.1):
    \"\"\"Calculate tax on an amount.\"\"\"
    return amount * rate

def apply_discount(price, discount):
    \"\"\"Apply a discount to a price.\"\"\"
    return price * (1 - discount)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
"""
    
    print("\nTest 1: Extract function 'calculate_tax'")
    print("-" * 60)
    
    try:
        # call_tool returns (content_list, result_dict)
        _, result = await mcp.call_tool(
            "extract_code",
            {
                "code": test_code,
                "target_type": "function",
                "target_name": "calculate_tax",
            },
        )
        
        print(f"✓ Success!")
        print(f"  Target code length: {len(result['target_code'])} chars")
        print(f"  Token estimate: {result.get('token_estimate', 'N/A')}")
        print(f"\nExtracted code:")
        print(result['full_code'])
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n" + "-" * 60)
    print("\nTest 2: Extract method 'Calculator.add'")
    print("-" * 60)
    
    try:
        _, result = await mcp.call_tool(
            "extract_code",
            {
                "code": test_code,
                "target_type": "method",
                "target_name": "Calculator.add",
            },
        )
        
        print(f"✓ Success!")
        print(f"  Target code length: {len(result['target_code'])} chars")
        print(f"\nExtracted code:")
        print(result['full_code'])
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n" + "-" * 60)
    print("\nTest 3: Extract class 'Calculator'")
    print("-" * 60)
    
    try:
        _, result = await mcp.call_tool(
            "extract_code",
            {
                "code": test_code,
                "target_type": "class",
                "target_name": "Calculator",
            },
        )
        
        print(f"✓ Success!")
        print(f"  Target code length: {len(result['target_code'])} chars")
        print(f"\nExtracted code:")
        print(result['full_code'])
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All smoke tests passed!")
    return True


async def test_crawl_project_basic():
    """Test basic project crawling functionality."""
    print("\n\nSmoke Test: crawl_project Tool")
    print("=" * 60)
    
    print("\nTest: Crawl current project (src/code_scalpel)")
    print("-" * 60)
    
    try:
        _, result = await mcp.call_tool(
            "crawl_project",
            {
                "root_path": "src/code_scalpel",
                "include_report": False,
                "complexity_threshold": 10,
            },
        )
        
        print(f"✓ Success!")
        print(f"  Files analyzed: {result.get('summary', {}).get('total_files', 0)}")
        print(f"  Total functions: {result.get('summary', {}).get('total_functions', 0)}")
        print(f"  Total classes: {result.get('summary', {}).get('total_classes', 0)}")
        print(f"  Average complexity: {result.get('summary', {}).get('average_complexity', 0):.2f}")
        
        # Show a sample file
        files = result.get('files', [])
        if files:
            sample = files[0]
            print(f"\n  Sample file: {sample.get('path', 'unknown')}")
            print(f"    Functions: {sample.get('function_count', 0)}")
            print(f"    Classes: {sample.get('class_count', 0)}")
            print(f"    Complexity: {sample.get('complexity', 0)}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ crawl_project smoke test passed!")
    return True


async def main():
    """Run all smoke tests."""
    result1 = await test_extract_code_basic()
    result2 = await test_crawl_project_basic()
    
    print("\n" + "=" * 60)
    if result1 and result2:
        print("✓ ALL SMOKE TESTS PASSED")
        print("=" * 60)
        print("\n🎉 Deployment successful! MCP server is ready for E2E testing.")
        print("\nNext steps:")
        print("  1. Add to Claude Desktop config")
        print("  2. Test with MCP Inspector")
        print("  3. Run full integration tests")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
