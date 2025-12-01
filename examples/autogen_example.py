"""
Example demonstrating AutogenScalpel for code analysis.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from integrations import AutogenScalpel

# Example code to analyze
code = """
def calculate_factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
"""

import asyncio

async def main():
    # Create the Scalpel analyzer
    scalpel = AutogenScalpel()
    
    # Analyze the code
    result = await scalpel.analyze_async(code)
    
    print("Analysis Results:")
    print(f"  Parsed: {result.ast_analysis.get('parsed')}")
    print(f"  Style Issues: {result.ast_analysis.get('style_issues_count')}")
    print(f"  Security Issues: {result.ast_analysis.get('security_issues_count')}")
    
    print("\nSuggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")
    
    # Demonstrate tool description for Autogen integration
    tool_desc = scalpel.get_tool_description()
    print(f"\nTool Name: {tool_desc['name']}")
    print(f"Description: {tool_desc['description']}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
