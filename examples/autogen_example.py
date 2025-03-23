from src.integrations import AutogenCodeAnalysisAgent

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

# Configure the agent
config = {
    "llm_config": [
        {"model": "gpt-4", "temperature": 0.7}
    ]
}

import asyncio

async def main():
    # Create and use the agent
    agent = AutogenCodeAnalysisAgent(config)
    result = await agent.analyze_code(code)

    print("Analysis Results:")
    print(result["analysis"])
    print("\nSuggestions:")
    for suggestion in result["suggestions"]:
        print(f"- {suggestion}")

# Run the async main function
asyncio.run(main())
