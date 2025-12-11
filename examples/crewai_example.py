"""
Example demonstrating CrewAIScalpel for code analysis and refactoring.

This example shows how to use CrewAIScalpel standalone for code analysis,
and how it can be integrated with CrewAI agents.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from integrations import CrewAIScalpel


async def main():
    # Initialize CrewAIScalpel
    scalpel = CrewAIScalpel()

    # Sample code with issues
    code = '''
def ProcessUserData(userData, settings):
    """Process user data with settings."""
    if userData is not None:
        if settings is not None:
            if settings.get("validate"):
                if userData.get("email"):
                    result = eval(f"process_{settings['mode']}(userData)")
                    return result
    return None
'''

    print("=== Code Analysis ===")
    result = await scalpel.analyze_async(code)
    print(f"Success: {result.success}")
    print(f"Total Issues: {result.analysis.get('total_issues')}")
    print("\nIssues Found:")
    for issue in result.issues:
        print(
            f"  - Type: {issue.get('type')}, Category: {issue.get('category', 'N/A')}"
        )

    print("\nSuggestions:")
    for suggestion in result.suggestions:
        print(f"  - {suggestion}")

    print("\n=== Security Scan ===")
    security_result = await scalpel.analyze_security_async(code)
    print(f"Risk Level: {security_result['risk_level'].upper()}")
    print("Recommendations:")
    for rec in security_result["recommendations"]:
        print(f"  - {rec}")

    print("\n=== Refactoring ===")
    refactor_result = await scalpel.refactor_async(code, "improve code quality")
    print(f"Refactored: {refactor_result.refactored_code is not None}")

    # Show available tools for CrewAI integration
    print("\n=== Available Tools for CrewAI ===")
    tools = scalpel.get_crewai_tools()
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:50]}...")


# Example of CrewAI integration (requires CrewAI installed and configured)
def demo_crewai_integration():
    """
    Demonstrates how to integrate CrewAIScalpel with CrewAI agents.

    Note: Requires:
        - pip install crewai
        - OPENAI_API_KEY or other LLM configuration
    """
    try:
        from crewai import Agent, Crew, Task
        from crewai.tools import BaseTool
        from pydantic import BaseModel, Field

        # Initialize Scalpel
        scalpel = CrewAIScalpel()

        # Create proper CrewAI tools from Scalpel methods
        class CodeInput(BaseModel):
            code: str = Field(description="Python code to analyze")

        class AnalyzeCodeTool(BaseTool):
            name: str = "analyze_code"
            description: str = (
                "Analyzes Python code for style issues and security vulnerabilities"
            )
            args_schema: type[BaseModel] = CodeInput

            def _run(self, code: str) -> str:
                result = scalpel.analyze(code)
                return str(result.to_dict())

        class SecurityScanTool(BaseTool):
            name: str = "security_scan"
            description: str = "Performs security analysis on Python code"
            args_schema: type[BaseModel] = CodeInput

            def _run(self, code: str) -> str:
                result = scalpel._analyze_security_sync(code)
                return str(result)

        # Create agents with proper tool instances
        code_reviewer = Agent(
            role="Code Reviewer",
            goal="Analyze code quality and suggest improvements",
            backstory="Expert code reviewer with deep understanding of Python best practices",
            tools=[AnalyzeCodeTool()],
        )

        security_analyst = Agent(
            role="Security Analyst",
            goal="Identify potential security vulnerabilities",
            backstory="Cybersecurity expert specializing in code security",
            tools=[SecurityScanTool()],
        )

        # Create tasks
        review_task = Task(
            description="Perform comprehensive code review", agent=code_reviewer
        )

        security_task = Task(
            description="Analyze code for security vulnerabilities",
            agent=security_analyst,
        )

        # Create crew
        analysis_crew = Crew(
            agents=[code_reviewer, security_analyst], tasks=[review_task, security_task]
        )

        return analysis_crew

    except ImportError as e:
        print(f"CrewAI not configured: {e}")
        print("To use CrewAI integration, install: pip install crewai")
        return None


if __name__ == "__main__":
    # Run the async demo
    asyncio.run(main())

    # Show CrewAI integration example
    print("\n=== CrewAI Integration Demo ===")
    crew = demo_crewai_integration()
    if crew:
        print("CrewAI crew created successfully!")
