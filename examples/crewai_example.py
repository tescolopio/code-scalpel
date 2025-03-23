from crewai import Agent, Crew, Task
from pdg_tools import CodeAnalysisToolkit

# Define agents
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Analyze code quality and suggest improvements",
    backstory="Expert code reviewer with deep understanding of Python best practices",
    tools=[CodeAnalysisToolkit()]
)

security_analyst = Agent(
    role="Security Analyst",
    goal="Identify potential security vulnerabilities",
    backstory="Cybersecurity expert specializing in code security",
    tools=[CodeAnalysisToolkit()]
)

# Create tasks
review_task = Task(
    description="Perform comprehensive code review",
    agent=code_reviewer
)

security_task = Task(
    description="Analyze code for security vulnerabilities",
    agent=security_analyst
)

# Create crew
analysis_crew = Crew(
    agents=[code_reviewer, security_analyst],
    tasks=[review_task, security_task]
)

# Execute analysis
result = analysis_crew.execute()