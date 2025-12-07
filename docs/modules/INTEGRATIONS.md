# AI Agent Integrations - Complete Reference

Code Scalpel provides pre-built integrations for popular AI agent frameworks.

---

## Table of Contents

1. [Overview](#overview)
2. [AutoGen Integration](#autogen-integration)
3. [CrewAI Integration](#crewai-integration)
4. [LangChain Integration](#langchain-integration)
5. [Claude Integration](#claude-integration)
6. [Custom Integrations](#custom-integrations)
7. [REST API Server](#rest-api-server)

---

## Overview

Code Scalpel supports multiple integration patterns:

| Method | Best For | Setup Complexity |
|--------|----------|------------------|
| **MCP Server** | Claude Desktop, Cursor | Low |
| **AutoGen** | Microsoft AutoGen agents | Medium |
| **CrewAI** | CrewAI multi-agent systems | Medium |
| **LangChain** | LangChain chains/agents | Medium |
| **REST API** | Custom applications | Low |

---

## AutoGen Integration

### Installation

```bash
pip install code-scalpel pyautogen
```

### AutogenScalpel

The `AutogenScalpel` class wraps Code Scalpel for AutoGen compatibility.

```python
from code_scalpel.integrations import AutogenScalpel

scalpel = AutogenScalpel()
```

### Available Tools

#### analyze_tool

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json

# Create Code Scalpel tools
scalpel = AutogenScalpel()

# Create assistant with analysis capability
assistant = AssistantAgent(
    name="code_analyst",
    llm_config={"config_list": config_list_from_json("OAI_CONFIG_LIST")},
    system_message="You are a code analysis expert."
)

# Register tool
assistant.register_function(
    function_map={"analyze_code": scalpel.analyze}
)

# Create user proxy
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config=False
)

# Start conversation
user.initiate_chat(
    assistant,
    message="Analyze this code:\n\ndef factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"
)
```

#### security_tool

```python
# Register security scanning
assistant.register_function(
    function_map={"security_scan": scalpel.security_scan}
)

user.initiate_chat(
    assistant,
    message="Check this code for vulnerabilities:\n\nquery = f'SELECT * FROM users WHERE id={user_input}'"
)
```

### Complete Example

```python
from autogen import AssistantAgent, UserProxyAgent
from code_scalpel.integrations import AutogenScalpel

# Initialize
scalpel = AutogenScalpel()

# Create specialized agents
analyst = AssistantAgent(
    name="code_analyst",
    system_message="""You analyze code structure and quality.
    Use analyze_code to examine code.
    Report on complexity, functions, and issues."""
)

security_expert = AssistantAgent(
    name="security_expert", 
    system_message="""You find security vulnerabilities.
    Use security_scan to detect issues.
    Provide CWE references and remediation."""
)

# Register tools
analyst.register_function(
    function_map={"analyze_code": scalpel.analyze}
)
security_expert.register_function(
    function_map={"security_scan": scalpel.security_scan}
)

# Create group chat
from autogen import GroupChat, GroupChatManager

groupchat = GroupChat(
    agents=[analyst, security_expert],
    messages=[],
    max_round=10
)

manager = GroupChatManager(groupchat=groupchat)

# Analyze code
user = UserProxyAgent(name="user")
user.initiate_chat(
    manager,
    message="""Analyze and check security of:
    
    def process_order(order_id):
        query = f"SELECT * FROM orders WHERE id={order_id}"
        return db.execute(query)
    """
)
```

### AutogenScalpel Methods

| Method | Description |
|--------|-------------|
| `analyze(code)` | Analyze code structure |
| `security_scan(code)` | Detect vulnerabilities |
| `symbolic_execute(code, function)` | Explore paths |
| `generate_tests(code, function)` | Generate test cases |

---

## CrewAI Integration

### Installation

```bash
pip install code-scalpel crewai
```

### CrewAIScalpel

```python
from code_scalpel.integrations import CrewAIScalpel

scalpel = CrewAIScalpel()
```

### Creating Tools

```python
from crewai import Agent, Task, Crew

# Create Code Scalpel tools
scalpel = CrewAIScalpel()
analysis_tool = scalpel.create_tool("analyze")
security_tool = scalpel.create_tool("security")

# Create agent with tools
code_reviewer = Agent(
    role="Senior Code Reviewer",
    goal="Review code for quality and security issues",
    backstory="You are an expert code reviewer with 15 years of experience.",
    tools=[analysis_tool, security_tool],
    verbose=True
)

# Create task
review_task = Task(
    description="""
    Review the following code:
    {code}
    
    1. Analyze the structure and complexity
    2. Check for security vulnerabilities
    3. Provide improvement recommendations
    """,
    agent=code_reviewer,
    expected_output="A detailed code review report"
)

# Create crew
crew = Crew(
    agents=[code_reviewer],
    tasks=[review_task],
    verbose=True
)

# Execute
result = crew.kickoff(inputs={
    "code": """
    def get_user(user_id):
        query = f"SELECT * FROM users WHERE id={user_id}"
        return db.execute(query)
    """
})
```

### Multi-Agent Example

```python
from crewai import Agent, Task, Crew, Process
from code_scalpel.integrations import CrewAIScalpel

scalpel = CrewAIScalpel()

# Specialized agents
architect = Agent(
    role="Software Architect",
    goal="Analyze code architecture and design patterns",
    tools=[scalpel.create_tool("analyze")],
)

security_analyst = Agent(
    role="Security Analyst",
    goal="Find security vulnerabilities and suggest fixes",
    tools=[scalpel.create_tool("security")],
)

test_engineer = Agent(
    role="Test Engineer",
    goal="Generate comprehensive test cases",
    tools=[scalpel.create_tool("test_gen")],
)

# Sequential tasks
architecture_task = Task(
    description="Analyze the code architecture",
    agent=architect,
)

security_task = Task(
    description="Perform security audit",
    agent=security_analyst,
)

testing_task = Task(
    description="Generate test cases for all functions",
    agent=test_engineer,
)

# Create crew with sequential process
crew = Crew(
    agents=[architect, security_analyst, test_engineer],
    tasks=[architecture_task, security_task, testing_task],
    process=Process.sequential
)

result = crew.kickoff(inputs={"code": code})
```

### CrewAIScalpel Methods

| Method | Returns |
|--------|---------|
| `create_tool("analyze")` | Analysis tool |
| `create_tool("security")` | Security scanning tool |
| `create_tool("symbolic")` | Symbolic execution tool |
| `create_tool("test_gen")` | Test generation tool |
| `create_tool("refactor")` | Refactor simulation tool |

---

## LangChain Integration

### Installation

```bash
pip install code-scalpel langchain langchain-openai
```

### Creating Tools

```python
from langchain.tools import Tool
from code_scalpel import CodeAnalyzer

analyzer = CodeAnalyzer()

# Create LangChain tool
analyze_tool = Tool(
    name="code_analyzer",
    description="Analyze Python code structure and complexity. Input should be the code to analyze.",
    func=lambda code: str(analyzer.analyze(code).model_dump())
)
```

### Agent with Tools

```python
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from code_scalpel import CodeAnalyzer
from code_scalpel.symbolic_execution_tools import SecurityAnalyzer

# Create tools
analyzer = CodeAnalyzer()
security = SecurityAnalyzer()

tools = [
    Tool(
        name="analyze_code",
        description="Analyze code structure. Returns functions, classes, complexity.",
        func=lambda code: str(analyzer.analyze(code).model_dump())
    ),
    Tool(
        name="security_scan",
        description="Scan code for security vulnerabilities.",
        func=lambda code: str(security.analyze_code(code).model_dump())
    ),
]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use agent
result = agent.invoke(
    "Analyze this code and check for security issues:\n\n"
    "def login(username, password):\n"
    "    query = f\"SELECT * FROM users WHERE user='{username}'\"\n"
    "    return db.execute(query)"
)
```

### Structured Output

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class AnalysisInput(BaseModel):
    code: str = Field(description="The source code to analyze")
    language: str = Field(default="python", description="Programming language")

def analyze_with_lang(code: str, language: str = "python") -> str:
    result = analyzer.analyze(code)
    return f"""
    Functions: {result.function_count}
    Classes: {result.class_count}
    Complexity: {result.metrics.cyclomatic_complexity}
    """

analyze_tool = StructuredTool.from_function(
    func=analyze_with_lang,
    name="analyze_code",
    description="Analyze code structure",
    args_schema=AnalysisInput
)
```

### LangChain Chain Example

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from code_scalpel import analyze_code

# Create analysis function
def get_code_summary(code: str) -> str:
    result = analyze_code(code)
    return f"""
    Functions: {', '.join(f.name for f in result.functions)}
    Classes: {', '.join(c.name for c in result.classes)}
    Complexity: {result.metrics.cyclomatic_complexity}
    Issues: {len(result.issues)}
    """

# Create chain
template = """
Based on this code analysis:
{analysis}

Original code:
{code}

Provide recommendations for improvement.
"""

prompt = PromptTemplate(
    input_variables=["analysis", "code"],
    template=template
)

llm = ChatOpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

# Use chain
code = "def foo(): pass"
analysis = get_code_summary(code)
result = chain.invoke({"analysis": analysis, "code": code})
```

---

## Claude Integration

### MCP Server (Recommended)

See [MCP Server documentation](MCP_SERVER.md) for Claude Desktop integration.

### Direct API Integration

```python
import anthropic
from code_scalpel import CodeAnalyzer

client = anthropic.Anthropic()
analyzer = CodeAnalyzer()

def analyze_with_claude(code: str) -> str:
    # Get Code Scalpel analysis
    analysis = analyzer.analyze(code)
    
    # Ask Claude to interpret
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"""
                I analyzed this code with Code Scalpel:
                
                Code:
                ```python
                {code}
                ```
                
                Analysis Results:
                - Functions: {analysis.function_count}
                - Classes: {analysis.class_count}
                - Complexity: {analysis.metrics.cyclomatic_complexity}
                - Issues: {analysis.issues}
                
                Please provide insights and recommendations.
                """
            }
        ]
    )
    
    return message.content[0].text

# Use
result = analyze_with_claude("""
def calculate_price(items, discount=0):
    total = sum(item.price for item in items)
    if discount > 0:
        total *= (1 - discount)
    return total
""")
print(result)
```

### Tool Use (Function Calling)

```python
import anthropic
from code_scalpel import CodeAnalyzer
from code_scalpel.symbolic_execution_tools import SecurityAnalyzer

client = anthropic.Anthropic()
code_analyzer = CodeAnalyzer()
security_analyzer = SecurityAnalyzer()

# Define tools
tools = [
    {
        "name": "analyze_code",
        "description": "Analyze code structure and complexity",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to analyze"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "security_scan",
        "description": "Scan code for security vulnerabilities",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to scan"}
            },
            "required": ["code"]
        }
    }
]

# Handle tool calls
def handle_tool_call(name: str, inputs: dict) -> str:
    if name == "analyze_code":
        result = code_analyzer.analyze(inputs["code"])
        return str(result.model_dump())
    elif name == "security_scan":
        result = security_analyzer.analyze_code(inputs["code"])
        return str(result.model_dump())
    return "Unknown tool"

# Conversation loop
messages = []
messages.append({
    "role": "user",
    "content": "Analyze this code for structure and security:\n\n"
               "def get_user(id):\n"
               "    return db.execute(f'SELECT * FROM users WHERE id={id}')"
})

while True:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    if response.stop_reason == "tool_use":
        # Handle tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = handle_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    else:
        # Final response
        print(response.content[0].text)
        break
```

---

## Custom Integrations

### Base Integration Pattern

```python
from code_scalpel import CodeAnalyzer, analyze_code
from code_scalpel.pdg_tools import build_pdg, PDGAnalyzer
from code_scalpel.symbolic_execution_tools import (
    SymbolicAnalyzer, SecurityAnalyzer
)

class CodeScalpelIntegration:
    """Base class for custom integrations."""
    
    def __init__(self):
        self.code_analyzer = CodeAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.symbolic_analyzer = SymbolicAnalyzer()
    
    def analyze(self, code: str) -> dict:
        """Analyze code structure."""
        result = self.code_analyzer.analyze(code)
        return {
            "functions": result.function_count,
            "classes": result.class_count,
            "complexity": result.metrics.cyclomatic_complexity,
            "issues": [str(i) for i in result.issues]
        }
    
    def security_scan(self, code: str) -> dict:
        """Scan for vulnerabilities."""
        result = self.security_analyzer.analyze_code(code)
        return {
            "has_vulnerabilities": result.has_vulnerabilities,
            "count": len(result.vulnerabilities),
            "vulnerabilities": [
                {
                    "type": v.vulnerability_type,
                    "severity": v.severity,
                    "line": v.line,
                    "cwe": v.cwe
                }
                for v in result.vulnerabilities
            ]
        }
    
    def get_data_flow(self, code: str, variable: str) -> dict:
        """Analyze data flow for a variable."""
        pdg = build_pdg(code)
        analyzer = PDGAnalyzer(pdg)
        
        return {
            "depends_on": analyzer.get_dependencies(variable),
            "affects": analyzer.get_dependents(variable)
        }
```

### REST Client Integration

```python
import requests
from typing import Any

class CodeScalpelClient:
    """HTTP client for Code Scalpel REST API."""
    
    def __init__(self, base_url: str = "http://localhost:8593"):
        self.base_url = base_url
    
    def analyze(self, code: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/analyze",
            json={"code": code}
        )
        response.raise_for_status()
        return response.json()
    
    def security_scan(self, code: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/security",
            json={"code": code}
        )
        response.raise_for_status()
        return response.json()
    
    def generate_tests(
        self, code: str, function_name: str
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/generate-tests",
            json={"code": code, "function_name": function_name}
        )
        response.raise_for_status()
        return response.json()
```

---

## REST API Server

For applications that prefer HTTP over MCP.

### Starting the Server

```bash
# Using module
python -m code_scalpel.integrations.rest_api_server --port 8593

# Using CLI
code-scalpel server --port 8593
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Analyze code structure |
| `/security` | POST | Security vulnerability scan |
| `/symbolic` | POST | Symbolic execution |
| `/tests` | POST | Generate unit tests |
| `/refactor` | POST | Simulate refactoring |
| `/health` | GET | Health check |

### Example Requests

```bash
# Analyze code
curl -X POST http://localhost:8593/analyze \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello(): pass"}'

# Security scan
curl -X POST http://localhost:8593/security \
  -H "Content-Type: application/json" \
  -d '{"code": "db.execute(f\"SELECT * FROM users WHERE id={user_id}\")"}'

# Generate tests
curl -X POST http://localhost:8593/tests \
  -H "Content-Type: application/json" \
  -d '{"code": "def is_even(n): return n % 2 == 0", "function_name": "is_even"}'
```

### Python Client

```python
import requests

class RESTClient:
    def __init__(self, base_url="http://localhost:8593"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def analyze(self, code):
        return self.session.post(
            f"{self.base_url}/analyze",
            json={"code": code}
        ).json()
    
    def security_scan(self, code):
        return self.session.post(
            f"{self.base_url}/security",
            json={"code": code}
        ).json()

# Usage
client = RESTClient()
result = client.analyze("def foo(): pass")
print(result)
```

---

*AI Agent Integrations - Bringing Code Scalpel to every AI framework.*
