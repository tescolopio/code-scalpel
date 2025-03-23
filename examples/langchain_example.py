from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.llms import OpenAI

# Define the tools
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Searches the internet using DuckDuckGo"
    ),
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Performs arithmetic calculations"
    )
]

# Initialize the agent
llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Run the agent
query = "What is the capital of France and what is 10 + 5?"
result = agent.run(query)
print(result)
