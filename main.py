from dotenv import load_dotenv
import os
from langchain_anthropic.chat_models import ChatAnthropic
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType


# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Set up Claude model
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.5,
    anthropic_api_key=anthropic_api_key
)

# Tool: multiply two numbers
def multiply_numbers(input: str) -> str:
    try:
        numbers = [int(x) for x in input.split()]
        result = numbers[0] * numbers[1]
        return f"The result is {result}."
    except:
        return "Please enter two numbers separated by a space."

math_tool = Tool(
    name="Multiplier",
    func=multiply_numbers,
    description="Multiplies two numbers. Input should be like: '6 7'"
)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=[math_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the agent something
response = agent.run("What is 3 multiplied by 3?")
print(response)
