from dotenv import load_dotenv
import os
from langchain_anthropic.chat_models import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory


# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set up Claude model
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.5,
    anthropic_api_key=anthropic_api_key
)

# Set up OpenAI model
chatgpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    openai_api_key=openai_api_key
)

def ask_claude(question: str) -> str:
    """Ask a question to the Claude model."""
    response = claude.invoke(question)
    return response.content

def ask_chatgpt(question: str) -> str:
    """Ask a question to the ChatGPT model."""
    response = chatgpt.invoke(question)
    return response.content

# Initialize the LLM (you can choose either Claude or ChatGPT)
tools = [
    Tool(
        name="ClaudeTool",
        func=ask_claude,
        description="Use this tool for thoughtful, nuanced responses from the Claude model."
    ),
    Tool(
        name="ChatGPTTool",
        func=ask_chatgpt,
        description="Use this tool for fast or technical answers from the ChatGPT model."
    )
]

# 
# Tool: multiply two numbers
# def multiply_numbers(input: str) -> str:
#     try:
#         numbers = [int(x) for x in input.split()]
#         result = numbers[0] * numbers[1]
#         return f"The result is {result}."
#     except:
#         return "Please enter two numbers separated by a space."

# math_tool = Tool(
#     name="Multiplier",
#     func=multiply_numbers,
#     description="Multiplies two numbers. Input should be like: '6 7'"
# )

# Set up memory for the agent
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Initialize LangChain Agent
agent = initialize_agent(
    tools=tools,
    llm=chatgpt,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Ask the agent something
# response = agent.run("What is 3 multiplied by 3?")
# print(response)

# # Example prompt (multi-turn dialogue)
# print(agent.run("Hi, I'm Wendy."))
# print(agent.run("What's my name?"))
# print(agent.run("Do you have a name?"))
# print(agent.run("But if you could pick a name, what would it be?"))

# Example usage of the agent with Claude and ChatGPT tools
response = agent.run("Use ClaudeTool to write a thoughtful thank-you email.")
print(response)

response2 = agent.run("Now use ChatGPTTool to summarize a blog post about RAG.")
print(response2)
