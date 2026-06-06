from langchain.agents import create_agent

from langchain.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

model = ChatOpenRouter(model="~anthropic/claude-haiku-latest")

agent = create_agent(model)

response = agent.invoke({"messages": [HumanMessage(content="What is the capital of the moon?")]})

print(response)

