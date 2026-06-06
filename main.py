from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


model = ChatOpenRouter(model="qwen/qwen3.6-flash", temperature=2.0)

response = model.invoke("What is the capital of the moon?")

print(response.content)