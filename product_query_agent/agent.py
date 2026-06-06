from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

import os

load_dotenv()

model = ChatOpenRouter(model="~anthropic/claude-haiku-latest")


PRODUCTS = {
    "wireless headphones": {
        "price": 79.99,
        "rating": 4.6,
        "description": "Over-ear Bluetooth, 30-hr battery, active noise cancellation.",
    },
    "smart watch": {
        "price": 199.99,
        "rating": 4.3,
        "description": "Tracks heart rate and sleep. 5-day battery, water-resistant.",
    },
    "mechanical keyboard": {
        "price": 129.00,
        "rating": 4.8,
        "description": "Tenkeyless, Cherry MX Brown switches, per-key RGB.",
    },
    "laptop stand": {
        "price": 34.99,
        "rating": 4.5,
        "description": "Adjustable aluminium, fits 11-17 inch laptops, folds flat.",
    },
}

REVIEWS = {
    "wireless headphones": {"reviews": 1262, "rating": 4.6},
    "smart watch":         {"reviews": 340,  "rating": 3.9},
    "mechanical keyboard": {"reviews": 67,   "rating": 4.8},
    "laptop stand":        {"reviews": 781,  "rating": 4.5},
}

@tool
def get_review(name: str) -> str:
    """Look up a product review by name. Returns the number of reviews and rating.
    Available products: wireless headphones, smart watch, mechanical keyboard, laptop stand."""
    r = REVIEWS.get(name.lower())
    if not r:
        return f"Review not available for '{name}'. Available: {', '.join(REVIEWS)}"
    return str(r)

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, rating, stock, and description."""
    p = PRODUCTS.get(name.lower())
    if not p:
        return f"Product not found. Available: {', '.join(PRODUCTS)}"
    return str(p)


agent = create_agent(
    model,
    tools=[get_product, get_review],
    system_prompt="You are a helpful product assistant for an online tech store.",
    checkpointer=MemorySaver(),
)


def ask(question: str) -> str:
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    print(response["messages"][-1].content)

ask("How do people like smart watches?")
ask("What is the price of this item?")