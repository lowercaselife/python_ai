from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import MemorySaver
import base64

load_dotenv()

model = ChatOpenRouter(model="~anthropic/claude-haiku-latest")


@tool
def get_diet_recommendation(bloodwork_report: str) -> str:
    """Generate personalized diet recommendations based on a patient's extracted bloodwork report.

    Call this tool after extracting all test results from the bloodwork image and identifying
    any values outside the normal range. Pass the complete analysis as input.

    Args:
        bloodwork_report: Full text summary of the patient's bloodwork results, including
                          all test values and any flagged abnormal findings.
    """
    rec_model = ChatOpenRouter(model="~anthropic/claude-haiku-latest")
    prompt = f"""You are a clinical nutritionist. Based on the following bloodwork report,
provide specific, evidence-based dietary recommendations to help the patient improve their health markers.

Bloodwork Report:
{bloodwork_report}

Provide practical recommendations organized by:
1. Foods to increase
2. Foods to reduce or avoid
3. Key nutrients to focus on
4. Lifestyle suggestions related to diet"""

    response = rec_model.invoke(prompt)
    return response.content


with open("blood_work.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

agent = create_agent(
    model,
    tools=[get_diet_recommendation],
    system_prompt=(
        "You are a medical assistant specializing in bloodwork analysis. "
        "When given a bloodwork image, first extract all test results and identify "
        "any values outside the normal range. Then call the get_diet_recommendation "
        "tool with your complete analysis to provide personalized dietary advice."
    ),
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "bloodwork-session-1"}}

message = HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    {"type": "text", "text": (
        "Analyze this bloodwork report. Extract all test results, flag any values "
        "outside the normal range, then use the get_diet_recommendation tool to "
        "provide personalized dietary recommendations for this patient."
    )},
])

for chunk, metadata in agent.stream({"messages": [message]}, config=config, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)

print()