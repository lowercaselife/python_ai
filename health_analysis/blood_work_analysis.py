from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


load_dotenv()

llm = ChatOpenRouter(model="qwen/qwen3.6-flash")

with open("blood_work.txt", "r") as f:
    blood_report = f.read()

# print(blood_report)

extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL 
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""

extraction_response = llm.invoke(extraction_prompt)
extracted_values = extraction_response.text

print("=== STAGE 1: EXTRACTED VALUES ===")
print(extracted_values)