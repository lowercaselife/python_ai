import sys
import os
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter

load_dotenv()

st.set_page_config(page_title="Blood Work Analyzer", page_icon="🩺", layout="wide")

st.title("🩺 Blood Work Analyzer")
st.caption("Upload a blood report to get an AI-powered analysis and personalized Indian diet plan.")

@st.cache_resource
def get_llm():
    return ChatOpenRouter(model="~anthropic/claude-haiku-latest")

def run_analysis(report_text: str):
    llm = get_llm()

    extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{report_text}
"""
    extraction_response = llm.invoke(extraction_prompt)
    extracted_values = extraction_response.text

    diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.
   Do not include any other sections in diet plan.

Blood Work Analysis:
{extracted_values}
"""
    diet_response = llm.invoke(diet_prompt)

    return extracted_values, diet_response.text


# --- Sidebar: report input ---
st.sidebar.header("Blood Report Input")
input_mode = st.sidebar.radio("Choose input method", ["Upload .txt file", "Paste report text"])

report_text = None

if input_mode == "Upload .txt file":
    uploaded = st.sidebar.file_uploader("Upload blood report (.txt)", type=["txt"])
    if uploaded:
        report_text = uploaded.read().decode("utf-8")
else:
    pasted = st.sidebar.text_area("Paste blood report here", height=300)
    if pasted.strip():
        report_text = pasted

# Default to the bundled sample if nothing provided
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "..", "blood_work.txt")
if report_text is None and os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "r") as f:
        report_text = f.read()
    st.sidebar.info("Using the bundled sample report (blood_work.txt).")

# --- Main area ---
if report_text:
    with st.expander("📄 Report Preview", expanded=False):
        st.text(report_text)

    if st.button("Analyze Report", type="primary", use_container_width=True):
        with st.spinner("Running analysis — this may take a moment…"):
            try:
                extracted, diet_plan = run_analysis(report_text)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("🔬 Extracted Test Values")
            lines = extracted.strip().splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if "HIGH" in line.upper():
                    st.markdown(f":red[{line}]")
                elif "LOW" in line.upper():
                    st.markdown(f":orange[{line}]")
                elif "NORMAL" in line.upper():
                    st.markdown(f":green[{line}]")
                else:
                    st.markdown(line)

        with col2:
            st.subheader("🥗 Health Summary & Diet Plan")
            st.markdown(diet_plan)
else:
    st.info("Provide a blood report via the sidebar to get started.")
