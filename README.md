# Python AI — LangChain & LangGraph Learning Workspace

A hands-on workspace for learning [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/) using [OpenRouter](https://openrouter.ai/) as the model provider. The repo progresses from basic model calls through prompt templates, agents, and real-world applied examples.

---

## Project Structure

```
python_ai/
├── main.py                        # Entry point — basic model invocation & prompt templates
├── agent.py                       # LangChain agent with tool use (create_agent)
├── health_analysis/
│   ├── blood_work.txt             # Sample blood work report (input data)
│   ├── blood_work_analysis.py     # Multi-stage LLM pipeline for medical data extraction
│   └── blood_work_analysis.ipynb  # Jupyter notebook version of the analysis
├── .env                           # API keys (not committed)
├── pyproject.toml                 # Project metadata and pinned dependencies
└── uv.lock                        # Lockfile (managed by uv)
```

---

## Setup

### Prerequisites
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (package manager) — or use `pip` with the `.venv` directly
- An [OpenRouter](https://openrouter.ai/) account and API key

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd python_ai

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
# or, if using uv:
uv sync
```

### Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

> Get your key at [openrouter.ai/keys](https://openrouter.ai/keys). The project uses `python-dotenv` to load this automatically.

---

## Key Concepts Covered

### 1. Basic Model Invocation (`main.py`)

Calling an LLM and printing its response using `ChatOpenRouter`:

```python
from langchain_openrouter import ChatOpenRouter

model = ChatOpenRouter(model="qwen/qwen3.6-flash", temperature=0)
response = model.invoke("What is the capital of France?")
print(response.content)
```

### 2. Reusable Completion Function

A `get_completion` helper that mirrors the classic OpenAI SDK pattern:

```python
def get_completion(prompt, model=llm_model):
    llm = ChatOpenRouter(model=model, temperature=0)
    response = llm.invoke(prompt)
    return response.content
```

### 3. Prompt Templates (`main.py`)

Using `ChatPromptTemplate` to build reusable, parameterised prompts:

```python
from langchain_core.prompts import ChatPromptTemplate

template = """Translate the text delimited by triple backticks
into a style that is {style}.
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template)
messages = prompt_template.format_messages(style=customer_style, text=customer_email)
response = get_completion(messages)
```

### 4. Agents (`agent.py`)

Building a LangChain agent with `create_agent` — an LLM with a reasoning harness:

```python
from langchain.agents import create_agent
from langchain_openrouter import ChatOpenRouter

model = ChatOpenRouter(model="anthropic/claude-haiku-latest")
agent = create_agent(model)
response = agent.invoke({"messages": [HumanMessage(content="...")]})
```

### 5. Applied LLM Pipeline — Health Analysis (`health_analysis/`)

A real-world multi-stage pipeline that reads a blood work report and extracts structured insights:

- **Stage 1:** Extract all test values and classify each as HIGH / LOW / NORMAL
- Demonstrates reading files, crafting domain-specific prompts, and chaining LLM calls

---

## Models

The project uses [OpenRouter](https://openrouter.ai/models) to access multiple model providers through a single API. Models used:

| Model | Use case | Notes |
|---|---|---|
| `qwen/qwen3.6-flash` | Default — fast iteration | Low latency, very economical |
| `moonshotai/kimi-k2.6` | High-quality responses | Large MoE model, slower (~30s) |
| `google/gemma-4-26b-a4b-it:free` | Free tier fallback | Rate-limited under load |

To swap models, change the model string — all models share the same `ChatOpenRouter` interface:

```python
model = ChatOpenRouter(model="moonshotai/kimi-k2.6")
```

---

## Dependencies

Core packages (see `pyproject.toml` for full pinned list):

| Package | Purpose |
|---|---|
| `langchain` | Core framework — chains, prompts, memory |
| `langchain-core` | Base abstractions (messages, runnables) |
| `langchain-openrouter` | OpenRouter integration for `ChatOpenRouter` |
| `langchain-openai` | OpenAI-compatible adapter |
| `langgraph` | Graph-based agent and workflow orchestration |
| `python-dotenv` | Load `.env` into environment variables |
| `ipykernel` | Jupyter notebook support |

---

## Running the Examples

```bash
# Basic model call and prompt templates
python main.py

# Agent example
python agent.py

# Health analysis pipeline
cd health_analysis
python blood_work_analysis.py
```
