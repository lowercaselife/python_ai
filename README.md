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
│   ├── blood_work_analysis.ipynb  # Jupyter notebook version of the analysis
│   └── streamlit_app/
│       └── app.py                 # Streamlit web frontend for the blood work analyzer
├── RAG/
│   ├── telecom_guide.pdf          # Source document for RAG (internal telecom reference)
│   ├── rag_demo.py                # Full RAG pipeline — load, chunk, embed, retrieve, answer
│   ├── rag_demo.ipynb             # Jupyter notebook version of the RAG demo
│   └── chroma_db/                 # Persisted Chroma vector store (auto-created on first run)
├── product_query_agent/
│   └── agent.py                   # Multi-tool agent with conversation memory (MemorySaver)
├── multimodal_agent/
│   ├── blood_work.png             # Sample bloodwork image (input)
│   └── multimodal.py              # Multimodal agent — reads bloodwork image, calls diet tool, streams output
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

### 4. Multi-Tool Agent with Conversation Memory (`product_query_agent/agent.py`)

A stateful agent that can look up product details and customer reviews across a multi-turn conversation, remembering context between questions.

**Tools:**
- `get_product(name)` — returns price, rating, and description from the `PRODUCTS` dict
- `get_review(name)` — returns review count and rating from the `REVIEWS` dict

Both tools list available product names in their docstrings and fallback messages so the agent can self-correct if it passes an unrecognised name.

**Conversation memory** is enabled via `MemorySaver` and a `thread_id`, allowing follow-up questions like *"What is the price of this item?"* to resolve *"this item"* from the previous turn:

```python
from langgraph.checkpoint.memory import MemorySaver

agent = create_agent(
    model,
    tools=[get_product, get_review],
    system_prompt="You are a helpful product assistant for an online tech store.",
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
```

### 5. Multimodal Agent with Streaming — Blood Work Analyser (`multimodal_agent/multimodal.py`)

A multimodal agentic pipeline that reads a blood work image and produces personalised dietary recommendations, with streamed output to reduce perceived latency.

**Flow:**

1. The blood work image is base64-encoded and passed to the agent as a multimodal message
2. The LLM extracts all test results from the image and flags any values outside the normal range
3. The agent calls the `get_diet_recommendation` tool with the extracted report
4. A second LLM call inside the tool generates evidence-based dietary recommendations
5. The final response streams to the terminal token-by-token via `AIMessageChunk`

**`get_diet_recommendation` tool:**

```python
@tool
def get_diet_recommendation(bloodwork_report: str) -> str:
    """Generate personalized diet recommendations based on a patient's extracted bloodwork report."""
    rec_model = ChatOpenRouter(model="~anthropic/claude-haiku-latest")
    response = rec_model.invoke(f"...{bloodwork_report}...")
    return response.content
```

**Streaming** uses LangGraph's `stream_mode="messages"`, filtering on `AIMessageChunk` to print only the LLM's text tokens as they arrive:

```python
for chunk, metadata in agent.stream({"messages": [message]}, config=config, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)
```

```bash
cd multimodal_agent
uv run python multimodal.py
```

---

### 6. Applied LLM Pipeline — Health Analysis (`health_analysis/`)

A real-world multi-stage pipeline that reads a blood work report and extracts structured insights:

- **Stage 1:** Extract all test values and classify each as HIGH / LOW / NORMAL
- **Stage 2:** Generate a plain-language health summary and an Indian diet plan (foods to avoid / eat more of)
- Demonstrates reading files, crafting domain-specific prompts, and chaining LLM calls

### 7. Streamlit Web Frontend (`health_analysis/streamlit_app/app.py`)

An interactive browser UI wrapping the two-stage health analysis pipeline:

- Upload a `.txt` blood report or paste one directly in the sidebar; falls back to the bundled `blood_work.txt` sample automatically
- Results displayed side-by-side: color-coded test values (red = HIGH, orange = LOW, green = NORMAL) alongside the health summary and diet plan
- Stateless single-page app — no database required

```bash
uv run streamlit run health_analysis/streamlit_app/app.py
```

### 8. Retrieval-Augmented Generation — RAG (`RAG/rag_demo.py`)

A complete RAG pipeline that lets an LLM answer questions grounded strictly in a PDF document, preventing hallucination by limiting it to retrieved context only.

**Pipeline stages:**

**① Load** — reads `telecom_guide.pdf` using `PyPDFLoader`, producing one `Document` per page.

**② Chunk** — splits pages into overlapping chunks with `RecursiveCharacterTextSplitter`:
```python
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
```

**③ Embed & store** — converts chunks to vectors using `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) and stores them in a **persisted Chroma vector store**. A guard prevents re-embedding on subsequent runs:
```python
if os.path.exists(DB_PATH):
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
```

**④ Retrieve** — a retriever fetches the top-3 most semantically similar chunks for any query:
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

**⑤ Answer via chain** — a LangChain LCEL chain wires everything together with the `|` pipe operator:
```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

| Step | Component | Role |
|---|---|---|
| Input | `retriever \| format_docs` | Embed query → fetch top-3 chunks → join into one string |
| Input | `RunnablePassthrough()` | Pass the original question through unchanged |
| Prompt | `ChatPromptTemplate` | Inject context + question into system/human messages |
| LLM | `ChatOpenRouter` (Claude Haiku) | Answer using only the provided context |
| Output | `StrOutputParser` | Strip `AIMessage` wrapper → return plain string |

---

## Models

The project uses [OpenRouter](https://openrouter.ai/models) to access multiple model providers through a single API. Models used:

| Model | Use case | Notes |
|---|---|---|
| `qwen/qwen3.6-flash` | Default — fast iteration | Low latency, very economical |
| `moonshotai/kimi-k2.6` | High-quality responses | Large MoE model, slower (~30s) |
| `google/gemma-4-26b-a4b-it:free` | Free tier fallback | Rate-limited under load |
| `~anthropic/claude-haiku-latest` | Multimodal agent & diet tool | Supports vision input; used for image-based bloodwork analysis |
| `anthropic/claude-haiku-latest` | Health analysis & RAG pipeline | Fast, instruction-following, cost-effective |

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
| `streamlit` | Web frontend for the blood work analyzer |
| `langchain-community` | Document loaders (PyPDFLoader) — being sunset, use standalone packages for new work |
| `langchain-huggingface` | HuggingFace embeddings integration |
| `langchain-chroma` | Chroma vector store integration |
| `sentence-transformers` | Local embedding model (`all-MiniLM-L6-v2`) |

---

## Running the Examples

```bash
# Basic model call and prompt templates
python main.py

# Agent example
python agent.py

# Health analysis pipeline (CLI)
cd health_analysis
python blood_work_analysis.py

# Health analysis — Streamlit web UI
uv run streamlit run health_analysis/streamlit_app/app.py

# RAG pipeline
uv run python RAG/rag_demo.py

# Product query agent (multi-tool, with memory)
uv run python product_query_agent/agent.py

# Multimodal bloodwork agent (streamed output)
cd multimodal_agent
uv run python multimodal.py
```
