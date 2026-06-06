import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openrouter import ChatOpenRouter

load_dotenv()


PDF_PATH = os.path.join(os.path.dirname(__file__), "telecom_guide.pdf")

loader = PyPDFLoader(PDF_PATH)
pages = loader.load()

# print(f"Loaded {len(pages)} pages from the PDF.")
# print("\n--- First page preview (first 500 chars) ---")
# print(pages[0].page_content[:500])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,       # ~150 words per chunk
    chunk_overlap=100,    # overlap keeps context at boundaries
    separators=["\n\n", "\n", ".", " "],  # tries paragraph → line → sentence → word
)

chunks = splitter.split_documents(pages)

# print(f"Total chunks: {len(chunks)}  (from {len(pages)} pages)")


# print(f"Avg chunk length: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
# print("\n--- Example chunk ---")
# print(chunks[5].page_content)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

if os.path.exists(DB_PATH):
    print("Loading existing vector store...")
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
else:
    print("Building vector store for the first time...")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)

# print(f"Vector store ready. {vector_store._collection.count()} vectors stored.")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# test_query = "What is VoLTE and how does it improve call quality?"
# retrieved = retriever.invoke(test_query)

# print(f"Query: {test_query}")
# print(f"Retrieved {len(retrieved)} chunks:\n")
# for i, doc in enumerate(retrieved, 1):
#     print(f"--- Chunk {i} ---")
#     print(doc.page_content[:300])
#     print()

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# --- System prompt: ground the LLM in the retrieved context ---
SYSTEM_PROMPT = """\
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

llm = ChatOpenRouter(model="~anthropic/claude-haiku-latest", temperature=1.0)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

question = "How does international roaming work and what charges should I expect?"

print(f"Q: {question}\n")
print("A:", chain.invoke(question))