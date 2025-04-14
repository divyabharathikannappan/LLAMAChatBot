import json
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from interaction import interact_with_user

# === Logging setup ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grants-assistant")

# === FastAPI setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Grants Assistant is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("grants_chat_ui.html")
    if not html_path.exists():
        return HTMLResponse("<h2>UI file not found.</h2>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)

# === Global state ===
chat_sessions = {}

# === Load models ===
try:
    logger.info("Loading interaction LLM...")
    interaction_llm = Ollama(model="llama3", temperature=0.1)
    logger.info("Interaction LLM loaded")

    logger.info("Loading response LLM...")
    response_llm = Ollama(model="llama3", temperature=0.1)
    logger.info("Response LLM loaded")
except Exception as e:
    logger.error(f"Error loading LLMs: {e}")
    raise

# === Load Chroma vector DB ===
try:
    logger.info("Initializing embedding function...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    logger.info("Loading Chroma DB...")
    doc_db = Chroma(persist_directory="./embeddings", embedding_function=embeddings)
    logger.info("Chroma DB loaded with %d documents", doc_db._collection.count())
except Exception as e:
    logger.error(f"Failed to load Chroma DB: {e}")
    raise

# === System Prompts ===
interaction_system_prompt = """
You are a helpful grants assistant. Your goal is to help users find relevant grants by collecting key information.
"""

response_system_prompt = """
You are a grants assistant. Provide matching grants using the retrieved context.
"""

# === Dummy RAG logic for now ===
async def retrieval_answer(session_id: str, query: str, country: str = "canada"):
    logger.info(f"[{session_id}] Performing RAG on: {query}")
    retriever = doc_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    combined = "\n\n".join([doc.page_content for doc in docs]) or "No grants found."
    yield json.dumps({"data": combined})

# === Chat endpoint ===
@app.get("/chat_stream/grants/{session_id}/{country}/{query}")
def chat_stream(session_id: str, country: str, query: str):
    logger.info(f"[{session_id}] Incoming query: {query}")
    return StreamingResponse(
        interact_with_user(
            session_id=session_id,
            query=query,
            chat_sessions=chat_sessions,
            interaction_llm=interaction_llm,
            response_llm=response_llm,
            doc_db=doc_db,
            interaction_system_prompt=interaction_system_prompt,
            response_system_prompt=response_system_prompt,
            retrieval_answer_fn=retrieval_answer
        ),
        media_type="text/event-stream"
    )