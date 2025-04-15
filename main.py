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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grants-assistant")

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

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse("<h2>UI file not found.</h2>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)

chat_sessions = {}

interaction_llm = Ollama(model="llama3", temperature=0.1)
response_llm = Ollama(model="llama3", temperature=0.1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
doc_db = Chroma(persist_directory="./embeddings", embedding_function=embeddings)

interaction_system_prompt = "You are a helpful grants assistant. Ask questions to collect business type, location, and funding purpose. Then generate a SEARCH QUERY."
response_system_prompt = "You are a grants assistant chatbot. Recommend grants based on context and query."
async def retrieval_answer(session_id: str, query: str, country: str = "canada"):
    logger = logging.getLogger("grants-assistant")

    logger.info(f"[{session_id}] Performing RAG on: {query}")
    retriever = doc_db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query)
    logger.info(f"[{session_id}] Retrieved {len(docs)} document(s)")

    if not docs:
        yield "Sorry, I couldn’t find any relevant grants for that search.\n\n"
        yield "[DONE]\n\n"
        return

    for doc in docs:
        content = doc.page_content.strip().replace("\n", " ")
        if not content.startswith("Grant Name:"):
            content = "Grant Name: Unnamed Grant " + content
        logger.info(f"[{session_id}] Grant doc: {content[:200]}...")
        yield f"data: {content}\n\n"

    yield "[DONE]\n\n"
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