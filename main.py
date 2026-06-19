from fastapi import FastAPI
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage

load_dotenv()

OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b"

app = FastAPI()


class IngestRequest(BaseModel):
    pdf_path: str
    source_id: str | None = None


@app.post("/ingest")
async def ingest_direct(req: IngestRequest):
    source_id = req.source_id or req.pdf_path
    chunks = load_and_chunk_pdf(req.pdf_path)
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage().upsert(ids, vecs, payloads)
    return {"ingested": len(chunks)}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    source: str | None = None


@app.post("/query")
async def query_direct(req: QueryRequest):
    query_vec = embed_texts([req.question])[0]
    store = QdrantStorage()
    found = store.search(query_vec, top_k=req.top_k, source=req.source)

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {req.question}\n"
        "Answer concisely using the context above."
    )
    resp = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        timeout=500,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ],
            "stream": False,
            "options": {
                "num_predict": 1024,
                "temperature": 0.2,
            },
        },
    )
    resp.raise_for_status()
    answer = resp.json()["message"]["content"].strip()
    return {"answer": answer, "sources": found["sources"], "num_contexts": len(found["contexts"])}


@app.get("/sources")
async def list_sources():
    store = QdrantStorage()
    return {"sources": store.list_sources()}
