import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from FlagEmbedding import FlagReranker
import ollama

load_dotenv()

# --- Config ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_oyak")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen3:8b")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
TOP_K           = 10
TOP_RERANK      = 5

# --- Init ---
app      = FastAPI()
client   = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)

# --- Request model ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K

# --- Helpers ---
def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

def retrieve(query: str, top_k: int) -> list[dict]:
    query_vector = embed(query)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results]

def rerank(query: str, chunks: list[dict], top_n: int = TOP_RERANK) -> list[dict]:
    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = reranker.compute_score(pairs, normalize=True)
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_n]]

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("type", "text")
        page       = chunk.get("page", "?")
        headings   = " > ".join(chunk.get("headings", [])) or "No heading"
        text       = chunk.get("text", "")
        parts.append(
            f"[Kaynak {i+1} | Tip: {chunk_type} | Sayfa: {page} | Bölüm: {headings}]\n{text}"
        )
    return "\n\n---\n\n".join(parts)

def generate(question: str, context: str) -> str:
    prompt = f"""Aşağıdaki kaynak bilgilere dayanarak soruyu Türkçe olarak cevapla.
Cevabında hangi kaynaktan aldığını belirt (örn: Kaynak 1, Sayfa 13).
Eğer cevap kaynaklarda yoksa "Bu bilgi mevcut belgelerde bulunamadı." de.

KAYNAKLAR:
{context}

SORU: {question}

CEVAP:"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(request: QueryRequest):
    chunks  = retrieve(request.question, request.top_k)
    chunks  = rerank(request.question, chunks)
    context = build_context(chunks)
    answer  = generate(request.question, context)

    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "type": c.get("type"),
                "page": c.get("page"),
                "headings": c.get("headings"),
                "text": c.get("text", "")[:200],
            }
            for c in chunks
        ]
    }