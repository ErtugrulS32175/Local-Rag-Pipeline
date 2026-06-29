import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import ollama

load_dotenv()

# --- Config ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_oyak")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen3:8b")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
TOP_K           = 15
TOP_RERANK      = 10

# --- Init ---
client       = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

print("Loading reranker...")
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
reranker_model     = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
reranker_model.eval()
if torch.cuda.is_available():
    reranker_model = reranker_model.cuda()
print("Reranker loaded.")

def embed_dense(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]

def embed_sparse(text: str):
    result = list(sparse_model.embed([text]))[0]
    return result.indices.tolist(), result.values.tolist()

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    dense_vector = embed_dense(query)
    sparse_indices, sparse_values = embed_sparse(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                limit=top_k,
            ),
            Prefetch(
                query={"indices": sparse_indices, "values": sparse_values},
                using="sparse",
                limit=top_k,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results.points]

def rerank(query: str, chunks: list[dict], top_n: int = TOP_RERANK) -> list[dict]:
    pairs = [[query, chunk["text"]] for chunk in chunks]
    inputs = reranker_tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze(-1).cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]
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

def ask(question: str) -> str:
    chunks  = retrieve(question)
    chunks  = rerank(question, chunks)
    context = build_context(chunks)

    prompt = f"""Aşağıdaki kaynak bilgilere dayanarak soruyu Türkçe olarak cevapla.
SADECE kaynaklarda açıkça belirtilen bilgileri kullan.
Kaynaklarda olmayan hiçbir bilgiyi ekleme veya tahmin etme.
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

# --- Interactive loop ---
print("RAG Pipeline hazır. Çıkmak için 'quit' yaz.\n")
while True:
    question = input("Soru: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue
    print("\nAranıyor ve rerank ediliyor...")
    answer = ask(question)
    print(f"\nCevap:\n{answer}\n")
    print("-" * 60)