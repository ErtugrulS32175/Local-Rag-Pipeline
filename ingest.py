import json
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseIndexParams
)
from qdrant_client.http.models import SparseVector
from fastembed import SparseTextEmbedding
import ollama
import uuid

load_dotenv()

# --- Config ---
OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR", r"C:\Users\ertug\Desktop\RAGtest\output"))
CHUNKS_PATH     = OUTPUT_DIR / "chunks.json"
METADATA_PATH   = OUTPUT_DIR / "picture_metadata.json"
PDF_PATH        = os.getenv("PDF_PATH", r"C:\Users\ertug\Desktop\RAGtest\2024.pdf")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_oyak")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))

# --- Init ---
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# --- Recreate collection with both dense and sparse vectors ---
if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted existing collection: {COLLECTION_NAME}")

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
        )
    }
)
print(f"Created collection: {COLLECTION_NAME}")

# --- Load chunks ---
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks_data = json.load(f)
print(f"Loaded {len(chunks_data)} text/table chunks.")

# --- Load picture metadata ---
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    picture_metadata = json.load(f)

image_chunks = [
    p for p in picture_metadata
    if p["description"] not in ("", "SKIP", None)
    and not p["description"].startswith("Error")
]
print(f"Loaded {len(image_chunks)} image chunks.")

# --- Helpers ---
def embed_dense(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text[:2000])
    return response["embedding"]

def embed_sparse(text: str) -> SparseVector:
    result = list(sparse_model.embed([text]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist()
    )

def upsert_batch(points: list[PointStruct]):
    client.upsert(collection_name=COLLECTION_NAME, points=points)

# --- Ingest text/table chunks ---
print("\nIngesting text/table chunks...")
batch = []
for i, chunk in enumerate(chunks_data):
    dense  = embed_dense(chunk["text"])
    sparse = embed_sparse(chunk["text"])
    batch.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense,
            "sparse": sparse,
        },
        payload=chunk,
    ))

    if len(batch) >= 32:
        upsert_batch(batch)
        batch = []
        print(f"  {i+1}/{len(chunks_data)} chunks ingested...")

if batch:
    upsert_batch(batch)
print("Text/table ingestion complete.")

# --- Ingest image chunks ---
print("\nIngesting image chunks...")
batch = []
for i, pic in enumerate(image_chunks):
    dense  = embed_dense(pic["description"])
    sparse = embed_sparse(pic["description"])
    batch.append(PointStruct(
        id=str(uuid.uuid4()),
        vector={
            "dense": dense,
            "sparse": sparse,
        },
        payload={
            "type": "image",
            "text": pic["description"],
            "page": pic["page"],
            "headings": [],
            "caption": pic["caption"],
            "image_path": pic["image_path"],
            "source": PDF_PATH,
        }
    ))

    if len(batch) >= 32:
        upsert_batch(batch)
        batch = []
        print(f"  {i+1}/{len(image_chunks)} image chunks ingested...")

if batch:
    upsert_batch(batch)
print("Image ingestion complete.")

# --- Summary ---
info = client.get_collection(COLLECTION_NAME)
print(f"\nCollection '{COLLECTION_NAME}' ready.")
print(f"Total vectors: {info.points_count}")