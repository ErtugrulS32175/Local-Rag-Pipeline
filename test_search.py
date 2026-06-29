import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from fastembed import SparseTextEmbedding

client = QdrantClient('localhost', port=6333)
sparse_model = SparseTextEmbedding(model_name='Qdrant/bm25')

query = 'OYAK halka acik sirketler HEKTAS'
dense = ollama.embeddings(model='bge-m3', prompt=query)['embedding']
sparse = list(sparse_model.embed([query]))[0]

results = client.query_points(
    collection_name='rag_oyak',
    prefetch=[
        Prefetch(query=dense, using='dense', limit=15),
        Prefetch(query={'indices': sparse.indices.tolist(), 'values': sparse.values.tolist()}, using='sparse', limit=15),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=15,
    with_payload=True,
)

print(f"Toplam sonuç: {len(results.points)}")
print()
for i, r in enumerate(results.points):
    text = r.payload.get('text', '')
    if 'HEKTA' in text or 'halka' in text.lower() or 'EREGL' in text or 'ISDMR' in text:
        print(f"--- Sonuç {i+1} ---")
        print(text[:400])
        print()