import streamlit as st
import requests
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

# Connections
qdrant_local = QdrantClient("localhost", port=6333)
reranker_local = CrossEncoder("BAAI/bge-reranker-v2-m3")
COLLECTION_LOCAL = "rag_local"

def embed_local(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "bge-m3", "input": [text]}
    )
    return response.json()["embeddings"][0]

def rag_query(question, top_k=30, top_n=5):
    query_vector = embed_local(question)

    hits = qdrant_local.query_points(
        collection_name=COLLECTION_LOCAL,
        query=query_vector,
        limit=top_k
    ).points

    documents = [hit.payload["text"] for hit in hits]
    pairs = [[question, doc] for doc in documents]
    scores = reranker_local.predict(pairs)

    ranked = sorted(zip(scores, hits), reverse=True)[:top_n]
    top_docs = [hit.payload["text"] for _, hit in ranked]
    sources = [hit.payload["page"] for _, hit in ranked]

    context = "\n\n".join(top_docs)

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen3:8b",
            "messages": [
                {"role": "system", "content": "Sadece verilen bağlama göre Türkçe cevap ver. Cevap bağlamda yoksa 'Bilmiyorum' de."},
                {"role": "user", "content": f"Bağlam:\n{context}\n\nSoru: {question}"}
            ],
            "stream": False,
            "options": {"temperature": 0},
            "think": False
        }
    )

    answer = response.json()["message"]["content"]
    return answer, sources

# Streamlit UI
st.title("OYAK RAG Sistemi")
st.caption("Tamamen lokal çalışan RAG pipeline")

question = st.text_input("Sorunuzu yazın:")

if st.button("Sor") and question:
    with st.spinner("Cevap aranıyor..."):
        answer, sources = rag_query(question)

    st.markdown("### Cevap")
    st.write(answer)

    st.markdown("### Kaynak Sayfalar")
    st.write(f"Sayfa: {sources}")