# Local RAG Pipeline

A fully local, offline RAG (Retrieval-Augmented Generation) pipeline for querying PDF documents.
Optimized for Turkish language users — no data leaves your machine.

## Features

- Completely offline — no data sent to external servers
- Turkish language support with multilingual models
- Page-based and chunk-based indexing strategies
- High accuracy with reranker
- OpenWebUI and Streamlit interfaces

## Tech Stack

| Component | Technology |
|---|---|
| LLM | qwen3:8b (Ollama) |
| Embedding | bge-m3 (Ollama) |
| Vector DB | Qdrant (Docker) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| UI | OpenWebUI + Streamlit |

## Requirements

- Docker
- Ollama
- Python 3.10+
- Anaconda

## Setup

1. Start Qdrant:
```bash
docker run -p 6333:6333 -v C:\qdrant_storage:/qdrant/storage qdrant/qdrant
```

2. Pull models:
```bash
ollama pull qwen3:8b
ollama pull bge-m3
```

3. Install dependencies:
```bash
pip install qdrant-client sentence-transformers streamlit requests python-dotenv
```

4. Download OYAK 2024 Annual Report PDF from [oyak.com.tr](https://www.oyak.com.tr) and place it in the project folder.

5. Run the Jupyter notebook to index the PDF into Qdrant.

## Usage

### Option 1: OpenWebUI (Recommended)
1. Start OpenWebUI:
```bash
docker start open-webui
```
2. Go to `http://localhost:3000`
3. Select "OYAK RAG Pipeline" from models
4. Start asking questions

### Option 2: Streamlit
```bash
streamlit run ragtest.py
```
Go to `http://localhost:8501`

## Notes

- Tested on Windows with RTX 4070 Laptop (8GB VRAM)
- qwen3:8b requires ~5GB VRAM
- bge-reranker-v2-m3 runs on CPU
- Qdrant data persists in `qdrant_storage` folder
