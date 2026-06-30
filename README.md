# Local Enterprise RAG Pipeline

A fully local, offline RAG (Retrieval-Augmented Generation) pipeline for querying PDF documents — built for enterprise-style document intelligence with no data leaving your machine.

Optimized for Turkish-language documents, with hybrid search, reranking, and multimodal (text + table + image) understanding.

## Architecture

PDF
- Text & Tables -> Docling (DocLayNet + TableFormer) -> HybridChunker
- Images -> Docling (extraction) -> qwen2.5vl:7b (description)

Both paths converge into:
- bge-m3 (dense embedding) + BM25 (sparse embedding)
- Qdrant (hybrid search)
- bge-reranker-v2-m3 (reranking)
- qwen3:8b (answer generation)
- FastAPI -> OpenWebUI (chat interface)

## Features

- **Fully Offline** — no data sent to external servers
- **Multimodal Parsing** — text, tables, and images all understood and indexed
- **Hybrid Search** — combines dense (semantic) and sparse (BM25 keyword) retrieval via Reciprocal Rank Fusion
- **Reranking** — cross-encoder reranker improves retrieval precision
- **Page-Level Source Citation** — every answer references the exact page it came from
- **Turkish-First** — tuned prompts and multilingual models for Turkish documents
- **OpenWebUI Integration** — chat interface via a custom pipe

## Tech Stack

| Component | Technology |
|---|---|
| Document parsing | Docling (DocLayNet + TableFormer) |
| Chunking | Docling HybridChunker |
| Image understanding | qwen2.5vl:7b (Ollama) |
| Embedding (dense) | bge-m3 (Ollama) |
| Embedding (sparse) | BM25 (FastEmbed) |
| Vector DB | Qdrant (Docker, hybrid vectors) |
| Reranker | BAAI/bge-reranker-v2-m3 |
| LLM | qwen3:8b (Ollama) |
| Backend | FastAPI |
| UI | OpenWebUI |

## Requirements

- Windows with NVIDIA GPU (tested on RTX 4070 Laptop, 8GB VRAM) + CUDA-enabled PyTorch
- Docker
- Ollama
- Python 3.10+ (Anaconda recommended)
- 32GB RAM recommended

## Example

This pipeline was built and tested using OYAK's publicly available [2024 Annual Report](https://www.oyak.com.tr) (Turkish, 108 pages, text + tables + charts). The pipe in OpenWebUI is named "OYAK RAG Pipeline" in this example setup — rename it as needed and works with any PDF.

## Setup

### 1. Start Qdrant
```bash
docker run -p 6333:6333 -v C:\qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 2. Pull Ollama Models
```bash
ollama pull qwen3:8b
ollama pull bge-m3
ollama pull qwen2.5vl:7b
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env` and adjust paths/collection name as needed.

### 5. Ingest a PDF

```bash
# Stop image model conflicts during table/layout parsing
ollama stop qwen2.5vl:7b

# Parse PDF: extract text, tables, and images
python docling_test.py

# Describe extracted images with the vision model
python docling_stage2.py

# Embed everything and upsert into Qdrant
python ingest.py
```

> **Note:** Steps 1 (`docling_test.py`) and 2 (`docling_stage2.py`) both use GPU-bound models. On 8GB VRAM, running Docling's table parser and the vision model simultaneously can cause out-of-memory errors. Run them sequentially, stopping any unused Ollama model in between (`ollama stop <model>`).

### 6. Query The Pipeline

**Terminal:**
```bash
python query.py
```

**API:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**OpenWebUI:**
1. Start OpenWebUI: `docker start open-webui`
2. Go to `http://localhost:3000`
3. Add a custom Pipe function pointing to `http://host.docker.internal:8000/query`
4. Select it in the chat and start asking questions

## How It Works

1. **Parsing** — Docling extracts text, tables, and images from the PDF using layout detection (DocLayNet) and table structure recognition (TableFormer). Table chunks belonging to the same logical table are merged to avoid fragmenting structured data.
2. **Image understanding** — Each extracted image is sent to a local vision-language model (qwen2.5vl:7b), which either describes its content (charts, infographics, photos) or flags it as decorative (logos, signatures) to skip.
3. **Chunking** — Docling's `HybridChunker` splits text and tables into semantically coherent chunks, preserving heading hierarchy and page numbers.
4. **Indexing** — Each chunk is embedded twice: once with `bge-m3` (dense, semantic) and once with BM25 (sparse, keyword). Both vectors are stored in the same Qdrant point.
5. **Retrieval** — A query is embedded the same way, and Qdrant performs hybrid search using Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
6. **Reranking** — The top candidates are reranked with a cross-encoder (`bge-reranker-v2-m3`) for higher precision.
7. **Generation** — The reranked passages are passed to `qwen3:8b`, which is instructed to answer strictly from the provided context and cite the page number.

## Enterprise Notes

This pipeline mirrors a production enterprise RAG architecture, with the following hardware-driven substitutions:

| Production Component | Local Equivalent | Why |
|---|---|---|
| vLLM (multi-user serving) | Ollama | Single-user, single-GPU setup |
| GPU cluster | Single RTX 4070 Laptop (8GB VRAM) | Local development hardware |
| Managed vector DB | Self-hosted Qdrant (Docker) | No cloud dependency |
| GraniteDocling / full VLM pipeline | Classical Docling pipeline + separate VLM step | 8GB VRAM cannot run a full-page VLM pipeline efficiently; this two-stage approach (DocLayNet/TableFormer for structure, qwen2.5vl for images) achieves comparable results in a fraction of the time on consumer hardware |

A GPU-cloud version of this pipeline (using GraniteDocling/SmolDocling and Qdrant Cloud, tested on rented GPU infrastructure) is planned as a follow-up project for environments with more VRAM.

## Known Limitations

- Running Docling's table/layout models concurrently with another GPU-resident model (e.g. the vision model) on 8GB VRAM can cause `std::bad_alloc` errors. Always stop unused Ollama models before running `docling_test.py`.
- Ingestion is a one-time, offline batch process — not designed to run on every query.
- Vision model descriptions are generated in English for consistency; the multilingual embedding model (`bge-m3`) bridges this with Turkish queries without quality loss.

## License

MIT
