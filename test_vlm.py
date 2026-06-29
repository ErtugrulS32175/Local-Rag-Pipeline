import torch
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import json

print(f"CUDA: {torch.cuda.is_available()}")

pdf_path = r"C:\Users\ertug\Desktop\RAGtest\2024.pdf"
output_dir = Path(r"C:\Users\ertug\Desktop\RAGtest\output")
output_dir.mkdir(parents=True, exist_ok=True)

# SmolDocling VLM pipeline
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

print("Parsing PDF with SmolDocling...")
result = converter.convert(pdf_path)
print("Parsing complete.")

# Chunk
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
    max_tokens=512
)
chunker = HybridChunker(tokenizer=tokenizer)
chunks = list(chunker.chunk(result.document))
print(f"Total chunks: {len(chunks)}")

# İlk 3 chunk
print("\n--- First 3 chunks ---")
for i, chunk in enumerate(chunks[:3]):
    page_no = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else 0
    print(f"\nChunk {i+1}:")
    print(f"  Page: {page_no}")
    print(f"  Headings: {chunk.meta.headings}")
    print(f"  Text: {chunk.text[:200]}")

# HEKTAŞ var mı kontrol et
print("\n--- HEKTAŞ check ---")
for chunk in chunks:
    if "HEKTA" in chunk.text or "HEKTS" in chunk.text:
        print(chunk.text[:300])
        print("---")