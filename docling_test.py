import torch
import json
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

print(torch.cuda.is_available())
print(torch.version.cuda)

pdf_path = r"C:\Users\ertug\Desktop\RAGtest\2024.pdf"
output_dir = Path(r"C:\Users\ertug\Desktop\RAGtest\output")
images_dir = output_dir / "images"
images_dir.mkdir(parents=True, exist_ok=True)
metadata_path = output_dir / "picture_metadata.json"

# Stage 1: Parse text + tables, save pictures to disk
pipeline_options = ThreadedPdfPipelineOptions(
    accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
    table_batch_size=2,
    layout_batch_size=8,
)
pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 1.0

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=ThreadedStandardPdfPipeline,
            pipeline_options=pipeline_options,
        )
    }
)

print("Stage 1: Parsing PDF...")
result = converter.convert(pdf_path)
print("Parsing complete.")

# Load existing metadata to preserve descriptions from Stage 2
existing_metadata = {}
if metadata_path.exists():
    with open(metadata_path, "r", encoding="utf-8") as f:
        existing = json.load(f)
        existing_metadata = {item["index"]: item for item in existing}
    print(f"Loaded existing metadata: {len(existing_metadata)} entries")

# Save pictures to disk and store metadata
picture_metadata = []
for i, pic in enumerate(result.document.pictures):
    if pic.image and pic.image.pil_image:
        page_no = pic.prov[0].page_no if pic.prov else 0
        img_path = images_dir / f"pic_{i:03d}_page{page_no}.png"
        pic.image.pil_image.save(img_path)

        # Preserve existing description if available
        existing_desc = existing_metadata.get(i, {}).get("description", "")

        picture_metadata.append({
            "index": i,
            "page": page_no,
            "image_path": str(img_path),
            "caption": pic.caption_text(doc=result.document) or "",
            "description": existing_desc
        })

print(f"Saved {len(picture_metadata)} pictures.")

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(picture_metadata, f, ensure_ascii=False, indent=2)
print(f"Metadata saved: {metadata_path}")

# Chunk the document
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
    max_tokens=512
)
chunker = HybridChunker(tokenizer=tokenizer)
chunks = list(chunker.chunk(result.document))
print(f"Total chunks: {len(chunks)}")

# Save chunks to disk — merge consecutive table chunks with same heading
chunks_data = []
i = 0
while i < len(chunks):
    chunk = chunks[i]

    # Determine chunk type
    chunk_type = "text"
    if chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if "table" in str(item.label).lower():
                chunk_type = "table"
                break

    page_no = chunk.meta.doc_items[0].prov[0].page_no if chunk.meta.doc_items else 0
    headings = chunk.meta.headings or []

    if chunk_type == "table":
        # Merge consecutive table chunks with same heading
        merged_text = chunk.text

        while i + 1 < len(chunks):
            next_chunk = chunks[i + 1]
            next_type = "text"
            if next_chunk.meta.doc_items:
                for item in next_chunk.meta.doc_items:
                    if "table" in str(item.label).lower():
                        next_type = "table"
                        break

            if next_type == "table" and next_chunk.meta.headings == headings:
                merged_text += "\n" + next_chunk.text
                i += 1
            else:
                break

        chunks_data.append({
            "type": "table",
            "text": merged_text,
            "page": page_no,
            "headings": headings,
            "source": str(pdf_path),
        })
    else:
        chunks_data.append({
            "type": chunk_type,
            "text": chunk.text,
            "page": page_no,
            "headings": headings,
            "source": str(pdf_path),
        })

    i += 1

chunks_path = output_dir / "chunks.json"
with open(chunks_path, "w", encoding="utf-8") as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)
print(f"Chunks saved: {chunks_path} ({len(chunks_data)} chunks)")