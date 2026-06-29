import json
import base64
import requests
from pathlib import Path

output_dir = Path(r"C:\Users\ertug\Desktop\RAGtest\output")
metadata_path = output_dir / "picture_metadata.json"

# Load metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    picture_metadata = json.load(f)

print(f"Total pictures to process: {len(picture_metadata)}")

def describe_image(image_path: str) -> str:
    """Send image to qwen2.5vl:7b via Ollama and get description or SKIP."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "qwen2.5vl:7b",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    },
                    {
                        "type": "text",
                        "text": (
                            "If this image is ONLY a logo, signature, or plain portrait photo with no data, "
                            "respond with exactly: SKIP\n"
                            "Otherwise describe it in English in three sentences, "
                            "including any numerical values if present."
                        )
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error: {e}"

# Process all pictures
skipped = 0
described = 0
errors = 0

for item in picture_metadata:
    # Skip already processed ones
    if item["description"] not in ("", None):
        continue

    print(f"Processing picture {item['index']} (page {item['page']})...", end=" ")
    description = describe_image(item["image_path"])

    if description == "SKIP":
        item["description"] = "SKIP"
        skipped += 1
        print("SKIP")
    elif description.startswith("Error"):
        item["description"] = ""
        errors += 1
        print(f"ERROR: {description}")
    else:
        item["description"] = description
        described += 1
        print(f"OK ({len(description)} chars)")

    # Save after every 10 pictures in case of interruption
    if (item["index"] + 1) % 10 == 0:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(picture_metadata, f, ensure_ascii=False, indent=2)
        print(f"--- Checkpoint saved ({item['index']+1}/{len(picture_metadata)}) ---")

# Final save
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(picture_metadata, f, ensure_ascii=False, indent=2)

print(f"\nDone.")
print(f"  Described: {described}")
print(f"  Skipped:   {skipped}")
print(f"  Errors:    {errors}")