#!/usr/bin/env python3
"""
load_docling_models.py

Demonstrates how to locally load Docling’s layout (RT-DETR) and table (TableFormer)
models from the Hugging Face repo `ds4sd/docling-models` and run a quick test.
"""

import os
from huggingface_hub import snapshot_download

# 1. (Optional) Force‐download all artifacts into a cache dir
REPO_ID = "ds4sd/docling-models"
CACHE_DIR = os.path.expanduser("~/.cache/docling-models")

print(f"Downloading Docling models to {CACHE_DIR}…")
local_path = snapshot_download(
    repo_id=REPO_ID,
    cache_dir=CACHE_DIR,
    repo_type="model",
    force_download=False  # set True to re-download every time
)
print(f"✅ Models available at: {local_path}\n")

# 2. Import and configure Docling converter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel import pipeline_options

print("\nDefinition of PdfPipelineOptions:\n")
import inspect
print(inspect.getsource(pipeline_options.PdfPipelineOptions))


print("Setting up DocumentConverter…")
pdf_opts = PdfPipelineOptions(
    # you can customize DPI, OCR backend, etc., here
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
    }
)

# 3. Quick sanity‐check: convert an empty or tiny PDF stub
#    (replace with an actual PDF path to test end-to-end)
stub_file = r"C:\Users\Pc\Desktop\axe_docl\batch1-0001.jpg"
if os.path.exists(stub_file):
    doc = converter.convert(source=stub_file).document
    print("Type of doc object:", type(doc))
    print("DoclingDocument attributes and methods:")
    print(dir(doc))
    print("DoclingDocument __dict__ (fields and values):")
    print(doc.__dict__)
    print(type(doc).__name__, "has", len(doc.__dict__), "fields." )
    print("Sample conversion result:")
    try:
        print(doc.model_dump_json()[:500], "…")
    except AttributeError:
        try:
            print(doc.json()[:500], "…")
        except AttributeError:
            import json
            print(json.dumps(doc.dict())[:500], "…")
    # Save doc.__dict__ to a file as JSON
    import json

    try:
        doc_data = doc.model_dump()
    except AttributeError:
        doc_data = doc.dict()

    dict_output_path = os.path.splitext(stub_file)[0] + "_doc_all_attrs.json"
    with open(dict_output_path, "w", encoding="utf-8") as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    print(f"✅ All doc attributes saved to: {dict_output_path}")
else:
    print(f"⚠️  Stub file not found at '{stub_file}'.")
    print("   Create a small image or PDF, or change `stub_file` path.")

