from PIL import Image
import torch
import os
import sys
import time
import safetensors.torch  # Required for loading safetensors
from transformers import (
    AutoTokenizer, 
    AutoProcessor, 
    AutoModelForImageTextToText,
    AutoImageProcessor,
    TableTransformerForObjectDetection,
    AutoConfig
)
from collections import defaultdict

# ===== CONFIGURATION =====
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
torch.set_grad_enabled(False)

MODEL_PATHS = {
    "layout": r"C:\Users\Pc\.cache\docling\models\ds4sd--docling-models\model_artifacts\layout",
    "structure": r"C:\Users\Pc\.cache\docling\models\ds4sd--docling-models\model_artifacts\tableformer\accurate",  # Update to your desktop path
    "ocr": "nanonets/Nanonets-OCR-s"
}
CACHE_DIR = "C:/models"
IMAGE_PATH = r"C:\Users\Pc\Desktop\axe_docl\batch1-0001.jpg"
OUTPUT_DIR = "output_tables"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLDS = {
    "layout": 0.85,
    "structure": 0.75
}

# ===== MODEL LOADING =====
def load_models():
    """Load all required models with diagnostics"""
    print("\n" + "="*50)
    print("LOADING MODELS")
    print("="*50)
    
    models = {}
    try:
        # Load layout detection model from local cache
        print("Loading layout model from local cache...")
        layout_processor = AutoImageProcessor.from_pretrained(
            MODEL_PATHS["layout"],
            cache_dir=CACHE_DIR
        )
        layout_model = TableTransformerForObjectDetection.from_pretrained(
            MODEL_PATHS["layout"],
            cache_dir=CACHE_DIR
        ).to(DEVICE).eval()
        models["layout"] = (layout_processor, layout_model)
        print("✓ Layout model loaded from local cache")
        
        # Load TableFormer structure model from desktop
        print("Loading TableFormer structure model...")
        structure_config = AutoConfig.from_pretrained(
            os.path.join(MODEL_PATHS["structure"], "tm_config.json")
        )
        
        # Create model instance and load weights
        structure_model = TableTransformerForObjectDetection(structure_config).to(DEVICE).eval()
        weights_path = os.path.join(MODEL_PATHS["structure"], "tableformer_accurate.safetensors")
        structure_model.load_state_dict(safetensors.torch.load_file(weights_path))
        
        # Create processor (using same as layout since TableFormer is TableTransformer-based)
        structure_processor = AutoImageProcessor.from_pretrained(
            MODEL_PATHS["layout"],  # Reuse layout processor config
            cache_dir=CACHE_DIR
        )
        models["structure"] = (structure_processor, structure_model)
        print("✓ TableFormer structure model loaded")
        
        # Load NanoNets OCR model
        print("Loading NanoNets OCR model...")
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        ocr_model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATHS["ocr"],
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True
        ).eval()
        ocr_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["ocr"], cache_dir=CACHE_DIR)
        ocr_processor = AutoProcessor.from_pretrained(MODEL_PATHS["ocr"], cache_dir=CACHE_DIR, use_fast=True)
        models["ocr"] = (ocr_model, ocr_processor, ocr_tokenizer)
        print("✓ NanoNets OCR model loaded")
        
        return models
        
    except Exception as e:
        print(f"✗ Model loading failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ===== TABLE DETECTION =====
def detect_tables(image, layout_processor, layout_model):
    """Detect table regions using layout model"""
    print("\nDetecting tables in document...")
    start_time = time.time()
    
    inputs = layout_processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = layout_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = layout_processor.post_process_object_detection(
        outputs, 
        threshold=CONFIDENCE_THRESHOLDS["layout"], 
        target_sizes=target_sizes
    )[0]
    
    tables = []
    for score, box in zip(results["scores"], results["boxes"]):
        if score > CONFIDENCE_THRESHOLDS["layout"]:
            box = [round(i, 2) for i in box.tolist()]
            tables.append({
                "box": box,
                "score": round(score.item(), 3)
            })
    
    print(f"✓ Found {len(tables)} tables in {time.time()-start_time:.2f}s")
    return tables

# ===== TABLE STRUCTURE RECOGNITION =====
def recognize_table_structure(table_image, structure_processor, structure_model):
    """Recognize table structure using TableFormer"""
    print("Analyzing table structure with TableFormer...")
    start_time = time.time()
    
    inputs = structure_processor(images=table_image, return_tensors="pt").to(DEVICE)
    outputs = structure_model(**inputs)
    target_sizes = torch.tensor([table_image.size[::-1]])
    results = structure_processor.post_process_object_detection(
        outputs, 
        threshold=CONFIDENCE_THRESHOLDS["structure"], 
        target_sizes=target_sizes
    )[0]
    
    structure_objects = []
    for score, box, label in zip(results["scores"], results["boxes"], results["labels"]):
        if score > CONFIDENCE_THRESHOLDS["structure"]:
            structure_objects.append({
                "label": structure_model.config.id2label[label.item()],
                "box": [round(i, 2) for i in box.tolist()],
                "score": round(score.item(), 3)
            })
    
    print(f"✓ Extracted {len(structure_objects)} structural elements in {time.time()-start_time:.2f}s")
    return structure_objects

# ===== STRUCTURE TO CELLS =====
def structure_to_cells(structure_objects, img_size):
    """Convert structural elements to table cells"""
    rows = []
    columns = []
    spanning_cells = []
    cells = []
    
    for obj in structure_objects:
        if obj["label"] == "table row":
            rows.append(obj)
        elif obj["label"] == "table column":
            columns.append(obj)
        elif obj["label"] == "table spanning cell":
            spanning_cells.append(obj)
        elif "table cell" in obj["label"]:
            cells.append(obj)
    
    # Sort rows and columns
    rows.sort(key=lambda x: x["box"][1])
    columns.sort(key=lambda x: x["box"][0])
    
    # Handle spanning cells
    for span in spanning_cells:
        span_x1, span_y1, span_x2, span_y2 = span["box"]
        row_indices = [i for i, row in enumerate(rows) 
                      if row["box"][1] <= span_y1 and row["box"][3] >= span_y2]
        col_indices = [i for i, col in enumerate(columns) 
                      if col["box"][0] <= span_x1 and col["box"][2] >= span_x2]
        
        if row_indices and col_indices:
            cells.append({
                "box": span["box"],
                "row_start": min(row_indices),
                "row_end": max(row_indices),
                "col_start": min(col_indices),
                "col_end": max(col_indices),
                "is_span": True
            })
    
    # Handle regular cells
    for cell in cells:
        if "is_span" not in cell:
            cell_x1, cell_y1, cell_x2, cell_y2 = cell["box"]
            row_indices = [i for i, row in enumerate(rows) 
                         if row["box"][1] <= cell_y1 and row["box"][3] >= cell_y2]
            col_indices = [i for i, col in enumerate(columns) 
                         if col["box"][0] <= cell_x1 and col["box"][2] >= cell_x2]
            
            if row_indices and col_indices:
                cell.update({
                    "row_start": min(row_indices),
                    "row_end": max(row_indices),
                    "col_start": min(col_indices),
                    "col_end": max(col_indices),
                    "is_span": False
                })
    
    return cells

# ===== NANONETS OCR FOR CELLS =====
def extract_cell_text(cell_image, ocr_model, ocr_processor):
    """Extract text from a single cell using NanoNets OCR"""
    try:
        # Create optimized prompt for cell extraction
        prompt = """Extract the text exactly as it appears in this image. Preserve:
- All numbers, symbols and special characters (€, $, %, etc.)
- Original formatting and spacing
- Line breaks within the cell
- Numeric formats (1 200,00)
Do NOT add any explanations or additional text. Output ONLY the extracted content."""

        messages = [
            {"role": "system", "content": "You are a precise OCR assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": cell_image},
                {"type": "text", "text": prompt}, 
            ]},
        ]
        
        # Apply chat template
        text = ocr_processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = ocr_processor(
            text=[text], 
            images=[cell_image], 
            return_tensors="pt"
        ).to(ocr_model.device)
        
        # Run inference
        with torch.inference_mode():
            outputs = ocr_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                num_beams=1
            )
        
        # Decode results
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        result = ocr_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return result.strip()
        
    except Exception as e:
        print(f"Cell OCR failed: {str(e)}")
        return ""

# ===== TABLE TEXT EXTRACTION =====
def extract_table_text(table_image, cells, ocr_model, ocr_processor):
    """Extract text for all cells in a table"""
    print("Extracting text from cells...")
    start_time = time.time()
    
    for cell in cells:
        # Crop cell image
        x1, y1, x2, y2 = cell["box"]
        cell_img = table_image.crop((x1, y1, x2, y2))
        
        # Enhance small cells
        if cell_img.width < 50 or cell_img.height < 20:
            cell_img = cell_img.resize(
                (max(100, cell_img.width * 2), 
                (max(40, cell_img.height * 2)),
                Image.LANCZOS
            )
            )
        
        # Extract text
        cell["text"] = extract_cell_text(cell_img, ocr_model, ocr_processor)
    
    print(f"✓ Extracted text from {len(cells)} cells in {time.time()-start_time:.2f}s")
    return cells

# ===== HTML GENERATION =====
def generate_html_table(cells):
    """Generate HTML table from cell structure"""
    if not cells:
        return "<table></table>"
    
    # Find max rows and columns
    max_row = max(cell["row_end"] for cell in cells)
    max_col = max(cell["col_end"] for cell in cells)
    
    # Create grid
    grid = [[None] * (max_col + 1) for _ in range(max_row + 1)]
    cell_map = {}
    
    for cell in cells:
        for r in range(cell["row_start"], cell["row_end"] + 1):
            for c in range(cell["col_start"], cell["col_end"] + 1):
                grid[r][c] = cell
    
    # Generate HTML
    html = ['<table border="1" style="border-collapse: collapse; width: 100%;">']
    for r in range(max_row + 1):
        html.append("<tr>")
        for c in range(max_col + 1):
            cell = grid[r][c]
            if cell is None or (r, c) in cell_map:
                continue
                
            cell_map[(r, c)] = True
            row_span = cell["row_end"] - cell["row_start"] + 1
            col_span = cell["col_end"] - cell["col_start"] + 1
            
            html.append(
                f'<td rowspan="{row_span}" colspan="{col_span}" '
                f'style="border: 1px solid black; padding: 4px; vertical-align: top">'
                f'{cell.get("text", "")}'
                '</td>'
            )
        html.append("</tr>")
    html.append("</table>")
    return "\n".join(html)

# ===== IMAGE PROCESSING =====
def crop_table(image, box):
    """Crop table region from document with padding"""
    x1, y1, x2, y2 = box
    padding = 15
    return image.crop((
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(image.width, x2 + padding),
        min(image.height, y2 + padding)
    ))

# ===== MAIN PIPELINE =====
def process_document(image_path, models):
    """Full document processing pipeline"""
    original_image = Image.open(image_path).convert("RGB")
    print(f"\nProcessing document: {os.path.basename(image_path)}")
    print(f"Original size: {original_image.size}")
    
    # Detect tables
    layout_processor, layout_model = models["layout"]
    tables = detect_tables(original_image, layout_processor, layout_model)
    
    if not tables:
        print("No tables detected in document")
        return []
    
    # Process each table
    results = []
    structure_processor, structure_model = models["structure"]
    ocr_model, ocr_processor, _ = models["ocr"]
    
    for i, table in enumerate(tables):
        print(f"\nProcessing table {i+1}/{len(tables)}")
        
        # Crop table region
        table_img = crop_table(original_image, table["box"])
        
        # Recognize table structure
        structure_objects = recognize_table_structure(
            table_img, 
            structure_processor, 
            structure_model
        )
        
        # Convert to cells
        cells = structure_to_cells(structure_objects, table_img.size)
        
        if not cells:
            print("No cells detected in table")
            continue
        
        # Extract text using NanoNets OCR
        cells_with_text = extract_table_text(table_img, cells, ocr_model, ocr_processor)
        
        # Generate HTML output
        html_table = generate_html_table(cells_with_text)
        
        # Save results
        table_name = f"table_{i+1}"
        results.append({
            "name": table_name,
            "html": html_table
        })
        
        # Save to files
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save HTML
        html_path = os.path.join(OUTPUT_DIR, f"{table_name}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_table)
        print(f"✓ Saved HTML table to {html_path}")
    
    return results

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Verify model paths exist
    for model_type, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Error: Model path not found for {model_type}: {path}")
            sys.exit(1)
    
    # Load models
    models = load_models()
    
    if not models:
        print("Failed to load models. Exiting.")
        sys.exit(1)
    
    # Process document
    start_time = time.time()
    results = process_document(IMAGE_PATH, models)
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE")
    print("="*50)
    print(f"Total tables extracted: {len(results)}")
    print(f"Total processing time: {time.time()-start_time:.2f} seconds")
    
    # Show sample output
    if results:
        print("\nFirst table preview saved as HTML")