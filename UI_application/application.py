import os
import uuid
import json
import tempfile
import logging
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from run_layout_analysis import LayoutPredictor, LayoutLabels
from structured_nanonets_extractor import StructuredNanonetsExtractor
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AxeFinanceExtractor")

# Create required directories
Path("static/uploads").mkdir(parents=True, exist_ok=True)
Path("templatesss").mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model cache
MODEL_CACHE = {}
PROCESSING_DATA = {}

class FinanceDocumentExtractor:
    def __init__(self):
        self.layout_predictor = None
        self.ocr_extractor = None
        self.layout_labels = LayoutLabels()
        
    def load_models(self):
        """Load both layout and OCR models"""
        # Load layout model
        if "layout" not in MODEL_CACHE:
            logger.info("Loading layout model...")
            layout_model_path = snapshot_download("ds4sd/docling-layout-egret-large")
            MODEL_CACHE["layout"] = LayoutPredictor(
                artifact_path=layout_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                base_threshold=0.3
            )
            logger.info("Layout model loaded")
        
        # Load OCR model
        if "ocr" not in MODEL_CACHE:
            logger.info("Loading OCR model...")
            MODEL_CACHE["ocr"] = StructuredNanonetsExtractor(
                model_path="nanonets/Nanonets-OCR-s",
                cache_dir="C:/models"
            )
            if not MODEL_CACHE["ocr"].load_model():
                raise RuntimeError("Failed to load OCR model")
            logger.info("OCR model loaded")
        
        self.layout_predictor = MODEL_CACHE["layout"]
        self.ocr_extractor = MODEL_CACHE["ocr"]
    
    def extract_document_data(self, image_path: str):
        """
        Extract financial data from document with visual grouping
        
        Steps:
        1. Perform layout analysis
        2. Group blocks by type and vertical position
        3. Extract text from each block with specialized prompts
        4. Return structured results
        """
        try:
            # Open and process image
            with Image.open(image_path) as img:
                # Get layout predictions
                layout_results = list(self.layout_predictor.predict(img))
                
                # Group blocks by type
                grouped_blocks = {}
                for block in layout_results:
                    label = block['label']
                    if label not in grouped_blocks:
                        grouped_blocks[label] = []
                    grouped_blocks[label].append(block)
                
                # Sort blocks vertically (top to bottom)
                for label, blocks in grouped_blocks.items():
                    grouped_blocks[label] = sorted(blocks, key=lambda x: x['t'])
                
                # Extract text for each block group
                results = []
                for label, blocks in grouped_blocks.items():
                    section_text = []
                    for block in blocks:
                        # Crop block region
                        bbox = (block['l'], block['t'], block['r'], block['b'])
                        cropped_img = img.crop(bbox)
                        
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                            temp_path = temp_file.name
                            cropped_img.save(temp_path)
                        
                        # Get appropriate prompt
                        prompt = self.get_block_prompt(label)
                        
                        # Extract text
                        text = self.ocr_extractor.extract_text(temp_path, prompt=prompt)
                        os.unlink(temp_path)
                        section_text.append(text)
                    
                    # Get average confidence for section
                    confidence = sum(b['confidence'] for b in blocks) / len(blocks) if blocks else 0
                    
                    results.append({
                        "label": label,
                        "confidence": confidence,
                        "blocks": [
                            {
                                "bounding_box": [b['l'], b['t'], b['r'], b['b']],
                                "text": text
                            } for b, text in zip(blocks, section_text)
                        ]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise
    
    def get_block_prompt(self, block_type: str) -> str:
        """Get specialized prompt for different document regions"""
        block_type = block_type.lower().replace(" ", "-")
        
        prompts = {
            "table": """Extract this financial table exactly as shown. 
            PRESERVE:
            - All numerical values with original formatting
            - Currency symbols and decimal places
            - Table structure with rows and columns
            
            OUTPUT FORMAT:
            +----------------+-----------+-----------+
            | Column Header  | Header 2  | Header 3  |
            +================+===========+===========+
            | Value 1        | 1,200.00€ | 15.5%     |
            +----------------+-----------+-----------+
            | Value 2        | 3,450.00€ | 22.3%     |
            +----------------+-----------+-----------+""",
            
            "title": """Extract the document title exactly as shown with:
            - Original capitalization
            - Special characters
            - Centering information""",
            
            "section-header": """Extract this section header preserving:
            - Font size and weight indications
            - Alignment
            - All text formatting""",
            
            "text": """Extract all text exactly as seen with:
            - Line breaks preserved
            - Numerical formatting maintained
            - Currency symbols and percentages
            - Bullet points and special characters""",
            
            "default": """Extract all visible text exactly as it appears in the document,
            preserving original formatting, numbers, and special characters."""
        }
        
        if "table" in block_type:
            return prompts["table"]
        elif "title" in block_type:
            return prompts["title"]
        elif "section-header" in block_type:
            return prompts["section-header"]
        elif "text" in block_type:
            return prompts["text"]
        return prompts["default"]

# Initialize extractor
extractor = FinanceDocumentExtractor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main page with upload form"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_document(image: UploadFile = File(...)):
    """Handle document upload and start processing"""
    try:
        # Generate unique ID for this process
        process_id = str(uuid.uuid4())
        os.makedirs(f"static/uploads/{process_id}", exist_ok=True)
        
        # Save uploaded image
        image_path = f"static/uploads/{process_id}/original.png"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        
        # Store processing data
        PROCESSING_DATA[process_id] = {
            "status": "uploaded",
            "image_path": image_path,
            "image_url": f"/static/uploads/{process_id}/original.png"
        }
        
        return JSONResponse({"status": "success", "process_id": process_id})
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/process/{process_id}")
async def process_document(process_id: str):
    """Process uploaded document"""
    if process_id not in PROCESSING_DATA:
        return JSONResponse({"status": "error", "message": "Invalid process ID"})
    
    try:
        # Load models if needed
        extractor.load_models()
        
        # Update status
        PROCESSING_DATA[process_id]["status"] = "processing"
        
        # Process document
        image_path = PROCESSING_DATA[process_id]["image_path"]
        results = extractor.extract_document_data(image_path)
        
        # Save results
        PROCESSING_DATA[process_id]["results"] = results
        PROCESSING_DATA[process_id]["status"] = "completed"
        
        return JSONResponse({"status": "success"})
    
    except Exception as e:
        PROCESSING_DATA[process_id]["status"] = "error"
        PROCESSING_DATA[process_id]["error"] = str(e)
        return JSONResponse({"status": "error", "message": str(e)})

@app.get("/results/{process_id}", response_class=HTMLResponse)
async def view_results(process_id: str, request: Request):
    """Display processing results"""
    if process_id not in PROCESSING_DATA:
        return RedirectResponse("/")
    
    data = PROCESSING_DATA[process_id]
    if data["status"] != "completed":
        return RedirectResponse("/")
    
    try:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "image_url": data["image_url"],
                "sections": data["results"]
            }
        )
    except Exception as e:
        logger.error(f"Template rendering failed: {str(e)}")
        return HTMLResponse(f"<h1>Error displaying results</h1><p>{str(e)}</p>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)