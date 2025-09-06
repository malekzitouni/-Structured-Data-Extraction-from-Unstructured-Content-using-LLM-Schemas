import os
import uuid
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
from run_layout_analysis import LayoutPredictor, LayoutLabels
from structured_nanonets_extractor import StructuredNanonetsExtractor
from huggingface_hub import snapshot_download
from run_mapping import SchemaMapper
from application_schema import BaseSchema

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AxeFinanceExtractor")

# Create required directories
Path("static/uploads").mkdir(parents=True, exist_ok=True)
Path("templatess").mkdir(parents=True, exist_ok=True)

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templatess")

# Model cache
MODEL_CACHE = {}
PROCESSING_DATA = {}

class FinanceDocumentExtractor:
    def __init__(self):
        self.layout_predictor = None
        self.ocr_extractor = None
        self.schema_mapper = None
        self.layout_labels = LayoutLabels()
        
    def load_models(self):
        """Load both layout, OCR and schema mapping models"""
        # Load layout model
        if "layout" not in MODEL_CACHE:
            logger.info("Loading layout model...")
            try:
                layout_model_path = snapshot_download("ds4sd/docling-layout-egret-large")
                MODEL_CACHE["layout"] = LayoutPredictor(
                    artifact_path=layout_model_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    base_threshold=0.3
                )
                logger.info("Layout model loaded")
            except Exception as e:
                logger.error(f"Failed to load layout model: {str(e)}")
                raise RuntimeError(f"Layout model loading failed: {str(e)}")
        
        # Load OCR model
        if "ocr" not in MODEL_CACHE:
            logger.info("Loading OCR model...")
            try:
                MODEL_CACHE["ocr"] = StructuredNanonetsExtractor(
                    model_path="nanonets/Nanonets-OCR-s",
                    cache_dir="C:/models"
                )
                if not MODEL_CACHE["ocr"].load_model():
                    raise RuntimeError("Failed to load OCR model")
                logger.info("OCR model loaded")
            except Exception as e:
                logger.error(f"Failed to load OCR model: {str(e)}")
                raise RuntimeError(f"OCR model loading failed: {str(e)}")
        
        # Load schema mapper
        if "schema_mapper" not in MODEL_CACHE:
            logger.info("Loading schema mapper...")
            try:
                MODEL_CACHE["schema_mapper"] = SchemaMapper()
                logger.info("Schema mapper loaded")
            except Exception as e:
                logger.error(f"Failed to load schema mapper: {str(e)}")
                raise RuntimeError(f"Schema mapper loading failed: {str(e)}")
        
        self.layout_predictor = MODEL_CACHE["layout"]
        self.ocr_extractor = MODEL_CACHE["ocr"]
        self.schema_mapper = MODEL_CACHE["schema_mapper"]
    
    def extract_document_data(self, image_path: str, user_schema: Dict[str, str]):
        """
        Full processing pipeline:
        1. Layout analysis
        2. OCR extraction
        3. Schema mapping
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
                ocr_results = []
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
                        section_text.append({
                            "bounding_box": [block['l'], block['t'], block['r'], block['b']],
                            "text": text
                        })
                    
                    # Get average confidence for section
                    confidence = sum(b['confidence'] for b in blocks) / len(blocks) if blocks else 0
                    
                    ocr_results.append({
                        "label": label,
                        "confidence": confidence,
                        "blocks": section_text
                    })
                
                # Combine extracted text for mapping
                combined_data = []
                for section in ocr_results:
                    for block in section["blocks"]:
                        combined_data.append({
                            "label": section["label"],
                            "text": block["text"],
                            "confidence": section["confidence"]
                        })
                
                # Map to user schema
                logger.info("Mapping data to user schema")
                mapped_data = self.schema_mapper.map_to_schema(
                    extracted_data=combined_data,
                    schema_definition=user_schema
                )
                
                return {
                    "layout_results": layout_results,
                    "ocr_results": ocr_results,
                    "mapped_data": mapped_data
                }
                
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
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
    return templates.TemplateResponse("index_updated.html", {
        "request": request,
        "example_schema": json.dumps(BaseSchema.example_schema(), indent=2)
    })

@app.post("/upload")
async def upload_document(
    image: UploadFile = File(...), 
    schema_file: UploadFile = File(...)
):
    """Handle document upload and schema upload, start processing"""
    try:
        # Generate unique ID for this process
        process_id = str(uuid.uuid4())
        upload_dir = Path(f"static/uploads/{process_id}")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded image
        image_path = upload_dir / "original.png"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        
        # Save schema file
        schema_path = upload_dir / "schema.json"
        schema_content = await schema_file.read()
        with open(schema_path, "wb") as f:
            f.write(schema_content)
        
        # Load and validate schema
        try:
            with open(schema_path, "r") as f:
                try:
                    user_schema = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON syntax: {str(e)}")
                    return JSONResponse(
                        {
                            "status": "error", 
                            "message": f"Invalid JSON in schema file: {str(e)}. "
                                       "Make sure to use double quotes for property names."
                        },
                        status_code=400
                    )
            
            # Validate schema structure
            try:
                BaseSchema.validate_schema(user_schema)
            except ValueError as e:
                logger.error(f"Schema validation error: {str(e)}")
                return JSONResponse(
                    {
                        "status": "error", 
                        "message": f"Invalid schema: {str(e)}. "
                                   "Example schema: " + json.dumps(BaseSchema.example_schema())
                    },
                    status_code=400
                )
        except Exception as e:
            logger.error(f"Schema validation error: {str(e)}")
            return JSONResponse(
                {"status": "error", "message": f"Schema validation failed: {str(e)}"},
                status_code=400
            )
        
        # Store processing data
        PROCESSING_DATA[process_id] = {
            "status": "uploaded",
            "image_path": str(image_path),
            "image_url": f"/static/uploads/{process_id}/original.png",
            "user_schema": user_schema,
            "schema_path": str(schema_path)
        }
        
        return JSONResponse({"status": "success", "process_id": process_id})
    
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/process/{process_id}")
async def process_document(process_id: str):
    """Process uploaded document with the provided schema"""
    if process_id not in PROCESSING_DATA:
        return JSONResponse(
            {"status": "error", "message": "Invalid process ID"},
            status_code=404
        )
    
    try:
        # Update status
        PROCESSING_DATA[process_id]["status"] = "processing"
        
        # Load models
        extractor.load_models()
        
        # Process document
        image_path = PROCESSING_DATA[process_id]["image_path"]
        user_schema = PROCESSING_DATA[process_id]["user_schema"]
        if not isinstance(PROCESSING_DATA[process_id]["user_schema"], dict):
            raise ValueError("Schema must be a dictionary")
            
        # Ensure schema values are strings
        validated_schema = {
            k: str(v).lower() if isinstance(v, (str, int, float)) else "string"
            for k, v in PROCESSING_DATA[process_id]["user_schema"].items()
        }
        
        results = extractor.extract_document_data(
            image_path, 
            validated_schema  # Use validated schema
        )
        
        # Save results
        PROCESSING_DATA[process_id]["results"] = results
        PROCESSING_DATA[process_id]["status"] = "completed"
        
        return JSONResponse({"status": "success"})
    
    except Exception as e:
        PROCESSING_DATA[process_id]["status"] = "error"
        PROCESSING_DATA[process_id]["error"] = str(e)
        logger.error(f"Processing failed: {str(e)}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
            "result_updated.html",
            {
                "request": request,
                "image_url": data["image_url"],
                "results": data["results"]
            }
        )
    except Exception as e:
        logger.error(f"Template rendering failed: {str(e)}")
        return HTMLResponse(f"<h1>Error displaying results</h1><p>{str(e)}</p>", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)