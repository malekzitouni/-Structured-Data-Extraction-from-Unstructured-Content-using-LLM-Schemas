import os
import json
import torch
import logging
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TableStructureRecognizer")

class TableStructureRecognizer:
    def __init__(self, model_name: str = "microsoft/table-transformer-structure-recognition", device: str = "cpu"):
        """
        Initialize Table Transformer model for table structure recognition
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on (cpu/cuda)
        """
        self.device = torch.device(device)
        logger.info(f"Loading Table Transformer model: {model_name}")
        
        try:
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Define class mapping
            self.id2label = self.model.config.id2label
            logger.info(f"Using class mapping: {self.id2label}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def recognize(self, image: Image.Image, table_bbox: dict, threshold: float = 0.1):
        """
        Recognize table structure from a cropped table image
        
        Args:
            image: Original document image (PIL Image)
            table_bbox: Bounding box of table region (dict with l,t,r,b)
            threshold: Confidence threshold for detection
            
        Returns:
            List of detected table elements with structure information
        """
        try:
            # Crop table region
            table_img = image.crop((
                max(0, table_bbox["l"]), 
                max(0, table_bbox["t"]), 
                min(image.width, table_bbox["r"]), 
                min(image.height, table_bbox["b"])
            ))
            
            # Save cropped image for debugging
            debug_dir = "debug_tables"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"table_{table_bbox['l']}_{table_bbox['t']}.png")
            table_img.save(debug_path)
            logger.info(f"Saved cropped table: {debug_path}")
            
            # Process image
            if table_img.mode != "RGB":
                table_img = table_img.convert("RGB")
            
            inputs = self.processor(images=table_img, return_tensors="pt").to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert outputs to COCO API format
            target_sizes = torch.tensor([table_img.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes,
                threshold=threshold
            )[0]
            
            # Format results
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                detections.append({
                    "label": self.id2label[label.item()],
                    "score": round(score.item(), 3),
                    "box": [round(coord.item(), 1) for coord in box],
                    "table_bbox": table_bbox
                })
            
            logger.info(f"Detected {len(detections)} elements")
            return detections
            
        except Exception as e:
            logger.error(f"Recognition error: {str(e)}")
            return []

def process_document(
    image_path: str,
    layout_results: list,
    output_dir: str = "table_output",
    device: str = "cpu",
    threshold: float = 0.1
):
    """
    Process a document to extract table structures
    
    Args:
        image_path: Path to document image
        layout_results: Output from layout analysis
        output_dir: Directory to save results
        device: Inference device
        threshold: Detection confidence threshold
    """
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load document image
        image = Image.open(image_path)
        logger.info(f"Loaded image: {image_path} ({image.size})")
        
        # Initialize table recognizer
        table_recognizer = TableStructureRecognizer(device=device)
        
        # Find table regions
        table_regions = [pred for pred in layout_results if pred["label"].lower() == "table"]
        logger.info(f"Found {len(table_regions)} table(s) in document")
        
        # Process each table region
        all_results = []
        for i, table_region in enumerate(table_regions):
            logger.info(f"\nProcessing table {i+1}/{len(table_regions)} at {table_region}")
            
            # Verify table region is valid
            if (table_region['r'] - table_region['l'] < 10 or 
                table_region['b'] - table_region['t'] < 10):
                logger.warning(f"Skipping invalid table region (too small): {table_region}")
                continue
            
            # Recognize table structure
            table_structure = table_recognizer.recognize(
                image, 
                table_region,
                threshold=threshold
            )
            
            # Save results
            output_path = os.path.join(output_dir, f"table_{i}_structure.json")
            with open(output_path, "w") as f:
                json.dump({
                    "table_region": table_region,
                    "elements": table_structure,
                    "image_size": image.size
                }, f, indent=2)
            
            logger.info(f"Saved table structure to {output_path}")
            all_results.append(table_structure)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Document processing error: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Table Structure Recognition with Table Transformer")
    parser.add_argument("--image", type=str, required=True, help="Path to document image")
    parser.add_argument("--layout-results", type=str, required=True, help="Path to layout results JSON file")
    parser.add_argument("--output-dir", type=str, default="table_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--threshold", type=float, default=0.1, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    try:
        # Load layout results
        with open(args.layout_results, "r") as f:
            layout_results = json.load(f)
        
        # Process document
        results = process_document(
            image_path=args.image,
            layout_results=layout_results,
            output_dir=args.output_dir,
            device=args.device,
            threshold=args.threshold
        )
        
        logger.info(f"Successfully processed {len(results)} tables")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")