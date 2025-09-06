
# run_block_data_extraction.py
# This script extracts structured text from document blocks using a custom OCR model.
# It requires the BlockTextExtractor class to be properly set up with the necessary model files.
import argparse
import json
import os
import tempfile
import logging
from PIL import Image
from structured_nanonets_extractor import StructuredNanonetsExtractor

# Configure logging




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BlockTextExtractor")

class BlockTextExtractor:
    def __init__(self, model_path="nanonets/Nanonets-OCR-s", cache_dir="C:/models"):
        self.ocr_extractor = StructuredNanonetsExtractor(model_path, cache_dir)
        if not self.ocr_extractor.load_model():
            raise RuntimeError("Failed to load OCR model")
        logger.info("OCR model loaded successfully")
        
        # Define block-specific prompts
        self.prompt_templates = {
            "table": """You are extracting a table from a document. Extract the table exactly as seen, reproducing it with visible ASCII-style borders and accurate layout.

RULES:
1. Extract the table exactly as seen — do NOT infer or omit data.
2. Preserve the exact visual layout: each row and column must match the scanned structure.
3. Render the table using ASCII-style borders with `+`, `-`, and `|` characters.

ASCII FORMAT:
+------------+----------+---------+
| Header 1   | Header 2 | Header 3|
+============+==========+=========+
| Cell 1     | Cell 2   | Cell 3  |
+------------+----------+---------+

FORMAT REQUIREMENTS:
- Use '+' for corners and intersections
- Use '-' for horizontal lines
- Use '|' for vertical column separators
- Ensure perfect column alignment using monospace spacing
- Keep column widths consistent across all rows
- Output only the bordered table""",
            
            "text": """Extract the text from this image exactly as it appears, preserving:
- Line breaks and spacing
- Formatting: bold (`**bold**`), italic (`*italic*`)
- Numbers, symbols, and special characters
- Do not add any markdown except for bold and italic""",
            
            "section-header": """Extract this section header exactly as it appears, including:
- All text formatting (bold, italic, underline)
- Font size variations
- Alignment and positioning
- Do not add any additional text or explanations""",
            
            "title": """Extract this document title exactly as it appears, preserving:
- Centering and alignment
- Font size and weight
- Any special characters or formatting
- Do not modify or abbreviate""",
            
            "default": """Extract all text from this document region exactly as it appears, including:
- All visible characters and symbols
- Line breaks and spacing
- Formatting indicators (bold, italic)
- Numeric values with original formatting"""
        }

    def get_block_prompt(self, block_type):
        """Get customized prompt based on block type"""
        block_type = block_type.lower().replace(" ", "-")
        
        if "table" in block_type:
            return self.prompt_templates["table"]
        elif "section-header" in block_type:
            return self.prompt_templates["section-header"]
        elif "title" in block_type:
            return self.prompt_templates["title"]
        elif "text" in block_type:
            return self.prompt_templates["text"]
        return self.prompt_templates["default"]

    def extract_from_blocks(self, image_path, layout_results, target_labels=None):
        """
        Extract text from document blocks
        
        Args:
            image_path: Path to the document image
            layout_results: Layout analysis results (list of dicts)
            target_labels: List of labels to extract (if None, extract all blocks)
        
        Returns:
            List of extracted blocks with text content
        """
        try:
            # Open the document image
            image = Image.open(image_path)
            logger.info(f"Processing image: {os.path.basename(image_path)}")
            
            # Filter layout results for target labels (if specified)
            if target_labels:
                target_blocks = [block for block in layout_results if block['label'] in target_labels]
                logger.info(f"Found {len(target_blocks)} target blocks to extract")
            else:
                target_blocks = layout_results
                logger.info(f"Extracting all {len(target_blocks)} blocks")
            
            results = []
            for i, block in enumerate(target_blocks):
                # Crop the block region from the image
                bbox = (block['l'], block['t'], block['r'], block['b'])
                cropped_img = image.crop(bbox)
                
                # Get customized prompt for this block type
                prompt = self.get_block_prompt(block['label'])
                
                # Save cropped image to temp file for OCR processing
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    cropped_img.save(temp_path)
                
                # Extract text from the cropped block with customized prompt
                logger.info(f"Extracting text from {block['label']} block {i+1}/{len(target_blocks)}")
                text = self.ocr_extractor.extract_text(temp_path, prompt=prompt)
                os.unlink(temp_path)  # Clean up temp file
                
                results.append({
                    "block_id": i,
                    "label": block['label'],
                    "confidence": block['confidence'],
                    "bounding_box": [block['l'], block['t'], block['r'], block['b']],
                    "text": text
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise

    def save_results(self, results, output_path):
        """Save extraction results to JSON and TXT files"""
        # Save JSON
        json_path = output_path if output_path.endswith('.json') else f"{output_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON results saved to: {json_path}")
        
        # Save TXT
        txt_path = os.path.splitext(json_path)[0] + ".txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            for i, block in enumerate(results):
                f.write(f"=== BLOCK {i+1} ===\n")
                f.write(f"Type: {block['label']}\n")
                f.write(f"Confidence: {block['confidence']:.2f}\n")
                f.write(f"Position: {block['bounding_box']}\n")
                f.write("\nEXTRACTED TEXT:\n")
                f.write(block['text'] + "\n")
                f.write("\n" + "="*50 + "\n\n")
        logger.info(f"Text results saved to: {txt_path}")
        
        return json_path, txt_path

def main():
    parser = argparse.ArgumentParser(description="Extract text from document blocks")
    parser.add_argument("--image", required=True, help="Path to document image")
    parser.add_argument("--layout-results", required=True, help="Path to layout analysis JSON file")
    parser.add_argument("--output", default="block_text_output", help="Output base name (without extension)")
    parser.add_argument("--labels", default=None, help="Comma-separated labels to extract (e.g., Table,Section-header). Omit to extract all blocks.")
    parser.add_argument("--model", default="nanonets/Nanonets-OCR-s", help="OCR model identifier")
    parser.add_argument("--cache-dir", default="C:/models", help="Model cache directory")
    
    args = parser.parse_args()
    
    try:
        # Load layout results
        with open(args.layout_results, 'r') as f:
            layout_results = json.load(f)
        
        # Parse target labels if provided
        target_labels = None
        if args.labels:
            target_labels = [label.strip() for label in args.labels.split(',')]
            logger.info(f"Target labels: {', '.join(target_labels)}")
        
        # Initialize extractor
        extractor = BlockTextExtractor(model_path=args.model, cache_dir=args.cache_dir)
        
        # Extract text from blocks
        results = extractor.extract_from_blocks(
            image_path=args.image,
            layout_results=layout_results,
            target_labels=target_labels
        )
        
        # Save results to both JSON and TXT
        json_path, txt_path = extractor.save_results(results, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("BLOCK TEXT EXTRACTION SUMMARY")
        print("="*50)
        print(f"Document: {os.path.basename(args.image)}")
        print(f"Layout results: {os.path.basename(args.layout_results)}")
        print(f"Blocks extracted: {len(results)}")
        print(f"JSON output: {json_path}")
        print(f"Text output: {txt_path}")
        
        # Print sample output
        if results:
            print("\nSAMPLE OUTPUT:")
            print("-"*50)
            sample = results[0]
            print(f"Block Type: {sample['label']}")
            print(f"Confidence: {sample['confidence']:.2f}")
            print(f"Position: {sample['bounding_box']}")
            print("\nExtracted Text:")
            print(sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text'])
        
    except Exception as e:
        logger.error(f"Block extraction failed: {str(e)}")

if __name__ == "__main__":
    main()