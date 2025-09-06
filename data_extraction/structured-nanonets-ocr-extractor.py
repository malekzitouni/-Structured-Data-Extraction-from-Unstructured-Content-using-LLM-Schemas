import argparse
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import os
import warnings
import torch
import time
import sys
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

class StructuredNanonetsExtractor:
    DEFAULT_PROMPT = """You are extracting structured data from a scanned document. Extract the text from the above document as if you were reading it naturally. Your goal is to detect all tables and reproduce them with **visible ASCII-style borders** and **accurate layout**.

GENERAL RULES:
1. Extract each table **exactly as seen**—do NOT infer or omit data.
2. NEVER skip rows or columns. Every row must include the full set of columns.
3. Preserve the **exact visual layout**: each row and column must match the scanned structure.
4. Render all tables using ASCII-style borders with `+`, `-`, and `|` characters.

🧾 ASCII FORMAT EXAMPLE:
+------------+----------+---------+
|  Product   | Quantity | Price   |
+============+==========+=========+
| Apple      |    10    | 5.00 €  |
+------------+----------+---------+
| Banana     |     6    | 3.60 €  |
+------------+----------+---------+

FORMAT REQUIREMENTS:
- Use '+' for corners and intersections
- Use '-' for horizontal lines
- Use '|' for vertical column separators
- Ensure perfect column alignment using monospace spacing
- Keep column widths consistent across all rows
- Output only valid, well-aligned bordered tables

FORMAT SPECIFICATIONS:
- Always include a **header row** with column labels
- Use formatting markers (`:--`, `:-:`, `--:`) **only if clearly visible in the image**
- Do NOT merge adjacent cells unless they are **visually merged**
- For empty cells, always render them as: `|        |`
- NEVER omit rows that appear only partially filled

MERGED CELLS:
- For cells merged across columns:
  - Render as: `| **Merged Cell** ||` (leave other cells empty)
- For rows merged under a label:
  - Repeat the main label or leave the subsequent rows blank (but aligned)

SPECIAL HANDLING:
- If a row appears visually broken or misaligned, still reproduce its structure **exactly as-is** and flag it using `<row_warning>`
- Do not combine text from adjacent cells, even if data seems related

FORMATTING TO PRESERVE:
- Numbers: `1 200,00 €`, `-5 %`, `+2 341`
- Currency: €, $, DNT, etc.
- Percentage signs, dashes, and bullet points
- Bold: `**text**`, Italic: `*text*`
- Empty entries: clearly shown using bordered cells
- Visual indicators:
  - Watermarks: `<watermark>TEXT</watermark>`
  - Page numbers: `<page_number>1</page_number>`
  - Checkboxes: ☐ (unchecked), ☑ (checked)

FINAL INSTRUCTIONS:
- Extract **all visible text**, including numeric tables, headers, and footers
- Do NOT reformat or summarize; replicate the original content faithfully
- Each table must be complete and rendered in bordered ASCII format

For non-tabular elements:
- Equations: Render in LaTeX format
- Images: `<img>Description</img>`
- Watermarks: `<watermark>TEXT</watermark>`
- Page numbers: `<page_number>NUM</page_number>`
- Checkboxes: ☐/☑"""

    def __init__(self, model_path="nanonets/Nanonets-OCR-s", cache_dir="C:/models", max_tokens=4000):
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.max_tokens = max_tokens
        self.model = None
        self.processor = None
        
        # Environment setup
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
        warnings.filterwarnings("ignore", message=".*slow image processor.*")
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
        warnings.filterwarnings("ignore", message=".*Some parameters are on the meta device.*")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # GPU setup
        if torch.cuda.is_available():
            try:
                nvmlInit()
                print("GPU memory monitoring enabled")
            except:
                print("Could not initialize NVML for memory monitoring")

    def _print_environment_info(self):
        print("\n" + "="*50)
        print("ENVIRONMENT INFORMATION")
        print("="*50)
        print(f"Python Executable: {sys.executable}")
        print(f"Python Version: {sys.version}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

    def load_model(self):
        try:
            self._print_environment_info()
            
            if torch.cuda.is_available():
                dtype = torch.float16
                device = "cuda"
                print("\nUsing GPU with half-precision (float16)")
            else:
                dtype = torch.float32
                device = "cpu"
                print("\nUsing CPU with full precision (float32)")
            
            print("Loading OCR model...")
            start_time = time.time()
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="auto",
                cache_dir=self.cache_dir,
                offload_folder=self.cache_dir,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            load_time = time.time() - start_time
            print(f"✓ Model loaded in {load_time:.2f}s | Device: {self.model.device} | Precision: {self.model.dtype}")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                cache_dir=self.cache_dir,
                use_fast=True
            )
            print("✓ Processor loaded")
            return True
            
        except Exception as e:
            print(f"✗ Load failed: {str(e)}")
            return False

    def extract_text(self, image_path, prompt=None):
        if not prompt:
            prompt = self.DEFAULT_PROMPT
            
        if not all([self.model, self.processor]):
            raise RuntimeError("Model not initialized. Call load_model() first")
        
        try:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            print("="*50)
            
            image = Image.open(image_path)
            print(f"Original size: {image.size} | Mode: {image.mode}")
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": prompt}, 
                ]},
            ]
            
            print("Applying chat template...")
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            print("Processing inputs...")
            inputs = self.processor( text=[text],  images=[image], padding=True,  return_tensors="pt" ).to(self.model.device)

            print("\nStarting OCR extraction...")
            start_time = time.time()
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False
                )
            
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            result = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
            
            print(f"✓ OCR completed in {time.time() - start_time:.2f} seconds")
            return result
            
        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Extract structured text from images using Nanonets OCR")
    parser.add_argument("-i", "--image", required=True, help="Path to input image file")
    parser.add_argument("-o", "--output", default="structured_output.txt", help="Output file path")
    parser.add_argument("-m", "--model", default="nanonets/Nanonets-OCR-s", help="Model identifier or path")
    parser.add_argument("-c", "--cache_dir", default="C:/models", help="Model cache directory")
    args = parser.parse_args()
    
    extractor = StructuredNanonetsExtractor(
        model_path=args.model,
        cache_dir=args.cache_dir
    )
    
    if extractor.load_model():
        try:
            result = extractor.extract_text(args.image)
            
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result)
                
            print(f"\n✓ OCR completed successfully!")
            print(f"Output saved to: {args.output}")
            
            print("\n" + "="*50)
            print("FIRST 500 CHARACTERS:")
            print("="*50)
            print(result[:500] + "..." if len(result) > 500 else result)
            
        except Exception as e:
            print(f"\n✗ Processing error: {str(e)}")
    else:
        print("\nOCR initialization failed. Check model loading errors.")

if __name__ == "__main__":
    main()