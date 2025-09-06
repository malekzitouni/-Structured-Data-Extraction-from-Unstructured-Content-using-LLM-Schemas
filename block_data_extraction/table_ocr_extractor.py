# table_ocr_extractor.py

# Description:
# This script loads a fine-tuned image-to-text OCR model and applies it to a scanned document image.
# It uses a carefully designed chat-style prompt to extract only **tables** from the image with high fidelity.
# The extracted tables are rendered using clean ASCII borders (`+`, `|`, `-`) and saved as structured text.
#
# Features:
# - Focuses exclusively on extracting tables (ignores paragraphs, headers, footers, etc.)
# - Preserves table structure, column alignment, merged cells, and empty values
# - Automatically detects and leverages GPU acceleration (if available)
# - Outputs results in a readable, fixed-width ASCII table format



# Output:
# The structured tables are saved to the file: `extracted_tables_batch1-0001.txt`

from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import os
import warnings
import torch
import time
import sys
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# ===== CONFIGURATION =====
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
warnings.filterwarnings("ignore", message=".*slow image processor.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*Some parameters are on the meta device.*")

MODEL_PATH = "nanonets/Nanonets-OCR-s"
CACHE_DIR = "C:/models"
IMAGE_PATH = r"C:\Users\Pc\Desktop\axe_docl\فاتورة بتصميم عصري باللون الأزرق والأبيض.jpg"
MAX_TOKENS = 4000

# ===== ENVIRONMENT DIAGNOSTICS =====
def print_environment_info():
    """Print environment information"""
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

# ===== MODEL LOADING =====
def load_ocr_components():
    """Load model components with optimizations"""
    try:
        print_environment_info()
        
        # Determine precision
        if torch.cuda.is_available():
            dtype = torch.float16
            device = "cuda"
            print("\nUsing GPU with half-precision (float16)")
        else:
            dtype = torch.float32
            device = "cpu"
            print("\nUsing CPU with full precision (float32)")
        
        # Load model
        print("Loading OCR model...")
        start_time = time.time()
        
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=CACHE_DIR,
            offload_folder=CACHE_DIR,
            low_cpu_mem_usage=True
        )
        
        model.eval()
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s | Device: {model.device} | Precision: {model.dtype}")
        
        # Load processor and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)
        processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR, use_fast=True)
        print("✓ Processor and tokenizer loaded")
        
        return model, processor
        
    except Exception as e:
        print(f"✗ Load failed: {str(e)}")
        return None, None

# ===== OCR FUNCTION =====
def run_ocr(image_path, model, processor):
    """Perform OCR extraction using chat template format"""
    if not all([model, processor]):
        raise RuntimeError("Model not initialized")
    
    try:
        print("\nPreparing image...")
        image = Image.open(image_path)
        print(f"Original size: {image.size} | Mode: {image.mode}")
        
        # Define prompt as in the working script


        prompt = """ 
    **TASK: Extract ONLY TABLES from this document. Ignore ALL other text (headers, paragraphs, footers, etc.).**

    RULES:
    1. **ONLY OUTPUT TABLES** with ASCII borders. Skip all non-tabular text.
    2. For each table:
        - Use `+`, `-`, and `|` to create borders
        - Preserve EXACT column alignment
        - Include headers and all data rows
    3. Handle merged cells visually:
        - Horizontal merge: `| Merged Text     ||`
        - Vertical merge: Repeat label or leave blank
    4. Empty cells: Render as `|        |`
    5. If NO tables exist, output: "No tables detected"

    FORMAT EXAMPLE:
    +-------------+-------------+
    | Header 1    | Header 2    |
    +=============+=============+
    | Cell 1      | Cell 2      |
    +-------------+-------------+

    **IGNORE:**
    - Watermarks, page numbers, images
    - Text outside tables
    - Equations, checkboxes, footnotes
    """


        # Create messages in required format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt}, 
            ]},
        ]
        
        # Apply chat template
        print("Applying chat template...")
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process inputs
        print("Processing inputs...")
        inputs = processor(
            text=[text], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)
        
        # Run inference
        print("\nStarting OCR extraction...")
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False
            )
        
        # Decode results
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(
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

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Setup environment
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Model cache directory: {CACHE_DIR}")
    
    if torch.cuda.is_available():
        try:
            nvmlInit()
            print("GPU memory monitoring enabled")
        except:
            print("Could not initialize NVML for memory monitoring")
    
    # Load model
    model, processor = load_ocr_components()
    
    if model and processor:
        print("\n" + "="*50)
        print(f"Processing: {os.path.basename(IMAGE_PATH)}")
        print("="*50)
        
        try:
            # Run OCR
            result = run_ocr(IMAGE_PATH, model, processor)
            
            # Save results
            output_path = "extracted_tables.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)
                
            print(f"\n✓ OCR completed successfully!")
            print(f"Output saved to: {output_path}")
            
            # Show sample output
            print("\n" + "="*50)
            print("FIRST 500 CHARACTERS:")
            print("="*50)
            print(result[:500] + "..." if len(result) > 500 else result)
            
        except Exception as e:
            print(f"\n✗ Processing error: {str(e)}")
    else:
        print("\nOCR initialization failed. Check model loading errors.")