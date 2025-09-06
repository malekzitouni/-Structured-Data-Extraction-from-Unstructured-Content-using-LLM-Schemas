# This script is a self‑contained pipeline that loads a fine‑tuned image‑to‑text (OCR) model
# applies it to an image using a chat‑style prompt for rich formatting
# saves the extracted, structured text to disk



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
IMAGE_PATH = r"C:\Users\Pc\Desktop\axe_docl\fiche_de_paie2.jpg"
MAX_TOKENS = 1000

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
        prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
        
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
            output_path = os.path.splitext(IMAGE_PATH)[0] + "_output.txt"
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