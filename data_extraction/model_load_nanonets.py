from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import os
import warnings
import torch

# ===== MAIN SCRIPT =====
# Set verbosity level to suppress specific warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*huggingface_hub.*symlinks.*")
warnings.filterwarnings("ignore", message=".*slow image processor.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
warnings.filterwarnings("ignore", message=".*Some parameters are on the meta device.*")

# Set your local cache directory on C drive
cache_dir = "C:/models"  
os.makedirs(cache_dir, exist_ok=True)

model_path = "nanonets/Nanonets-OCR-s"

print("="*50)
print("Starting OCR Model Initialization")
print("="*50)

# Initialize components
model = None
processor = None
model_ready = False

try:
    # Load model with disk offloading
    print("[1/3] Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        cache_dir=cache_dir,
        offload_folder=cache_dir,
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"✓ Model loaded successfully | Device: {model.device} | Precision: {model.dtype}")
    
    # Load tokenizer and processor
    print("[2/3] Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir, use_fast=True)
    print("✓ Tokenizer and processor loaded")
    
    # Final verification
    print("[3/3] Verifying component integration...")
    if model and processor:
        # Quick validation without test inference
        required_attrs = ['generate', 'device', 'dtype']
        if all(hasattr(model, attr) for attr in required_attrs):
            model_ready = True
            print("✓ All components ready for OCR")
        else:
            print("✗ Model missing required attributes")
    else:
        print("✗ Critical components not initialized")
        
except Exception as e:
    print(f"✗ Initialization failed: {str(e)}")

def ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=4096):
    """Perform OCR extraction using Nanonets model"""
    if not model_ready:
        raise RuntimeError("OCR aborted: Model not properly initialized")
    
    prompt = """Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes."""
    
    try:
        image = Image.open(image_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt}, 
            ]},
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=False
        )
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
        
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {str(e)}")

# OCR Execution
print("\n" + "="*50)
if model_ready:
    print("Starting OCR Extraction")
    print("="*50)
    
    image_path = r"C:\Users\Pc\Desktop\axe_docl\batch1-0001.jpg"
    try:
        result = ocr_page_with_nanonets_s(image_path, model, processor, max_new_tokens=15000)
        print(result)
    except Exception as e:
        print(f"✗ {str(e)}")
else:
    print("OCR Initialization Failed")
    print("="*50)
    print("Please check the error messages above")
    print("Possible solutions:")
    print("- Verify internet connection for model download")
    print("- Check available disk space in C:/models")
    print("- Ensure PyTorch/CUDA compatibility")
    print("- Validate image file path exists")