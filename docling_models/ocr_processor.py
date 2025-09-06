from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
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
MAX_IMAGE_DIM = 1024  # Maximum dimension for image processing

# ===== QUANTIZATION CONFIG =====
def get_quantization_config():
    """Create optimized quantization configuration for 6GB VRAM"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # Optimized for inference
        bnb_4bit_use_double_quant=True,      # Additional quantization savings
        bnb_4bit_compute_dtype=torch.float16 # Use FP16 for computations
    )

# ===== ENVIRONMENT DIAGNOSTICS =====
def print_environment_info():
    """Print environment information with GPU diagnostics"""
    print("\n" + "="*50)
    print("ENVIRONMENT INFORMATION")
    print("="*50)
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory/1024**3:.2f} GB")
        
        # Initialize NVML for detailed memory info
        try:
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(device)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            print(f"Current GPU Usage: {mem_info.used/1024**3:.2f} GB / {mem_info.total/1024**3:.2f} GB")
        except:
            print("NVML initialization failed - using PyTorch memory stats")
            print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"Cached:    {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# ===== MEMORY MONITOR =====
class GPUMonitor:
    """Context manager for GPU memory monitoring"""
    def __init__(self, name):
        self.name = name
        self.start_mem = 0
        
    def __enter__(self):
        if torch.cuda.is_available():
            self.start_mem = torch.cuda.memory_allocated()
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if torch.cuda.is_available():
            end_mem = torch.cuda.memory_allocated()
            delta = end_mem - self.start_mem
            print(f"GPU Memory Δ: {delta/1024**2:.2f} MB ({self.name})")

# ===== MODEL LOADING =====
def load_ocr_components():
    """Load model components with advanced optimizations"""
    try:
        print_environment_info()
        
        # Determine precision and device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"\nUsing {'GPU' if device == 'cuda' else 'CPU'} with {'half' if dtype == torch.float16 else 'full'} precision")
        
        # Create quantization config
        quantization_config = get_quantization_config() if device == "cuda" else None
        
        # Load model with optimizations
        print("Loading OCR model with optimizations...")
        start_time = time.time()
        
        with GPUMonitor("Model Loading"):
            model = AutoModelForImageTextToText.from_pretrained(
                MODEL_PATH,
                torch_dtype=dtype,
                device_map="auto",
                cache_dir=CACHE_DIR,
                offload_folder=CACHE_DIR,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2" if device == "cuda" else None
            )
            model.eval()
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s | Device: {model.device} | Precision: {model.dtype}")
        
        # Load processor with optimizations
        print("Loading processor with optimizations...")
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            cache_dir=CACHE_DIR,
            use_fast=True,  # Use fast tokenizer
            truncation=True  # Enable truncation by default
        )
        print("✓ Processor loaded")
        
        # Warm up model with a small input
        if device == "cuda":
            print("Warming up model...")
            with torch.no_grad():
                dummy_input = processor(
                    text="Warmup", 
                    images=Image.new("RGB", (100, 100)),
                    return_tensors="pt"
                ).to(model.device)
                model.generate(**dummy_input, max_new_tokens=10)
            torch.cuda.empty_cache()
            print("✓ Model warmed up")
        
        return model, processor
        
    except Exception as e:
        print(f"✗ Load failed: {str(e)}")
        return None, None

# ===== IMAGE OPTIMIZATION =====
def optimize_image(image_path, max_dim=MAX_IMAGE_DIM):
    """Optimize image for processing with VRAM constraints"""
    print("\nOptimizing image for processing...")
    image = Image.open(image_path)
    print(f"Original size: {image.size} | Mode: {image.mode}")
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
        print(f"Converted to RGB")
    
    # Calculate new dimensions preserving aspect ratio
    width, height = image.size
    scale = min(max_dim/width, max_dim/height)
    
    if scale < 1:
        new_size = (int(width * scale), int(height * scale))
        print(f"Resizing to {new_size} (scale: {scale:.2f})")
        image = image.resize(new_size, Image.LANCZOS)
    
    print(f"Final size: {image.size}")
    return image

# ===== OPTIMIZED OCR FUNCTION =====
def run_ocr(image_path, model, processor):
    """Perform OCR extraction with advanced optimizations"""
    if not all([model, processor]):
        raise RuntimeError("Model not initialized")
    
    try:
        # Optimize image before processing
        image = optimize_image(image_path)
        
        # Define optimized prompt
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
        with GPUMonitor("Template Processing"):
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        
        # Process inputs with memory monitoring
        print("Processing inputs...")
        with GPUMonitor("Input Processing"):
            inputs = processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt",
                truncation=True,  # Ensure truncation
                max_length=512     # Limit input length
            ).to(model.device)
        
        # Configure generation for efficiency
        gen_config = {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": False,            # Disable sampling for deterministic output
            "num_beams": 1,                # Use greedy search (fastest)
            "early_stopping": True,        # Stop when all beams finish
            "use_cache": True,             # Use KV caching
            "output_attentions": False,    # Don't return attention weights
            "output_hidden_states": False, # Don't return hidden states
            "pad_token_id": processor.tokenizer.pad_token_id,
            "eos_token_id": processor.tokenizer.eos_token_id
        }
        
        # Run inference with memory monitoring
        print("\nStarting OCR extraction...")
        start_time = time.time()
        
        with GPUMonitor("OCR Generation"), torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=True):
            outputs = model.generate(**inputs, **gen_config)
        
        # Decode results
        print("Decoding results...")
        with GPUMonitor("Result Decoding"):
            generated_ids = outputs[:, inputs.input_ids.shape[1]:]
            result = processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )[0]
        
        inference_time = time.time() - start_time
        print(f"✓ OCR completed in {inference_time:.2f} seconds")
        
        # Calculate tokens per second
        tokens_generated = generated_ids.shape[1]
        tokens_per_sec = tokens_generated / inference_time if inference_time > 0 else 0
        print(f"Tokens generated: {tokens_generated} | Tokens/sec: {tokens_per_sec:.2f}")
        
        return result
        
    except Exception as e:
        raise RuntimeError(f"OCR processing failed: {str(e)}")
    finally:
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # Setup environment
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Model cache directory: {CACHE_DIR}")
    
    # Initialize GPU monitoring
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
            # Run OCR with memory monitoring
            with GPUMonitor("Total OCR Process"):
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
            
            # Performance summary
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                print("\n" + "="*50)
                print("PERFORMANCE SUMMARY")
                print("="*50)
                print(f"GPU: {torch.cuda.get_device_name(device)}")
                print(f"Peak memory usage: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
                print(f"Peak memory reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
            
        except Exception as e:
            print(f"\n✗ Processing error: {str(e)}")
    else:
        print("\nOCR initialization failed. Check model loading errors.")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()