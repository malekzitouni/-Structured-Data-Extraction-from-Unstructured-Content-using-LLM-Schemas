from transformers import AutoModelForImageTextToText
import os
import torch

MODEL_PATH      = "nanonets/Nanonets-OCR-s"
CACHE_DIR       = "C:/models"
ARCHITECTURE_FP = os.path.join(CACHE_DIR, "model_architecture.txt")

def save_model_architecture(model, file_path):
    """(As you already have it.)"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write(f"MODEL ARCHITECTURE: {MODEL_PATH}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Class: {type(model).__name__}\n")
        f.write(f"Device: {model.device}\n")
        f.write(f"Data Type: {model.dtype}\n")
        f.write(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        if hasattr(model, 'encoder'):
            f.write("="*50 + "\nVISION ENCODER STRUCTURE\n" + "="*50 + "\n")
            f.write(str(model.encoder))
        if hasattr(model, 'decoder'):
            f.write("\n\n" + "="*50 + "\nTEXT DECODER STRUCTURE\n" + "="*50 + "\n")
            f.write(str(model.decoder))
        f.write("\n\n" + "="*50 + "\nOTHER COMPONENTS\n" + "="*50 + "\n")
        for name, module in model.named_children():
            if name not in ('encoder','decoder'):
                f.write(f"\n{name}:\n{module}\n")

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1) Load only the model (no processor needed here)
    print("Loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,    # or float16 if you have GPU
        device_map="auto",            # let 🤗 place it on GPU/CPU
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"Model loaded on {model.device} as {model.dtype}")

    # 2) Save its architecture
    save_model_architecture(model, ARCHITECTURE_FP)
    print(f"\n✓ Architecture dumped to {ARCHITECTURE_FP}\n")

    # 3) Print the first 20 lines for a quick look
    with open(ARCHITECTURE_FP, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 20:
                print(line.rstrip())
            else:
                print("… (see full architecture in file)")
                break
