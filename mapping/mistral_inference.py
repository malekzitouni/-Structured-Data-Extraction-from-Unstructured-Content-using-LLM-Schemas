from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Path to your local Mistral model
model_path = r"C:\Users\Pc\.cache\huggingface\hub\models--mistralai--Mistral-7B-Instruct-v0.2"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model in 4-bit quantization (fits in 6GB VRAM)...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",          # automatically chooses GPU if available
    torch_dtype=torch.float16,  # lighter precision
    load_in_4bit=True           # quantize to fit in small VRAM
)

print("Creating text-generation pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Test prompt
prompt = " what is the evolution of AI in the last 10 years? How has it changed the world?"

print("\nGenerating response...")
output = generator(
    prompt,
    max_new_tokens=100,   # keep small for speed
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print("\n--- MODEL OUTPUT ---\n")
print(output[0]["generated_text"])
