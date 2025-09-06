# cli_infer.py
from pathlib import Path
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import GenerationConfig

def load_model(local_path: str):
    local_dir = Path(local_path)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local model path not found: {local_dir}")

    # Try to load to GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"local_files_only": True}
    # If you have limited memory, you can pass device_map="auto" (requires accelerate),
    # or low_cpu_mem_usage=True. Keep simple here and move to device after load.
    print(f"Loading tokenizer from {local_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(local_dir), **kwargs)
    print(f"Loading model from {local_dir} to {device} (this may take a while)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(local_dir),
            torch_dtype=torch.float16 if device=="cuda" else torch.float32,
            low_cpu_mem_usage=True,
            **kwargs
        )
    except Exception as e:
        # fallback: load without low_cpu_mem_usage
        model = AutoModelForCausalLM.from_pretrained(str(local_dir), **kwargs)

    model.to(device)
    model.eval()
    return model, tokenizer, device

def postprocess_output(text: str, stop_sequences=None):
    if not stop_sequences:
        return text.strip()
    for s in stop_sequences:
        idx = text.find(s)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()

def generate(model, tokenizer, device, prompt, max_new_tokens=256, temperature=0.7, stop_sequences=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Use only max_new_tokens (avoid passing max_length simultaneously)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else None,
        # early_stopping=True  # optional
    )

    with torch.no_grad():
        out_ids = model.generate(**inputs, **gen_kwargs)

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    # Optionally trim at stop sequences like role markers to avoid trailing role labels
    return postprocess_output(text, stop_sequences=stop_sequences)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local path to model folder")
    parser.add_argument("--prompt", required=True, help="Prompt to send to model")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path)
    out = generate(model, tokenizer, device, args.prompt, args.max_new_tokens, args.temperature)
    print("\n=== Generation ===\n")
    print(out)

if __name__ == "__main__":
    main()
