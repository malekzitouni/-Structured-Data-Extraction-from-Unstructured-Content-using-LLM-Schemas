from pathlib import Path

# Path to your local cache
model_dir = Path(r"C:\Users\Pc\.cache\docling-models")

if not model_dir.exists():
    print(f"Directory {model_dir} does not exist.")
else:
    print(f"\nContents of {model_dir}:\n")
    for path in model_dir.rglob("*"):
        if path.is_file():
            print(" -", path.relative_to(model_dir))

