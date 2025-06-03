from pathlib import Path
import os

model_dir = Path(r"G:\apps\Grain size\models")
print(f"Checking directory: {model_dir}")
print(f"Directory exists: {model_dir.exists()}")

# List all files to see everything
print("All files in directory:")
if model_dir.exists():
    for item in os.listdir(model_dir):
        print(f"  - {item}")
else:
    print(f"  Directory {model_dir} does not exist.")

print("\nFiles found by glob('*.pt'):")
if model_dir.exists():
    model_files = list(model_dir.glob("*.pt"))
    if not model_files:
        print("  No *.pt files found by glob.")
    else:
        for f in model_files:
            print(f"  - {f.name}")
else:
    print(f"  Cannot glob because directory {model_dir} does not exist.") 