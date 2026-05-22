"""
Quick test for the pytorch_to_triton translator.

Stage 1: syntax validation (no GPU needed)
Stage 2: correctness check (requires CUDA GPU)
"""
import torch
from dotenv import load_dotenv

load_dotenv()

from src.translator.pytorch_to_triton import translate, validate

# --- 1. Generate the kernel ---
pytorch_op = "out = x * y + z"
input_shapes = {"x": [1024], "y": [1024], "z": [1024]}

print("Translating:", pytorch_op)
print("Input shapes:", input_shapes)
print()

kernel_code = translate(pytorch_op, input_shapes=input_shapes, verbose=True)

print("=== Generated Kernel ===")
print(kernel_code)
print()

# --- 2. Syntax validation (no GPU needed) ---
result = validate(kernel_code)
print("=== Validation ===")
print(f"  Syntax valid : {result['syntax_valid']}")
print(f"  Imports ok   : {result['imports_ok']}")
print(f"  Correctness  : {result['correctness']}  (None = not checked)")
if result["errors"]:
    print(f"  Errors       : {result['errors']}")
print()

# --- 3. Correctness check (only runs if CUDA is available) ---
if torch.cuda.is_available():
    print("CUDA available — running correctness check...")

    x = torch.randn(1024)
    y = torch.randn(1024)
    z = torch.randn(1024)

    def pytorch_baseline(x, y, z):
        return x * y + z

    result = validate(
        kernel_code,
        pytorch_fn=pytorch_baseline,
        test_inputs=[x, y, z],
    )
    print(f"  Correctness  : {result['correctness']}")
    if result["errors"]:
        print(f"  Errors       : {result['errors']}")
else:
    print("No CUDA GPU detected — skipping correctness check.")
    print("(Triton kernels require an NVIDIA GPU to run)")
