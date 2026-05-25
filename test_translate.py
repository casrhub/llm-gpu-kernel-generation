"""
Test the Fireworks-based pytorch_to_triton translator.

Runs generate_kernel() on a simple elementwise operation and prints
the per-attempt validation results.
"""
import torch
from dotenv import load_dotenv

load_dotenv()

from src.translator.pytorch_to_triton import generate_kernel

pytorch_op  = "out = x * y + z"
input_shapes = {"x": [1024], "y": [1024], "z": [1024]}

print(f"Translating: {pytorch_op}")
print(f"Input shapes: {input_shapes}")

result = generate_kernel(
    pytorch_code=pytorch_op,
    input_shapes=input_shapes,
    # Supply pytorch_fn + test_inputs if you have a CUDA GPU:
    # pytorch_fn=lambda x, y, z: x * y + z,
    # test_inputs=[torch.randn(1024), torch.randn(1024), torch.randn(1024)],
    generation_model="accounts/fireworks/models/kimi-k2p5",
    repair_model="accounts/fireworks/models/gpt-oss-120b",
    max_attempts=3,
    verbose=True,
)

print("\n=== Result ===")
print(f"Success:  {result['success']}")
print(f"Attempts: {result['attempts']}")
print(f"\n=== Generated Kernel ===")
print(result["code"])

if result["history"]:
    print("\n=== Attempt History ===")
    for entry in result["history"]:
        print(f"  Attempt {entry['attempt']} failed [{entry['layer']}]: {entry['errors']}")
