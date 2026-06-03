"""
Test the full two-step pipeline:
    Step 1 — generate_kernel()  : generate + validate a Triton kernel
    Step 2 — optimize_kernel()  : autotune BLOCK_SIZE on the actual GPU

Requires a CUDA GPU. Run in Colab:
    %run test_optimize.py
"""
import torch
from dotenv import load_dotenv

load_dotenv()

from src.translator.pytorch_to_triton import generate_kernel
from src.optimizer.autotune import optimize_kernel

# ── Operation to test ─────────────────────────────────────────────────────────
pytorch_code = "out = x * y + z"
input_shapes = {"x": [1024], "y": [1024], "z": [1024]}
pytorch_fn   = lambda x, y, z: x * y + z
test_inputs  = [torch.randn(1024), torch.randn(1024), torch.randn(1024)]
category     = "elementwise"

# ── Step 1: Generate and validate ────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — generate_kernel()")
print("=" * 60)

gen_result = generate_kernel(
    pytorch_code     = pytorch_code,
    input_shapes     = input_shapes,
    pytorch_fn       = pytorch_fn,
    test_inputs      = test_inputs,
    generation_model = "accounts/fireworks/models/gpt-oss-20b",
    repair_model     = "accounts/fireworks/models/gpt-oss-20b",
    max_attempts     = 3,
    verbose          = True,
)

print(f"\nGeneration success : {gen_result['success']}")
print(f"Attempts           : {gen_result['attempts']}")

if not gen_result["success"]:
    print("Generation failed — cannot proceed to optimization")
    exit(1)

# ── Step 2: Optimize ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — optimize_kernel()")
print("=" * 60)

opt_result = optimize_kernel(
    kernel_code = gen_result["code"],
    test_inputs = test_inputs,
    pytorch_fn  = pytorch_fn,
    category    = category,
    output_path = "fma_optimized.py",
    verbose     = True,
)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Optimization success : {opt_result['success']}")

if opt_result["success"]:
    print(f"Best BLOCK_SIZE      : {opt_result['best_block_size']}")
    print(f"Best num_warps       : {opt_result['best_num_warps']}")
    print(f"Kernel time          : {opt_result['ms_kernel']:.3f} ms")
    print(f"PyTorch time         : {opt_result['ms_pytorch']:.3f} ms")
    print(f"Speedup              : {opt_result['speedup']:.2f}x")
    if opt_result.get("output_path"):
        print(f"Saved to             : fma_optimized.py")
else:
    print(f"Errors: {opt_result['errors']}")
