"""
Test the full two-step pipeline:
    Step 1 — generate_kernel()  : generate + validate a Triton kernel
    Step 2 — optimize_kernel()  : autotune BLOCK_SIZE on the actual GPU

Requires a CUDA GPU. Run in Colab:
    %run test_optimize.py
    %run test_optimize.py --op rd_01_sum
    %run test_optimize.py --op cp_01_softmax
    %run test_optimize.py --op ew_02_fma --model accounts/fireworks/models/llama-v3p1-70b-instruct
"""
import sys
import os
import argparse
import torch
from dotenv import load_dotenv

load_dotenv()

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise EnvironmentError(
        "FIREWORKS_API_KEY not set. Run this first:\n"
        "  from google.colab import userdata; import os\n"
        "  os.environ['FIREWORKS_API_KEY'] = userdata.get('FIREWORKS_API_KEY')"
    )

from src.translator.pytorch_to_triton import generate_kernel
from src.optimizer.autotune import optimize_kernel
from benchmark.operations import BENCHMARK_OPS

# ── Parse operation argument ──────────────────────────────────────────────────
DEFAULT_MODEL = "accounts/fireworks/models/gpt-oss-20b"

parser = argparse.ArgumentParser()
parser.add_argument("--op", default="ew_02_fma",
                    help="Operation id from benchmark/operations.py")
parser.add_argument("--model", default=DEFAULT_MODEL,
                    help="Fireworks model ID for generation and repair")
args, _ = parser.parse_known_args()

op = next((o for o in BENCHMARK_OPS if o["id"] == args.op), None)
if op is None:
    ids = [o["id"] for o in BENCHMARK_OPS]
    print(f"Unknown op '{args.op}'. Available: {ids}")
    sys.exit(1)

pytorch_code = op["pytorch_code"]
input_shapes = op["input_shapes"]
pytorch_fn   = op["pytorch_fn"]
test_inputs  = op["test_inputs"]
category     = op["category"]

print(f"Operation : {op['id']}  ({category})")
print(f"Model     : {args.model}")
print(f"Code      : {pytorch_code}")

# ── Step 1: Generate and validate ────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — generate_kernel()")
print("=" * 60)

gen_result = generate_kernel(
    pytorch_code     = pytorch_code,
    input_shapes     = input_shapes,
    pytorch_fn       = pytorch_fn,
    test_inputs      = test_inputs,
    generation_model = args.model,
    repair_model     = args.model,
    max_attempts     = 3,
    api_key          = FIREWORKS_API_KEY,
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
    output_path = f"{op['id']}_optimized.py",
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
        print(f"Saved to             : {op['id']}_optimized.py")
else:
    print(f"Errors: {opt_result['errors']}")
