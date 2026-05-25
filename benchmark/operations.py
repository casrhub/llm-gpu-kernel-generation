"""
benchmark/operations.py

Shared benchmark definition used by run_method.py, run_baseline.py, and compare.py.

10 operations across three categories of increasing complexity:
  - Elementwise  (5): single-pass, one output per input element
  - Reduction    (3): multi-element → scalar or smaller tensor
  - Compound     (2): require understanding of two-pass Triton algorithms

All pytorch_fn callables operate on CPU tensors.
validate_correctness() moves test_inputs to CUDA internally.
"""
import torch


def _relu(x):
    return torch.relu(x)

def _silu(x):
    return x * torch.sigmoid(x)

def _add(x, y):
    return x + y

def _fma(x, y, z):
    return x * y + z

def _mul(x, y):
    return x * y

def _sum_reduction(x):
    # Returns a 1-element tensor so the kernel output shape is defined
    return x.sum().unsqueeze(0)

def _max_reduction(x):
    return x.max().unsqueeze(0)

def _l2_norm_sq(x):
    return (x * x).sum().unsqueeze(0)

def _softmax(x):
    return torch.softmax(x, dim=0)

def _layer_norm(x):
    mean = x.mean()
    std  = torch.sqrt(x.var() + 1e-5)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Benchmark operations
# Each entry:
#   id           — unique string used as filename stem
#   category     — "elementwise" | "reduction" | "compound"
#   pytorch_code — string given to generate_kernel / baseline_runner
#   input_shapes — dict passed to generate_kernel for prompt context
#   pytorch_fn   — callable(CPU tensors) → CPU tensor  (correctness baseline)
#   test_inputs  — list of CPU tensors with seed=42
# ---------------------------------------------------------------------------

torch.manual_seed(42)

BENCHMARK_OPS = [
    # ── Elementwise ──────────────────────────────────────────────────────────
    {
        "id":           "ew_01_add",
        "category":     "elementwise",
        "pytorch_code": "out = x + y",
        "input_shapes": {"x": [1024], "y": [1024]},
        "pytorch_fn":   _add,
        "test_inputs":  [torch.randn(1024), torch.randn(1024)],
    },
    {
        "id":           "ew_02_fma",
        "category":     "elementwise",
        "pytorch_code": "out = x * y + z",
        "input_shapes": {"x": [1024], "y": [1024], "z": [1024]},
        "pytorch_fn":   _fma,
        "test_inputs":  [torch.randn(1024), torch.randn(1024), torch.randn(1024)],
    },
    {
        "id":           "ew_03_mul",
        "category":     "elementwise",
        "pytorch_code": "out = x * y",
        "input_shapes": {"x": [1024], "y": [1024]},
        "pytorch_fn":   _mul,
        "test_inputs":  [torch.randn(1024), torch.randn(1024)],
    },
    {
        "id":           "ew_04_relu",
        "category":     "elementwise",
        "pytorch_code": "out = torch.relu(x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _relu,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_05_silu",
        "category":     "elementwise",
        "pytorch_code": "out = x * torch.sigmoid(x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _silu,
        "test_inputs":  [torch.randn(1024)],
    },

    # ── Reduction ─────────────────────────────────────────────────────────────
    {
        "id":           "rd_01_sum",
        "category":     "reduction",
        "pytorch_code": "out = x.sum()",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _sum_reduction,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "rd_02_max",
        "category":     "reduction",
        "pytorch_code": "out = x.max()",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _max_reduction,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "rd_03_l2",
        "category":     "reduction",
        "pytorch_code": "out = (x * x).sum()",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _l2_norm_sq,
        "test_inputs":  [torch.randn(1024)],
    },

    # ── Compound ─────────────────────────────────────────────────────────────
    {
        "id":           "cp_01_softmax",
        "category":     "compound",
        "pytorch_code": "out = torch.softmax(x, dim=0)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _softmax,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "cp_02_layernorm",
        "category":     "compound",
        "pytorch_code": "out = (x - x.mean()) / torch.sqrt(x.var() + 1e-5)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _layer_norm,
        "test_inputs":  [torch.randn(1024)],
    },
]
