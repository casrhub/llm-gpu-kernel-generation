"""
benchmark/operations.py

Shared benchmark definition used by run_method.py, run_baseline.py, and compare.py.

25 operations across three categories of increasing complexity:
  - Elementwise  (12): single-pass, one output per input element
  - Reduction     (7): multi-element → scalar or smaller tensor
  - Compound      (6): require understanding of two-pass Triton algorithms

All pytorch_fn callables operate on CPU tensors.
validate_correctness() moves test_inputs to CUDA internally.
"""
import torch


# ── Reference functions (CPU) ────────────────────────────────────────────────

# Elementwise
def _add(x, y):      return x + y
def _fma(x, y, z):   return x * y + z
def _mul(x, y):      return x * y
def _relu(x):        return torch.relu(x)
def _silu(x):        return x * torch.sigmoid(x)
def _exp(x):         return torch.exp(x)
def _sigmoid(x):     return torch.sigmoid(x)
def _tanh(x):        return torch.tanh(x)
def _leaky_relu(x):  return torch.where(x > 0, x, 0.01 * x)
def _clamp(x):       return torch.clamp(x, min=-1.0, max=1.0)
def _scale_bias(x):  return 2.0 * x + 0.5
def _squared(x):     return x * x

# Reduction — return unsqueeze(0) so correctness check has a defined shape
def _sum_reduction(x):  return x.sum().unsqueeze(0)
def _max_reduction(x):  return x.max().unsqueeze(0)
def _l2_norm_sq(x):     return (x * x).sum().unsqueeze(0)
def _mean_reduction(x): return x.mean().unsqueeze(0)
def _min_reduction(x):  return x.min().unsqueeze(0)
def _var_reduction(x):  return ((x - x.mean()) ** 2).mean().unsqueeze(0)
def _norm_reduction(x): return torch.norm(x).unsqueeze(0)

# Compound
def _softmax(x):         return torch.softmax(x, dim=0)
def _layer_norm(x):
    mean = x.mean()
    std  = torch.sqrt(x.var() + 1e-5)
    return (x - mean) / std
def _gelu(x):            return x * torch.sigmoid(1.702 * x)
def _rms_norm(x):        return x / torch.sqrt((x * x).mean() + 1e-5)
def _log_softmax(x):     return torch.log_softmax(x, dim=0)
def _scaled_softmax(x):  return torch.softmax(x / (64 ** 0.5), dim=0)


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
    {
        "id":           "ew_06_exp",
        "category":     "elementwise",
        "pytorch_code": "out = torch.exp(x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _exp,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_07_sigmoid",
        "category":     "elementwise",
        "pytorch_code": "out = torch.sigmoid(x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _sigmoid,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_08_tanh",
        "category":     "elementwise",
        "pytorch_code": "out = torch.tanh(x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _tanh,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_09_leaky_relu",
        "category":     "elementwise",
        "pytorch_code": "out = torch.where(x > 0, x, 0.01 * x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _leaky_relu,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_10_clamp",
        "category":     "elementwise",
        "pytorch_code": "out = torch.clamp(x, min=-1.0, max=1.0)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _clamp,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_11_scale_bias",
        "category":     "elementwise",
        "pytorch_code": "out = 2.0 * x + 0.5",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _scale_bias,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "ew_12_squared",
        "category":     "elementwise",
        "pytorch_code": "out = x * x",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _squared,
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
    {
        "id":           "rd_04_mean",
        "category":     "reduction",
        "pytorch_code": "out = x.mean()",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _mean_reduction,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "rd_05_min",
        "category":     "reduction",
        "pytorch_code": "out = x.min()",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _min_reduction,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "rd_06_var",
        "category":     "reduction",
        "pytorch_code": "out = ((x - x.mean()) ** 2).mean()",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _var_reduction,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "rd_07_norm",
        "category":     "reduction",
        "pytorch_code": "out = torch.norm(x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _norm_reduction,
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
    {
        "id":           "cp_03_gelu",
        "category":     "compound",
        "pytorch_code": "out = x * torch.sigmoid(1.702 * x)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _gelu,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "cp_04_rms_norm",
        "category":     "compound",
        "pytorch_code": "out = x / torch.sqrt((x * x).mean() + 1e-5)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _rms_norm,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "cp_05_log_softmax",
        "category":     "compound",
        "pytorch_code": "out = torch.log_softmax(x, dim=0)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _log_softmax,
        "test_inputs":  [torch.randn(1024)],
    },
    {
        "id":           "cp_06_scaled_softmax",
        "category":     "compound",
        "pytorch_code": "out = torch.softmax(x / (64 ** 0.5), dim=0)",
        "input_shapes": {"x": [1024]},
        "pytorch_fn":   _scaled_softmax,
        "test_inputs":  [torch.randn(1024)],
    },
]
