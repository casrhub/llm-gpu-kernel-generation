"""
src/optimizer/autotune.py

Post-validation kernel optimizer using Triton's autotuning.

Takes a validated kernel from generate_kernel() and finds the optimal
hardware-specific hyperparameters (BLOCK_SIZE, num_warps) by benchmarking
a predefined config space on the actual GPU.

This is a separate clean step from generation and validation:
    generate_kernel()  →  correctness guaranteed
    optimize_kernel()  →  performance maximized
"""
import re
import os
import tempfile
import importlib.util
from typing import Optional


# ---------------------------------------------------------------------------
# Config spaces per operation category.
# The same config space is searched on every GPU — the hardware decides which
# config wins. A T4, A100, and H100 will likely pick different winners from
# the same list because they have different warp sizes, L2 cache sizes, and
# memory bandwidth.
# ---------------------------------------------------------------------------
CONFIGS = {
    "elementwise": [
        {"BLOCK_SIZE": 256,  "num_warps": 4},
        {"BLOCK_SIZE": 512,  "num_warps": 4},
        {"BLOCK_SIZE": 1024, "num_warps": 4},
        {"BLOCK_SIZE": 1024, "num_warps": 8},
        {"BLOCK_SIZE": 2048, "num_warps": 8},
    ],
    "reduction": [
        {"BLOCK_SIZE": 256,  "num_warps": 4},
        {"BLOCK_SIZE": 512,  "num_warps": 4},
        {"BLOCK_SIZE": 1024, "num_warps": 8},
        {"BLOCK_SIZE": 2048, "num_warps": 8},
    ],
    "compound": [
        {"BLOCK_SIZE": 128,  "num_warps": 4},
        {"BLOCK_SIZE": 256,  "num_warps": 4},
        {"BLOCK_SIZE": 512,  "num_warps": 8},
        {"BLOCK_SIZE": 1024, "num_warps": 8},
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_autotune_decorator(category: str) -> str:
    """Build the @triton.autotune decorator string for a given category."""
    configs = CONFIGS.get(category, CONFIGS["elementwise"])
    config_lines = [
        f'        triton.Config({{"BLOCK_SIZE": {c["BLOCK_SIZE"]}}}, num_warps={c["num_warps"]})'
        for c in configs
    ]
    return (
        "@triton.autotune(\n"
        "    configs=[\n"
        + ",\n".join(config_lines) + "\n"
        "    ],\n"
        '    key=["n_elements"],\n'
        ")\n"
    )


def _inject_autotune(kernel_code: str, category: str) -> str:
    """
    Transform validated kernel code to use @triton.autotune.

    Three transformations:
      1. Prepend @triton.autotune(...) before @triton.jit
      2. Remove BLOCK_SIZE=<value> keyword arg from the kernel launch call
         inside the launcher — autotune injects it automatically from the
         winning config, so passing it explicitly causes an error
      3. Remove the hardcoded BLOCK_SIZE = <int> assignment in the launcher body
         since that value is now managed by autotune
    """
    # 1. Inject autotune decorator before @triton.jit
    decorator = _build_autotune_decorator(category)
    code = kernel_code.replace("@triton.jit", decorator + "@triton.jit", 1)

    # 2. Remove BLOCK_SIZE=<value> from the kernel call (e.g. BLOCK_SIZE=BLOCK_SIZE)
    code = re.sub(r",\s*BLOCK_SIZE\s*=\s*\w+", "", code)

    # 3. Remove the hardcoded BLOCK_SIZE = <int> line from the launcher body
    code = re.sub(r"\n[ \t]*BLOCK_SIZE\s*=\s*\d+[ \t]*\n", "\n", code)

    return code


def _find_launcher(namespace: dict):
    """Return the launcher function (plain Python def) from the module."""
    import inspect
    return next(
        (
            v for k, v in namespace.items()
            if inspect.isfunction(v)
            and not k.startswith("_")
            and k not in ("torch", "triton", "tl")
            and k[0].islower()
        ),
        None,
    )


def _find_autotuner(namespace: dict):
    """Return the @triton.autotune-wrapped kernel object from the module."""
    try:
        from triton.runtime.autotuner import Autotuner
        return next(
            (v for k, v in namespace.items() if isinstance(v, Autotuner)),
            None,
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# optimize_kernel() — main entry point
# ---------------------------------------------------------------------------

def optimize_kernel(
    kernel_code: str,
    test_inputs: list,
    pytorch_fn,
    category: str = "elementwise",
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Find the optimal hardware-specific hyperparameters for a validated Triton kernel.

    The LLM that generated the kernel picks a single BLOCK_SIZE (almost always 1024,
    because that is what the system prompt example shows). This is a guess — the
    optimal BLOCK_SIZE depends on the specific GPU (T4, A100, H100 all have different
    warp sizes, cache sizes, and memory bandwidth), and no LLM can know that at
    generation time.

    This function replaces that single guess with a search over a predefined config
    space. Triton runs the kernel with each config on the actual GPU, measures
    wall-clock time, and selects the fastest. The winning config is then cached —
    next time the kernel is called with the same n_elements it runs immediately
    without re-benchmarking.

    Two code transformations are applied to the validated kernel:
      1. @triton.autotune(...) is prepended before @triton.jit
      2. BLOCK_SIZE=<value> is removed from the launcher call — autotune injects
         it automatically from the winning config, passing it explicitly errors

    Args:
        kernel_code:  Validated kernel source string from generate_kernel()
        test_inputs:  List of CPU torch.Tensors (same as used in validation)
        pytorch_fn:   CPU callable — benchmarked on GPU as the baseline
        category:     "elementwise" | "reduction" | "compound"
                      Controls which config space is searched
        output_path:  If provided, write the optimized kernel code to this path
        verbose:      Print config search results and timing breakdown

    Returns:
        dict with keys:
            success          (bool)
            optimized_code   (str)    — transformed kernel with @triton.autotune
            best_block_size  (int)    — winning BLOCK_SIZE chosen by the GPU
            best_num_warps   (int)    — winning num_warps chosen by the GPU
            ms_kernel        (float)  — winning config wall-clock time in ms
            ms_pytorch       (float)  — PyTorch baseline time on GPU in ms
            speedup          (float)  — ms_pytorch / ms_kernel
            errors           (list[str])
    """
    import torch
    import triton

    result = {
        "success":         False,
        "optimized_code":  None,
        "best_block_size": None,
        "best_num_warps":  None,
        "ms_kernel":       None,
        "ms_pytorch":      None,
        "speedup":         None,
        "errors":          [],
    }

    if not torch.cuda.is_available():
        result["errors"].append("CUDA GPU required for autotuning — skipped")
        return result

    # ── Transform kernel code ─────────────────────────────────────────────────
    optimized_code = _inject_autotune(kernel_code, category)
    result["optimized_code"] = optimized_code

    if verbose:
        n_configs = len(CONFIGS.get(category, CONFIGS["elementwise"]))
        print("=" * 60)
        print(f"AUTOTUNE  category={category}  configs={n_configs}")
        print("=" * 60)
        print(optimized_code)
        print("=" * 60)

    # ── Import transformed kernel from temp file ──────────────────────────────
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(optimized_code)
        tmp.close()
        spec   = importlib.util.spec_from_file_location("_triton_kernel_opt", tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        result["errors"].append(f"Failed to import optimized kernel: {e}")
        return result
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    launcher  = _find_launcher(vars(module))
    autotuner = _find_autotuner(vars(module))

    if launcher is None:
        result["errors"].append("No launcher function found in optimized kernel")
        return result

    # ── Move inputs to GPU ────────────────────────────────────────────────────
    cuda_inputs = [
        t.cuda() if isinstance(t, torch.Tensor) else t for t in test_inputs
    ]

    # ── First call triggers autotuning: benchmarks all configs, picks winner ──
    try:
        _ = launcher(*cuda_inputs)
        torch.cuda.synchronize()
    except Exception as e:
        result["errors"].append(f"Autotuning failed: {e}")
        return result

    # ── Benchmark the winning config ──────────────────────────────────────────
    try:
        ms_kernel = triton.testing.do_bench(lambda: launcher(*cuda_inputs))
    except Exception as e:
        result["errors"].append(f"Kernel benchmark failed: {e}")
        return result

    # ── Benchmark PyTorch baseline on GPU ─────────────────────────────────────
    try:
        pytorch_cuda = [
            t.cuda() if isinstance(t, torch.Tensor) else t for t in test_inputs
        ]
        ms_pytorch = triton.testing.do_bench(lambda: pytorch_fn(*pytorch_cuda))
    except Exception as e:
        result["errors"].append(f"PyTorch benchmark failed: {e}")
        return result

    speedup = ms_pytorch / ms_kernel

    # ── Extract best config from autotuner ───────────────────────────────────
    best_block_size = None
    best_num_warps  = None
    if autotuner is not None and hasattr(autotuner, "best_config"):
        best_block_size = autotuner.best_config.kwargs.get("BLOCK_SIZE")
        best_num_warps  = autotuner.best_config.num_warps

    if verbose:
        print(f"Best BLOCK_SIZE : {best_block_size}")
        print(f"Best num_warps  : {best_num_warps}")
        print(f"Kernel time     : {ms_kernel:.3f} ms")
        print(f"PyTorch time    : {ms_pytorch:.3f} ms")
        print(f"Speedup         : {speedup:.2f}x")

    # ── Optionally save optimized kernel to disk ──────────────────────────────
    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(optimized_code)
        if verbose:
            print(f"Saved to {output_path}")

    result.update({
        "success":         True,
        "best_block_size": best_block_size,
        "best_num_warps":  best_num_warps,
        "ms_kernel":       ms_kernel,
        "ms_pytorch":      ms_pytorch,
        "speedup":         speedup,
    })

    return result
