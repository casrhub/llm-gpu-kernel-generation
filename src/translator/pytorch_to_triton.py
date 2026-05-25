"""
pytorch_to_triton.py

Translates PyTorch operations to optimized Triton GPU kernels using Fireworks AI.

Key design decisions:
- GBNF grammar-based constrained decoding: the model physically cannot generate
  output that violates the required Triton structure (imports, @triton.jit).
- Three-layer validation pipeline:
    Layer 1 — static:       ast.parse + ast.walk Triton-specific rules (no GPU needed)
    Layer 2 — gpu compile:  exec() to check imports resolve and kernel is definable
    Layer 3 — correctness:  torch.allclose vs PyTorch baseline (requires CUDA GPU)
- Self-repair retry loop: each failed attempt passes the error back to the model
  as feedback, so it knows exactly what to fix.
"""
import ast
from typing import Optional

# ---------------------------------------------------------------------------
# GBNF Grammar
# Enforces required Triton structure at the token level.
# The model cannot physically generate output that violates these rules.
#
# Guarantees:
#   - Output starts with the three required imports (exact strings)
#   - @triton.jit appears immediately after the imports
#   - Everything after is free-form Python (kernel body + launcher)
# ---------------------------------------------------------------------------
TRITON_GRAMMAR = r"""
root     ::= imports nl+ "@triton.jit" nl rest
imports  ::= "import torch" nl "import triton" nl "import triton.language as tl"
rest     ::= any-char*
any-char ::= [^\x00]
nl       ::= "\n"
"""

# ---------------------------------------------------------------------------
# System prompt — Triton teaching material sent on every call.
# Kept stable so it can be prompt-cached by the provider.
# ---------------------------------------------------------------------------
TRITON_SYSTEM_PROMPT = """You are an expert GPU kernel engineer specializing in Triton.
Your task is to convert PyTorch operations into correct, efficient Triton GPU kernels.

## Required output structure (enforced by grammar — do not deviate)

import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def kernel_name_launcher(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    kernel_name[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output

## Critical rules — violations will be caught and sent back to you

1. ALWAYS add mask= to every tl.load() and tl.store() call
   mask = offsets < n_elements
   x = tl.load(x_ptr + offsets, mask=mask)          # correct
   x = tl.load(x_ptr + offsets)                      # WRONG — will fail

2. BLOCK_SIZE must be a power of 2: 32, 64, 128, 256, 512, 1024

3. ALWAYS annotate block-size constants with tl.constexpr
   BLOCK_SIZE: tl.constexpr   # correct
   BLOCK_SIZE                 # WRONG

4. ALWAYS call tl.program_id(axis=0) to get the block index

5. NO arbitrary Python control flow inside @triton.jit kernels
   Use tl.where() for conditionals, not if/else

6. Launcher must assert inputs are CUDA tensors and call the kernel with [grid](...)

## Grid computation

1D elementwise:  grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
2D tiled:        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

## Common tl functions

tl.load(ptr, mask, other=0.0)   tl.store(ptr, value, mask)
tl.sum(x, axis=0)               tl.max(x, axis=0)
tl.exp(x)   tl.log(x)          tl.sqrt(x)   tl.abs(x)
tl.where(condition, x, y)       tl.dot(a, b)
tl.zeros([M, N], dtype=tl.float32)
"""

# ---------------------------------------------------------------------------
# Valid tl.* attribute names for static analysis whitelist
# ---------------------------------------------------------------------------
KNOWN_TL_ATTRS = {
    "load", "store", "program_id", "arange", "zeros", "zeros_like", "full",
    "sum", "max", "min", "dot", "where", "exp", "log", "sqrt", "abs",
    "cast", "constexpr", "cdiv", "multiple_of", "debug_barrier",
    "atomic_add", "atomic_max", "atomic_min", "atomic_and", "atomic_or",
    "make_block_ptr", "advance", "float16", "float32", "float64",
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
}


# ---------------------------------------------------------------------------
# translate() — single LLM call with GBNF constrained decoding
# ---------------------------------------------------------------------------
def translate(
    pytorch_code: str,
    input_shapes: Optional[dict] = None,
    feedback: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = "accounts/fireworks/models/kimi-k2p5",
    verbose: bool = False,
) -> str:
    """
    Call Fireworks AI with GBNF grammar to generate a Triton kernel.

    Args:
        pytorch_code:  PyTorch operation to translate
        input_shapes:  e.g. {"x": [1024], "y": [1024]}
        feedback:      Error message from a previous failed attempt (drives self-repair)
        api_key:       Fireworks API key — defaults to FIREWORKS_API_KEY env var
        model:         Fireworks model ID
        verbose:       Print prompt and raw response

    Returns:
        Generated kernel code string (grammar guarantees structural correctness)
    """
    from fireworks.client import Fireworks

    user_message = f"Convert this PyTorch operation to a Triton kernel:\n\n{pytorch_code}"

    if input_shapes:
        shapes_str = ", ".join(f"{k}: {v}" for k, v in input_shapes.items())
        user_message += f"\n\nInput shapes: {shapes_str}"

    if feedback:
        user_message += (
            f"\n\n---\nYour previous attempt failed. Fix the following issues:\n\n"
            f"{feedback}"
        )

    messages = [
        {"role": "system", "content": TRITON_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print("=" * 60)
        print(f"MODEL : {model}")
        print(f"GRAMMAR enforces: imports + @triton.jit")
        print("-" * 60)
        print("USER MESSAGE:")
        print(user_message)
        print("=" * 60)

    client = Fireworks(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "grammar", "grammar": TRITON_GRAMMAR},
        messages=messages,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content

    # Some models (e.g. gpt-oss) leak internal chain-of-thought tokens
    # like <|end|><|start|>assistant... into the response. Truncate there.
    if "<|" in raw:
        raw = raw[:raw.index("<|")]

    if verbose:
        print("RAW RESPONSE:")
        print(raw)
        print("=" * 60)

    return raw.strip()


# ---------------------------------------------------------------------------
# validate_static() — Layer 1, no GPU needed
# ---------------------------------------------------------------------------
def validate_static(kernel_code: str) -> dict:
    """
    AST-based static analysis for Triton-specific semantic rules.
    Runs on any machine — no GPU, no Triton installation required.

    Checks:
      - Valid Python syntax (ast.parse)
      - @triton.jit decorator is present
      - tl.program_id() is called
      - Every tl.load() has mask= keyword argument
      - Every tl.store() has mask= keyword argument
      - BLOCK_SIZE literals are powers of 2
      - tl.* calls use known function names (whitelist)
      - A launcher function (non-kernel callable) exists

    Returns:
        dict with keys: valid (bool), errors (list), warnings (list)
    """
    results = {"valid": False, "errors": [], "warnings": []}

    # --- Syntax check ---
    try:
        tree = ast.parse(kernel_code)
    except SyntaxError as e:
        results["errors"].append(f"Syntax error: {e}")
        return results

    # --- Walk AST ---
    has_triton_jit = False
    has_program_id = False
    has_launcher   = False
    loads_without_mask  = []
    stores_without_mask = []
    unknown_tl_calls    = []
    non_power_of_2      = []

    for node in ast.walk(tree):

        # Detect @triton.jit and launcher functions
        if isinstance(node, ast.FunctionDef):
            is_kernel = any(
                isinstance(d, ast.Attribute) and d.attr == "jit"
                for d in node.decorator_list
            )
            if is_kernel:
                has_triton_jit = True
            else:
                has_launcher = True

        # Inspect tl.* calls
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                attr = func.attr
                kw_names = [kw.arg for kw in node.keywords]

                if attr == "load" and "mask" not in kw_names:
                    loads_without_mask.append(f"line {node.lineno}")

                if attr == "store" and "mask" not in kw_names:
                    stores_without_mask.append(f"line {node.lineno}")

                if attr == "program_id":
                    has_program_id = True

                # Unknown tl.* call check
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "tl"
                    and attr not in KNOWN_TL_ATTRS
                ):
                    unknown_tl_calls.append(f"tl.{attr} (line {node.lineno})")

        # BLOCK_SIZE literal power-of-2 check
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and "BLOCK" in target.id.upper()
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, int)
                ):
                    n = node.value.value
                    if n > 0 and (n & (n - 1)) != 0:
                        non_power_of_2.append(
                            f"{target.id}={n} (line {node.lineno})"
                        )

    # --- Collect results ---
    if not has_triton_jit:
        results["errors"].append("No @triton.jit decorator found")
    if not has_program_id:
        results["errors"].append(
            "tl.program_id() not called — kernel has no thread ID"
        )
    if loads_without_mask:
        results["errors"].append(
            f"tl.load() missing mask= at: {', '.join(loads_without_mask)}"
        )
    if stores_without_mask:
        results["errors"].append(
            f"tl.store() missing mask= at: {', '.join(stores_without_mask)}"
        )
    if non_power_of_2:
        results["errors"].append(
            f"BLOCK size not power of 2: {', '.join(non_power_of_2)}"
        )
    if unknown_tl_calls:
        results["warnings"].append(
            f"Unknown tl.* calls (may not exist in Triton): "
            f"{', '.join(unknown_tl_calls)}"
        )
    if not has_launcher:
        results["warnings"].append(
            "No launcher function found — kernel cannot be called from Python"
        )

    results["valid"] = len(results["errors"]) == 0
    return results


# ---------------------------------------------------------------------------
# validate_gpu() — Layer 2, requires Triton + GPU
# ---------------------------------------------------------------------------
def validate_gpu(kernel_code: str) -> dict:
    """
    Compile and define the kernel module by writing to a temp .py file.

    Triton's @jit decorator reads the function's source file at compile time —
    it cannot compile a function defined via exec() in memory. Writing to a
    real .py file and importing it via importlib resolves this.

    Returns:
        dict with keys: valid (bool), errors (list)
    """
    import tempfile
    import importlib.util
    import os

    results = {"valid": False, "errors": []}

    # Write kernel to a real .py file so @triton.jit can read source
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(kernel_code)
        tmp.close()

        spec   = importlib.util.spec_from_file_location("_triton_kernel", tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        results["valid"] = True

    except ImportError as e:
        results["errors"].append(f"Import error: {e}")
    except Exception as e:
        results["errors"].append(f"Runtime error during kernel definition: {e}")
    finally:
        os.unlink(tmp.name)

    return results


# ---------------------------------------------------------------------------
# validate_correctness() — Layer 3, requires CUDA GPU
# ---------------------------------------------------------------------------
def validate_correctness(
    kernel_code: str,
    pytorch_fn,
    test_inputs: list,
) -> dict:
    """
    Compare kernel output against a PyTorch baseline using torch.allclose.

    Args:
        kernel_code:  Generated kernel source
        pytorch_fn:   Callable that computes the expected result on CPU tensors
        test_inputs:  List of CPU torch.Tensor inputs

    Returns:
        dict with keys: valid (bool), errors (list), max_diff (float | None)
    """
    import torch
    import tempfile
    import importlib.util
    import os

    results = {"valid": False, "errors": [], "max_diff": None}

    # Must write to a real .py file — same reason as validate_gpu():
    # @triton.jit reads the function's source file at compile time and
    # cannot compile a function defined via exec() in memory.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tmp.write(kernel_code)
        tmp.close()

        spec   = importlib.util.spec_from_file_location("_triton_kernel_correctness", tmp.name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        results["errors"].append(f"Exec failed: {e}")
        return results
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    launcher = _find_launcher(vars(module))
    if launcher is None:
        results["errors"].append("No launcher function found in generated code")
        return results

    try:
        cuda_inputs = [
            t.cuda() if isinstance(t, torch.Tensor) else t
            for t in test_inputs
        ]
        expected = pytorch_fn(*test_inputs)
        actual   = launcher(*cuda_inputs).cpu()

        max_diff = (expected - actual).abs().max().item()
        results["max_diff"] = max_diff

        if torch.allclose(expected, actual, rtol=1e-3, atol=1e-3):
            results["valid"] = True
        else:
            results["errors"].append(
                f"Output mismatch — max diff: {max_diff:.6f} | "
                f"expected range [{expected.min().item():.4f}, {expected.max().item():.4f}] | "
                f"got range [{actual.min().item():.4f}, {actual.max().item():.4f}]"
            )
    except Exception as e:
        results["errors"].append(f"Correctness check error: {e}")

    return results


# ---------------------------------------------------------------------------
# generate_kernel() — main entry point with self-repair retry loop
# ---------------------------------------------------------------------------
def generate_kernel(
    pytorch_code: str,
    input_shapes: Optional[dict] = None,
    pytorch_fn=None,
    test_inputs: Optional[list] = None,
    max_attempts: int = 3,
    api_key: Optional[str] = None,
    generation_model: str = "accounts/fireworks/models/kimi-k2p5",
    repair_model: str = "accounts/fireworks/models/kimi-k2p5",
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Generate a Triton kernel with a self-repair retry loop.

    Uses two models with different roles:
        generation_model — frontier model for the first attempt (hard task: design from scratch)
        repair_model     — model for subsequent repairs (easy task: fix a known error in existing code)

    Each failed attempt feeds the exact error + broken code back to the repair model,
    so it knows precisely what to fix rather than starting from a blank page.

    Validation layers run in order — each layer only runs if the previous passed:
        1. Static (no GPU) — always runs
        2. GPU compile     — runs if CUDA available
        3. Correctness     — runs if pytorch_fn + test_inputs provided + CUDA available

    Args:
        pytorch_code:       PyTorch operation to translate
        input_shapes:       e.g. {"x": [1024], "y": [1024]}
        pytorch_fn:         Optional CPU callable for correctness baseline
        test_inputs:        Optional list of CPU torch.Tensors
        max_attempts:       Maximum repair attempts after initial generation (default 3)
        api_key:            Fireworks API key
        generation_model:   Frontier model for initial generation
        repair_model:       Model for targeted repair on failure
        output_path:        If provided, write the validated kernel to this .py file on success
                            e.g. "fma.py" → then use: from fma import fused_multiply_add
        verbose:            Print attempt details

    Returns:
        dict with keys:
            success      (bool)         — all validation layers passed
            code         (str)          — best generated code (last attempt)
            attempts     (int)          — how many LLM calls were made
            history      (list[dict])   — per-attempt error details
            output_path  (str | None)   — path the kernel was written to, if output_path was set
    """
    import torch

    cuda_available = torch.cuda.is_available()
    feedback       = None
    last_code      = None
    history        = []

    for attempt in range(1, max_attempts + 1):
        # Attempt 1: frontier model (generate from scratch)
        # Attempt 2+: repair model (fix a known error in existing code)
        active_model = generation_model if attempt == 1 else repair_model

        if verbose:
            print(f"\n{'─' * 60}")
            role = "generation" if attempt == 1 else "repair"
            print(f"Attempt {attempt}/{max_attempts}  [{role}]  model: {active_model}")
            if feedback:
                print(f"Feedback to model:\n{feedback}")
            print("─" * 60)

        # ── Generate / Repair ─────────────────────────────────────────────
        code      = translate(
            pytorch_code,
            input_shapes=input_shapes,
            feedback=feedback,
            api_key=api_key,
            model=active_model,
            verbose=verbose,
        )
        last_code = code

        # ── Layer 1: Static ───────────────────────────────────────────────
        static = validate_static(code)
        if verbose:
            status = "✅ PASS" if static["valid"] else "❌ FAIL"
            print(f"[Static]      {status}")
            for e in static["errors"]:
                print(f"              ERROR   {e}")
            for w in static["warnings"]:
                print(f"              WARNING {w}")

        if not static["valid"]:
            error_str = "; ".join(static["errors"])
            feedback  = (
                f"Static analysis failed:\n{error_str}\n\n"
                f"Fix these issues in the kernel. Problematic code:\n{code}"
            )
            history.append({
                "attempt": attempt, "layer": "static", "errors": static["errors"]
            })
            continue

        # ── Layer 2: GPU compile ──────────────────────────────────────────
        if cuda_available:
            gpu = validate_gpu(code)
            if verbose:
                status = "✅ PASS" if gpu["valid"] else "❌ FAIL"
                print(f"[GPU compile] {status}")
                for e in gpu["errors"]:
                    print(f"              ERROR   {e}")

            if not gpu["valid"]:
                error_str = "; ".join(gpu["errors"])
                feedback  = (
                    f"The kernel failed to compile on the GPU:\n{error_str}\n\n"
                    f"Problematic code:\n{code}"
                )
                history.append({
                    "attempt": attempt, "layer": "gpu", "errors": gpu["errors"]
                })
                continue
        elif verbose:
            print("[GPU compile] ⏭  SKIPPED (no CUDA)")

        # ── Layer 3: Correctness ──────────────────────────────────────────
        if pytorch_fn is not None and test_inputs is not None and cuda_available:
            correctness = validate_correctness(code, pytorch_fn, test_inputs)
            if verbose:
                status = "✅ PASS" if correctness["valid"] else "❌ FAIL"
                diff   = f"  max_diff={correctness['max_diff']:.6f}" if correctness["max_diff"] is not None else ""
                print(f"[Correctness] {status}{diff}")
                for e in correctness["errors"]:
                    print(f"              ERROR   {e}")

            if not correctness["valid"]:
                error_str = "; ".join(correctness["errors"])
                feedback  = (
                    f"The kernel compiled but produced wrong numerical results:\n"
                    f"{error_str}\n\n"
                    f"Check your pointer arithmetic, offsets, and masking logic.\n\n"
                    f"Problematic code:\n{code}"
                )
                history.append({
                    "attempt": attempt,
                    "layer": "correctness",
                    "errors": correctness["errors"],
                    "max_diff": correctness["max_diff"],
                })
                continue
        elif verbose:
            print("[Correctness] ⏭  SKIPPED (no pytorch_fn / test_inputs / CUDA)")

        # ── All layers passed ─────────────────────────────────────────────
        if output_path is not None:
            with open(output_path, "w") as f:
                f.write(code)
            if verbose:
                print(f"\n💾 Kernel saved to {output_path}")

        if verbose:
            print(f"\n✅ Kernel generated successfully in {attempt} attempt(s)")

        return {
            "success":     True,
            "code":        code,
            "attempts":    attempt,
            "history":     history,
            "output_path": output_path,
        }

    # Exhausted all attempts
    if verbose:
        print(f"\n❌ Failed after {max_attempts} attempt(s)")

    return {
        "success":  False,
        "code":     last_code,
        "attempts": max_attempts,
        "history":  history,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _find_launcher(namespace: dict):
    """Return the first non-kernel, non-private callable in an exec'd namespace."""
    return next(
        (
            v for k, v in namespace.items()
            if callable(v)
            and not k.startswith("_")
            and not hasattr(v, "__triton_kernel__")
            and k not in ("torch", "triton", "tl")
            and k[0].islower()
        ),
        None,
    )
