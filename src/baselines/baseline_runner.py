"""
src/baselines/baseline_runner.py

Single-shot frontier model baseline — no GBNF grammar, no self-repair loop.
Uses the same 3-layer validation as the proposed method so results are
directly comparable.

Usage:
    from src.baselines.baseline_runner import run_baseline

    result = run_baseline(
        pytorch_code="out = x + y",
        input_shapes={"x": [1024], "y": [1024]},
        pytorch_fn=lambda x, y: x + y,
        test_inputs=[torch.randn(1024), torch.randn(1024)],
        model="accounts/fireworks/models/deepseek-v4-0",
        verbose=True,
    )
"""
import re
from typing import Optional


def _extract_code(raw: str) -> str:
    """
    Extract Python code from a response that may contain markdown,
    explanations, or other prose.

    Priority:
      1. ```python ... ``` fenced block
      2. ``` ... ``` fenced block
      3. Content starting from the first 'import torch' line
      4. Raw response as-is
    """
    # 1. ```python block
    m = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2. Generic ``` block
    m = re.search(r"```\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. Anything from 'import torch' onward
    m = re.search(r"(import torch.*)", raw, re.DOTALL)
    if m:
        return m.group(1).strip()

    return raw.strip()


def run_baseline(
    pytorch_code: str,
    input_shapes: Optional[dict] = None,
    pytorch_fn=None,
    test_inputs: Optional[list] = None,
    model: str = "accounts/fireworks/models/deepseek-v4-0",
    api_key: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """
    Call a frontier model with direct prompting (no GBNF, no self-repair).

    The same 3-layer validation pipeline from pytorch_to_triton.py is
    applied so results are directly comparable to the proposed method.

    Args:
        pytorch_code:  PyTorch operation to translate
        input_shapes:  e.g. {"x": [1024], "y": [1024]}
        pytorch_fn:    Optional CPU callable for correctness baseline
        test_inputs:   Optional list of CPU torch.Tensors
        model:         Fireworks model ID for the baseline
        api_key:       Fireworks API key (defaults to FIREWORKS_API_KEY env)
        output_path:   If provided, save generated code here on success
        verbose:       Print prompt and response

    Returns:
        dict with keys:
            success          (bool)
            code             (str)
            attempts         (int)   — always 1 for baselines
            compilation_pass (bool)
            correctness_pass (bool)
            max_diff         (float | None)
            errors           (list[str])
            output_path      (str | None)
    """
    import torch
    from fireworks.client import Fireworks
    from src.translator.pytorch_to_triton import (
        TRITON_SYSTEM_PROMPT,
        validate_static,
        validate_gpu,
        validate_correctness,
    )

    cuda_available = torch.cuda.is_available()

    # ── Build prompt (same content as proposed method, no grammar) ────────────
    user_message = f"Convert this PyTorch operation to a Triton kernel:\n\n{pytorch_code}"
    if input_shapes:
        shapes_str = ", ".join(f"{k}: {v}" for k, v in input_shapes.items())
        user_message += f"\n\nInput shapes: {shapes_str}"

    messages = [
        {"role": "system", "content": TRITON_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    if verbose:
        print("=" * 60)
        print(f"BASELINE MODEL : {model}")
        print(f"NO grammar — single shot, no repair")
        print("-" * 60)
        print("USER MESSAGE:")
        print(user_message)
        print("=" * 60)

    # ── Call model (no response_format / grammar) ─────────────────────────────
    client   = Fireworks(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4096,
    )
    raw  = response.choices[0].message.content
    code = _extract_code(raw)

    if verbose:
        print("RAW RESPONSE:")
        print(raw)
        print("-" * 60)
        print("EXTRACTED CODE:")
        print(code)
        print("=" * 60)

    result = {
        "success":          False,
        "code":             code,
        "attempts":         1,
        "compilation_pass": False,
        "correctness_pass": False,
        "max_diff":         None,
        "errors":           [],
        "output_path":      None,
    }

    # ── Layer 1: Static ───────────────────────────────────────────────────────
    static = validate_static(code)
    if verbose:
        status = "✅ PASS" if static["valid"] else "❌ FAIL"
        print(f"[Static]      {status}")
        for e in static["errors"]:
            print(f"              ERROR   {e}")
        for w in static["warnings"]:
            print(f"              WARNING {w}")

    if not static["valid"]:
        result["errors"] = static["errors"]
        return result

    # ── Layer 2: GPU compile ──────────────────────────────────────────────────
    if cuda_available:
        gpu = validate_gpu(code)
        if verbose:
            status = "✅ PASS" if gpu["valid"] else "❌ FAIL"
            print(f"[GPU compile] {status}")
            for e in gpu["errors"]:
                print(f"              ERROR   {e}")

        if not gpu["valid"]:
            result["errors"] = gpu["errors"]
            return result

        result["compilation_pass"] = True
    elif verbose:
        print("[GPU compile] ⏭  SKIPPED (no CUDA)")

    # ── Layer 3: Correctness ──────────────────────────────────────────────────
    if pytorch_fn is not None and test_inputs is not None and cuda_available:
        correctness = validate_correctness(code, pytorch_fn, test_inputs)
        if verbose:
            status = "✅ PASS" if correctness["valid"] else "❌ FAIL"
            diff   = f"  max_diff={correctness['max_diff']:.6f}" if correctness["max_diff"] is not None else ""
            print(f"[Correctness] {status}{diff}")
            for e in correctness["errors"]:
                print(f"              ERROR   {e}")

        result["max_diff"] = correctness["max_diff"]

        if not correctness["valid"]:
            result["errors"] = correctness["errors"]
            return result

        result["correctness_pass"] = True
    elif verbose:
        print("[Correctness] ⏭  SKIPPED (no pytorch_fn / test_inputs / CUDA)")

    # ── All layers passed ─────────────────────────────────────────────────────
    result["success"] = True

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(code)
        result["output_path"] = output_path
        if verbose:
            print(f"\n💾 Kernel saved to {output_path}")

    if verbose:
        print(f"\n✅ Baseline kernel generated successfully")

    return result
