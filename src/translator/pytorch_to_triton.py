"""
Translates PyTorch operations to optimized Triton GPU kernels using OpenAI.
"""
import re
import ast
from openai import OpenAI
from typing import Optional

TRITON_SYSTEM_PROMPT = """You are an expert GPU kernel engineer specializing in Triton, OpenAI's GPU programming language.
Your task is to convert PyTorch operations into correct, efficient Triton kernels.

## Triton Kernel Structure

Every Triton kernel follows this pattern:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_name(
    # Pointers to tensors
    x_ptr,
    y_ptr,
    output_ptr,
    # Tensor dimensions
    n_elements,
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Compute program ID (which block this instance handles)
    pid = tl.program_id(axis=0)

    # 2. Compute element offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 3. Create mask to guard out-of-bounds memory access
    mask = offsets < n_elements

    # 4. Load input data with masking
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. Compute
    output = x + y

    # 6. Store result with masking
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Critical Rules

1. **Always use `tl.constexpr`** for compile-time constants like BLOCK_SIZE, BLOCK_M, BLOCK_N, BLOCK_K
2. **Always mask memory accesses** — use `mask = offsets < n_elements` before tl.load/tl.store
3. **BLOCK_SIZE must be a power of 2**: 16, 32, 64, 128, 256, 512, 1024
4. **`tl.program_id(axis=0)`** gives the block index along dimension 0 (use axis=1, axis=2 for higher dims)
5. **`tl.arange(0, BLOCK_SIZE)`** creates a range of offsets [0, 1, ..., BLOCK_SIZE-1]
6. **Pointer arithmetic**: for a 2D tensor of shape (M, N) stored row-major, element [i, j] is at `ptr + i * N + j`
7. **No Python control flow inside kernels** — use tl.where() for conditional logic
8. **tl.load/tl.store** accept `other=` keyword for fill value when mask is False (default 0.0)

## Launcher Function Pattern

Every kernel needs a Python launcher that:
1. Validates inputs are CUDA tensors
2. Computes the grid dimensions
3. Calls the kernel

```python
def kernel_name_launcher(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda, "Inputs must be CUDA tensors"
    assert x.shape == y.shape, "Input shapes must match"
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    kernel_name[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
```

## Grid Computation

- **1D elementwise**: `grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)`
- **2D tiled (matmul)**: `grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))`
- `triton.cdiv(a, b)` is ceiling division: `(a + b - 1) // b`

## Elementwise Template

For operations like `out = f(x, y)` on flat tensors:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * y  # Replace with actual operation
    tl.store(output_ptr + offsets, output, mask=mask)

def elementwise_launcher(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    elementwise_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
```

## Reduction Template

For operations like `out = sum(x)` across a dimension:

```python
@triton.jit
def reduction_kernel(
    x_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(output_ptr + pid, result)
```

## Multi-Dimensional Indexing

For a 2D tensor of shape (M, N):
```python
row_idx = tl.program_id(axis=0)   # which row
col_offsets = tl.arange(0, BLOCK_SIZE)
mask = col_offsets < N
ptr = x_ptr + row_idx * N + col_offsets
x = tl.load(ptr, mask=mask)
```

## Common tl Functions

- `tl.load(ptr, mask, other=0.0)` — load from global memory
- `tl.store(ptr, value, mask)` — store to global memory
- `tl.sum(x, axis=0)` — reduce-sum along axis
- `tl.max(x, axis=0)` — reduce-max along axis
- `tl.exp(x)` — elementwise exp
- `tl.log(x)` — elementwise log
- `tl.sqrt(x)` — elementwise sqrt
- `tl.where(condition, x, y)` — elementwise conditional
- `tl.dot(a, b)` — matrix multiply within a block (for matmul kernels)
- `tl.zeros([M, N], dtype=tl.float32)` — allocate zeroed accumulator
- `tl.cast(x, tl.float32)` — type casting

## Output Format

Respond with ONLY valid Python code. No markdown, no explanation, no text before or after.
The code must be a complete, runnable Python module containing:
1. All necessary imports (torch, triton, triton.language as tl)
2. The @triton.jit kernel function(s)
3. The Python launcher function(s)

The launcher function name should match the operation (e.g., elementwise_add, softmax, matmul).
"""


def translate(
    pytorch_code: str,
    input_shapes: Optional[dict] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> str:
    """
    Translate PyTorch operation to a Triton kernel.

    Args:
        pytorch_code: PyTorch operation code or description to translate
        input_shapes: Optional dict mapping argument names to shapes, e.g. {"x": [1024], "y": [1024]}
        api_key: Optional API key; defaults to OPENAI_API_KEY env var
        model: OpenAI model to use (default: gpt-4o-mini)
        verbose: Print the prompt sent and raw response received

    Returns:
        Generated Triton kernel code (imports + kernel + launcher)
    """
    client = OpenAI(api_key=api_key)

    user_message = f"Convert this PyTorch operation to a Triton kernel:\n\n{pytorch_code}"
    if input_shapes:
        shapes_str = ", ".join(f"{k}: {v}" for k, v in input_shapes.items())
        user_message += f"\n\nInput shapes: {shapes_str}"

    messages = [
        {"role": "system", "content": TRITON_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if verbose:
        print("=" * 60)
        print(f"MODEL: {model}")
        print("=" * 60)
        print("SYSTEM PROMPT:")
        print(TRITON_SYSTEM_PROMPT)
        print("-" * 60)
        print("USER MESSAGE:")
        print(user_message)
        print("=" * 60)

    stream = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=messages,
        stream=True,
    )

    chunks = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)

    raw_response = "".join(chunks)

    if verbose:
        print("RAW RESPONSE:")
        print(raw_response)
        print("=" * 60)

    return _extract_code(raw_response)


def validate(
    kernel_code: str,
    pytorch_fn=None,
    test_inputs: Optional[list] = None,
) -> dict:
    """
    Validate generated Triton kernel code.

    Args:
        kernel_code: The generated Python/Triton source code string
        pytorch_fn: Optional callable — PyTorch baseline to compare against
        test_inputs: Optional list of torch.Tensor inputs for correctness check

    Returns:
        dict with keys:
            syntax_valid (bool): AST parse succeeded
            imports_ok (bool): exec() without ImportError/NameError
            correctness (bool|None): output matches pytorch_fn, or None if not checked
            errors (list[str]): collected error messages
    """
    results = {
        "syntax_valid": False,
        "imports_ok": False,
        "correctness": None,
        "errors": [],
    }

    try:
        ast.parse(kernel_code) # We can verify syntax with pythons ast module, since triton code is valid Python
        results["syntax_valid"] = True
    except SyntaxError as e:
        results["errors"].append(f"Syntax error: {e}")
        return results

    namespace = {}
    try:
        exec(kernel_code, namespace)
        results["imports_ok"] = True
    except ImportError as e:
        results["errors"].append(f"Import error: {e}")
    except Exception as e:
        results["errors"].append(f"Execution error: {e}")

    if pytorch_fn is not None and test_inputs is not None and results["imports_ok"]:
        results["correctness"] = _check_correctness(
            namespace, pytorch_fn, test_inputs, results["errors"]
        )

    return results


def _check_correctness(
    namespace: dict,
    pytorch_fn,
    test_inputs: list,
    errors: list,
) -> bool:
    """Run kernel against PyTorch baseline and compare outputs with torch.allclose."""
    import torch

    launcher = _find_launcher(namespace)
    if launcher is None:
        errors.append("No launcher function found in generated code")
        return False

    try:
        cuda_inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in test_inputs]
        expected = pytorch_fn(*test_inputs)
        actual = launcher(*cuda_inputs)

        if isinstance(actual, torch.Tensor):
            actual_cpu = actual.cpu()
        else:
            actual_cpu = actual

        if not torch.allclose(expected, actual_cpu, rtol=1e-3, atol=1e-3):
            max_diff = (expected - actual_cpu).abs().max().item()
            errors.append(f"Correctness check failed — max diff: {max_diff:.6f}")
            return False

        return True

    except Exception as e:
        errors.append(f"Correctness check error: {e}")
        return False


def _find_launcher(namespace: dict):
    """Find the launcher function in the exec'd namespace (non-kernel, non-private callable)."""
    candidates = [
        v for k, v in namespace.items()
        if callable(v)
        and not k.startswith("_")
        and not hasattr(v, "__triton_kernel__")
        and k not in ("torch", "triton", "tl")
        and not k[0].isupper()
    ]
    return candidates[0] if candidates else None


def _extract_code(text: str) -> str:
    """Extract Python code from LLM response, stripping markdown code fences if present."""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
