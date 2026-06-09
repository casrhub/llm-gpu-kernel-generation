# llm-gpu-kernel-generation

Automated generation of optimized GPU kernels in Triton using small language models with constrained grammar and multi-layer validation.

## The Problem

Modern LLMs can generate code in dozens of languages, but ask them to write GPU kernels and they struggle:
- Generated kernels have syntax errors or invalid Triton constructs
- Code compiles but has memory bugs or race conditions  
- Code is correct but painfully slow

You need a system that guarantees correctness *and* performance—not just one or the other.

## The Approach

Instead of using massive models (670B+), we use a **small LLM** (~20B) with intelligent guardrails:

1. **Constrained Generation (GBNF)**: Grammar rules prevent the model from generating invalid Triton structure at token-level
2. **3-Layer Validation**: Static checks → GPU compilation → numerical correctness
3. **Self-Repair Loop**: When validation fails, feed the exact error back to the model so it can fix itself

The hypothesis: **a small model with the right infrastructure outperforms a large model with none**.

## Quick Start

### Installation

```bash
git clone https://github.com/casrhub/llm-gpu-kernel-generation.git
cd llm-gpu-kernel-generation
pip install -r requirements.txt
export FIREWORKS_API_KEY=your_api_key_here
```

### Run a Campaign

Generate and benchmark 25 GPU kernels across different sizes:

```bash
python benchmark/run_campaign.py \
  --tracks method,baseline \
  --method-model accounts/fireworks/models/gpt-oss-20b \
  --baseline-model accounts/fireworks/models/deepseek-v4-0 \
  --sizes 1024,2048,4096 \
  --repeats 2 \
  --optimize
```

For a quick validation (no API calls):
```bash
python benchmark/run_campaign.py \
  --tracks method \
  --sizes 1024 \
  --repeats 1 \
  --dry-run
```

Results are saved as JSON in `benchmark/results_campaign/` with a manifest summarizing all runs.

## How It Works

```mermaid
graph TD
  %% Inputs
  I1["pytorch_code (string)"]
  I2["input_shapes (dict)"]
  I3["pytorch_fn (CPU ref)"]
  I4["test_inputs (tensors)"]
  I5["max_attempts (int)"]

  %% Main generation and validation path
  GK["generate_kernel()"]
  M["gpt-oss-20b"]
  T["translate()<br/>SLM + GBNF grammar"]
  L1["Layer 1 - Static (AST)<br/>Syntax + Triton rules"]
  L2["Layer 2 - GPU compile<br/>tempfile + importlib"]
  L3["Layer 3 - Correctness<br/>pytorch_fn vs kernel output"]
  VK["Validated kernel (string)"]
  FB["FAIL -> feedback<br/>error msg + broken code"]
  GPU["Requires Google Colab GPU T4"]

  %% Optimization path
  OK["optimize_kernel()<br/>hardware-aware autotuning"]
  INJ["_inject_autotune()<br/>Remove hardcoded BLOCK_SIZE<br/>add @triton.autotune"]
  CFG["Config space<br/>BLOCK_SIZE: 256 / 512 / 1024<br/>num_warps: 4 or 8"]
  BM["Benchmark all configs<br/>picks fastest on target GPU"]
  OUT["OUTPUT<br/>best_block_size<br/>best_num_warps + speedup vs PyTorch"]

  I1 --> T
  I2 --> T
  I3 --> L3
  I4 --> L3
  I5 --> GK

  GK --> T
  M -.-> T

  T --> L1
  L1 -->|PASS| L2
  L2 -->|PASS| L3
  L3 -->|PASS| VK

  L1 -->|FAIL| FB
  L2 -->|FAIL| FB
  L3 -->|FAIL| FB
  FB -. self-repair loop .-> T

  GPU --> L2
  GPU --> L3

  VK --> OK
  OK --> INJ
  INJ --> CFG
  CFG --> BM
  BM --> OUT

  style GK fill:#dff0d8,stroke:#b2c8a3
  style M fill:#ecebe5,stroke:#c8c6be
  style T fill:#e9e6fb,stroke:#c0b8ea
  style L1 fill:#f7ecd8,stroke:#d8bf8f
  style L2 fill:#f7ecd8,stroke:#d8bf8f
  style L3 fill:#f7ecd8,stroke:#d8bf8f
  style VK fill:#dff0d8,stroke:#b2c8a3
  style FB fill:#f8e2e2,stroke:#d7a6a6
  style OK fill:#e1effc,stroke:#9ebddd
  style INJ fill:#e1effc,stroke:#9ebddd
  style CFG fill:#e1effc,stroke:#9ebddd
  style BM fill:#e1effc,stroke:#9ebddd
  style OUT fill:#eef6df,stroke:#b7c895
```

Each validation layer is stricter than the last. The model gets up to 3 repair attempts before an operation is marked as failed.

## Project Structure

```
src/
  translator/
    pytorch_to_triton.py     # Main pipeline: generation + validation + repair
  baselines/
    baseline_runner.py       # Baseline: direct prompting, no guardrails

benchmark/
  operations.py              # 25 GPU operations (elementwise, reduction, compound)
  run_campaign.py            # Extended benchmark runner: multi-size/seed/model
  run_method.py              # Evaluate proposed method
  run_baseline.py            # Evaluate baseline approach
  results_campaign/          # Output JSONs and manifests

report/                       # LaTeX + analysis scripts
test_*.py                     # Unit tests
```

## Benchmarking

The campaign runner evaluates:
- **25 operations** across 3 categories (elementwise, reduction, compound)
- **Multiple vector sizes** (1024 to 8192 elements)
- **Multiple models** (method: gpt-oss-20b, baseline: deepseek-v4-0)
- **Auto-tuning** for `BLOCK_SIZE` optimization per operation

Output metrics per operation:
- Success rate (generation + validation passed)
- Compilation errors, validation errors
- Speedup vs PyTorch (with optimized `BLOCK_SIZE`)
- Number of repair attempts needed

Sample command for targeting specific categories:
```bash
python benchmark/run_campaign.py \
  --categories reduction \
  --sizes 1024,2048,4096 \
  --repeats 3 \
  --optimize \
  --output-tag reduction_deep
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Triton (NVIDIA GPUs only)
- Fireworks AI API key (for LLM calls)

Tested on: NVIDIA T4 (Colab), A100

## Key Files

- **[src/translator/pytorch_to_triton.py](src/translator/pytorch_to_triton.py)**: Core generation + validation logic
- **[benchmark/run_campaign.py](benchmark/run_campaign.py)**: Campaign runner with multi-size/seed/model support
- **[benchmark/operations.py](benchmark/operations.py)**: Benchmark operations and PyTorch reference implementations

