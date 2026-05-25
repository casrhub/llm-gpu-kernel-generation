"""
benchmark/run_method.py

Run the proposed SLM + GBNF + self-repair pipeline on all benchmark operations
and save results to results_method.json.

Usage (Colab):
    %run benchmark/run_method.py

    # Or with a custom SLM:
    SLM_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct \
        python benchmark/run_method.py
"""
import sys, os, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.operations import BENCHMARK_OPS
from src.translator.pytorch_to_triton import generate_kernel

# ── Config ────────────────────────────────────────────────────────────────────
SLM_MODEL     = os.environ.get(
    "SLM_MODEL",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
)
REPAIR_MODEL  = os.environ.get("REPAIR_MODEL", SLM_MODEL)
MAX_ATTEMPTS  = int(os.environ.get("MAX_ATTEMPTS", "3"))
OUTPUT_FILE   = os.environ.get("RESULTS_FILE", "results_method.json")

print(f"{'=' * 60}")
print(f"METHOD BENCHMARK")
print(f"  SLM model    : {SLM_MODEL}")
print(f"  Repair model : {REPAIR_MODEL}")
print(f"  Max attempts : {MAX_ATTEMPTS}")
print(f"  Operations   : {len(BENCHMARK_OPS)}")
print(f"  Output       : {OUTPUT_FILE}")
print(f"{'=' * 60}\n")

results = []

for op in BENCHMARK_OPS:
    print(f"[{op['id']}]  {op['pytorch_code']}")
    t0 = time.time()

    result = generate_kernel(
        pytorch_code    = op["pytorch_code"],
        input_shapes    = op["input_shapes"],
        pytorch_fn      = op["pytorch_fn"],
        test_inputs     = op["test_inputs"],
        generation_model= SLM_MODEL,
        repair_model    = REPAIR_MODEL,
        max_attempts    = MAX_ATTEMPTS,
        verbose         = False,
    )

    elapsed = time.time() - t0
    status  = "✅" if result["success"] else "❌"
    print(f"  {status}  success={result['success']}  "
          f"attempts={result['attempts']}  "
          f"max_diff={result.get('max_diff', 'N/A')}  "
          f"time={elapsed:.1f}s")

    # Collect per-attempt layer info from history
    history_summary = [
        {"attempt": h["attempt"], "layer": h["layer"], "errors": h["errors"]}
        for h in result.get("history", [])
    ]

    results.append({
        "op_id":            op["id"],
        "category":         op["category"],
        "pytorch_code":     op["pytorch_code"],
        "success":          result["success"],
        "attempts":         result["attempts"],
        "compilation_pass": result["success"] or any(
            h["layer"] in ("correctness",) for h in result.get("history", [])
        ),
        "correctness_pass": result["success"],
        "max_diff":         result.get("max_diff"),
        "elapsed_s":        round(elapsed, 2),
        "history":          history_summary,
    })

# ── Summary ───────────────────────────────────────────────────────────────────
total         = len(results)
n_success     = sum(r["success"] for r in results)
n_compile     = sum(r["compilation_pass"] for r in results)
mean_attempts = sum(r["attempts"] for r in results) / total

print(f"\n{'=' * 60}")
print(f"SUMMARY")
print(f"  Correctness rate : {n_success}/{total}  ({100*n_success/total:.1f}%)")
print(f"  Compilation rate : {n_compile}/{total}  ({100*n_compile/total:.1f}%)")
print(f"  Mean attempts    : {mean_attempts:.2f}")
print(f"{'=' * 60}")

output = {
    "method":       "slm_pipeline",
    "slm_model":    SLM_MODEL,
    "repair_model": REPAIR_MODEL,
    "max_attempts": MAX_ATTEMPTS,
    "summary": {
        "total":             total,
        "correctness_rate":  round(n_success  / total, 4),
        "compilation_rate":  round(n_compile  / total, 4),
        "mean_attempts":     round(mean_attempts, 4),
    },
    "results": results,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
