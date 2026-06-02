"""
benchmark/run_campaign.py

Extended benchmark campaign runner.

What it adds over run_method.py / run_baseline.py:
1) Scales operations to multiple vector sizes (e.g., 1024, 2048, 4096, 8192)
2) Repeats each operation with different random seeds
3) Runs multiple model tracks in one execution
4) Saves campaign metadata + per-op results as JSON

Typical usage (Colab):
    python benchmark/run_campaign.py \
      --tracks method,baseline \
      --method-model accounts/fireworks/models/gpt-oss-20b \
      --baseline-model accounts/fireworks/models/deepseek-v4-0 \
      --sizes 1024,2048,4096 \
      --repeats 2 \
      --optimize
"""

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.operations import BENCHMARK_OPS
from src.baselines.baseline_runner import run_baseline
from src.optimizer.autotune import optimize_kernel
from src.translator.pytorch_to_triton import generate_kernel


def _parse_csv_list(raw: str, cast_fn=str):
    return [cast_fn(x.strip()) for x in raw.split(",") if x.strip()]


def _resize_shape(shape, size):
    # This benchmark uses 1D vectors. For safety, only replace first dim.
    if not shape:
        return shape
    out = list(shape)
    out[0] = size
    return out


def _build_extended_ops(base_ops, sizes, repeats, base_seed, categories=None, limit_base_ops=0):
    selected = []
    allowed = set(categories) if categories else None

    for op in base_ops:
        if allowed and op["category"] not in allowed:
            continue
        selected.append(op)

    if limit_base_ops and limit_base_ops > 0:
        selected = selected[:limit_base_ops]

    extended = []
    for op_idx, op in enumerate(selected):
        for size in sizes:
            resized_shapes = {
                name: _resize_shape(shape, size)
                for name, shape in op["input_shapes"].items()
            }
            for rep in range(1, repeats + 1):
                seed = base_seed + op_idx * 1000 + size * 10 + rep
                torch.manual_seed(seed)
                test_inputs = []
                for shape in resized_shapes.values():
                    test_inputs.append(torch.randn(*shape))

                new_op = deepcopy(op)
                new_op["id"] = f"{op['id']}_n{size}_r{rep}"
                new_op["input_shapes"] = resized_shapes
                new_op["test_inputs"] = test_inputs
                new_op["campaign_size"] = size
                new_op["campaign_rep"] = rep
                new_op["campaign_seed"] = seed
                new_op["source_op_id"] = op["id"]
                extended.append(new_op)

    return extended


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "-")


def _run_track(
    track,
    model,
    ops,
    api_key,
    optimize,
    max_attempts,
    repair_model,
):
    print("\n" + "=" * 72)
    print(f"TRACK: {track.upper()} | model={model}")
    print(f"Operations: {len(ops)} | optimize={optimize}")
    print("=" * 72)

    rows = []

    for i, op in enumerate(ops, start=1):
        print(f"[{i:03d}/{len(ops)}] {op['id']} :: {op['pytorch_code']}")
        t0 = time.time()

        if track == "method":
            out = generate_kernel(
                pytorch_code=op["pytorch_code"],
                input_shapes=op["input_shapes"],
                pytorch_fn=op["pytorch_fn"],
                test_inputs=op["test_inputs"],
                max_attempts=max_attempts,
                api_key=api_key,
                generation_model=model,
                repair_model=repair_model or model,
                verbose=False,
            )
            success = bool(out.get("success"))
            attempts = int(out.get("attempts", max_attempts))
            code = out.get("code")
            history = out.get("history", [])
            compilation_pass = success or any(h.get("layer") == "correctness" for h in history)
            correctness_pass = success
            errors = []
            if not success and history:
                errors = history[-1].get("errors", [])
        else:
            out = run_baseline(
                pytorch_code=op["pytorch_code"],
                input_shapes=op["input_shapes"],
                pytorch_fn=op["pytorch_fn"],
                test_inputs=op["test_inputs"],
                model=model,
                api_key=api_key,
                verbose=False,
            )
            success = bool(out.get("success"))
            attempts = 1
            code = out.get("code")
            compilation_pass = bool(out.get("compilation_pass"))
            correctness_pass = bool(out.get("correctness_pass"))
            errors = out.get("errors", [])

        elapsed = round(time.time() - t0, 2)

        ms_kernel = None
        ms_pytorch = None
        speedup = None
        best_block_size = None
        best_num_warps = None

        if optimize and success and code:
            try:
                opt = optimize_kernel(
                    kernel_code=code,
                    test_inputs=op["test_inputs"],
                    pytorch_fn=op["pytorch_fn"],
                    category=op["category"],
                    verbose=False,
                )
                if opt.get("success"):
                    ms_kernel = round(opt["ms_kernel"], 4)
                    ms_pytorch = round(opt["ms_pytorch"], 4)
                    speedup = round(opt["speedup"], 4)
                    best_block_size = opt.get("best_block_size")
                    best_num_warps = opt.get("best_num_warps")
            except Exception as e:
                errors = list(errors) + [f"optimize error: {e}"]

        tag = "PASS" if success else "FAIL"
        speed_txt = f" speedup={speedup:.2f}x" if speedup is not None else ""
        print(
            f"  {tag} attempts={attempts} compile={compilation_pass} "
            f"correct={correctness_pass} t={elapsed:.1f}s{speed_txt}"
        )

        rows.append({
            "op_id": op["id"],
            "source_op_id": op["source_op_id"],
            "category": op["category"],
            "size": op["campaign_size"],
            "repeat": op["campaign_rep"],
            "seed": op["campaign_seed"],
            "pytorch_code": op["pytorch_code"],
            "success": success,
            "attempts": attempts,
            "compilation_pass": compilation_pass,
            "correctness_pass": correctness_pass,
            "elapsed_s": elapsed,
            "ms_kernel": ms_kernel,
            "ms_pytorch": ms_pytorch,
            "speedup": speedup,
            "best_block_size": best_block_size,
            "best_num_warps": best_num_warps,
            "errors": errors,
        })

    total = len(rows)
    n_success = sum(r["success"] for r in rows)
    n_compile = sum(r["compilation_pass"] for r in rows)
    mean_attempts = sum(r["attempts"] for r in rows) / total if total else 0.0
    speedups = [r["speedup"] for r in rows if r["speedup"] is not None]
    mean_speedup = sum(speedups) / len(speedups) if speedups else None

    summary = {
        "total": total,
        "correctness_rate": round(n_success / total, 4) if total else 0.0,
        "compilation_rate": round(n_compile / total, 4) if total else 0.0,
        "mean_attempts": round(mean_attempts, 4),
        "mean_speedup": round(mean_speedup, 4) if mean_speedup is not None else None,
        "n_speedup_measured": len(speedups),
    }

    print("\n" + "-" * 72)
    print(
        f"Summary {track}/{_slug(model)}: "
        f"correct={n_success}/{total} ({100*summary['correctness_rate']:.1f}%), "
        f"compile={n_compile}/{total} ({100*summary['compilation_rate']:.1f}%), "
        f"attempts={summary['mean_attempts']:.2f}"
    )
    if mean_speedup is not None:
        print(f"Mean speedup: {mean_speedup:.2f}x over {len(speedups)} optimized ops")
    print("-" * 72)

    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Extended benchmark campaign runner")
    parser.add_argument("--tracks", default="method,baseline", help="Comma list: method,baseline")
    parser.add_argument("--method-model", default="accounts/fireworks/models/gpt-oss-20b")
    parser.add_argument("--repair-model", default="", help="Optional; defaults to method model")
    parser.add_argument("--baseline-model", default="accounts/fireworks/models/deepseek-v4-0")
    parser.add_argument("--sizes", default="1024,2048,4096")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--categories", default="", help="Optional comma list")
    parser.add_argument("--limit-base-ops", type=int, default=0, help="0 means all")
    parser.add_argument("--optimize", action="store_true", help="Run autotune timing")
    parser.add_argument("--output-dir", default="benchmark/results_campaign")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")

    args = parser.parse_args()

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY not set")

    tracks = _parse_csv_list(args.tracks)
    sizes = _parse_csv_list(args.sizes, int)
    categories = _parse_csv_list(args.categories) if args.categories else None

    if not sizes or any(s <= 0 for s in sizes):
        raise ValueError("--sizes must contain positive integers")
    if args.repeats <= 0:
        raise ValueError("--repeats must be > 0")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be > 0")

    ops = _build_extended_ops(
        base_ops=BENCHMARK_OPS,
        sizes=sizes,
        repeats=args.repeats,
        base_seed=args.seed,
        categories=categories,
        limit_base_ops=args.limit_base_ops,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_meta = {
        "timestamp": stamp,
        "tracks": tracks,
        "method_model": args.method_model,
        "repair_model": args.repair_model or args.method_model,
        "baseline_model": args.baseline_model,
        "sizes": sizes,
        "repeats": args.repeats,
        "seed": args.seed,
        "max_attempts": args.max_attempts,
        "categories": categories,
        "limit_base_ops": args.limit_base_ops,
        "optimize": args.optimize,
        "n_base_ops": len(BENCHMARK_OPS),
        "n_extended_ops": len(ops),
    }

    print("=" * 72)
    print("EXTENDED CAMPAIGN CONFIG")
    print(json.dumps(campaign_meta, indent=2))
    print("=" * 72)

    if args.dry_run:
        print("Dry run mode enabled. No model calls were executed.")
        return

    for track in tracks:
        track = track.lower()
        if track not in ("method", "baseline"):
            raise ValueError(f"Unknown track: {track}")

        model = args.method_model if track == "method" else args.baseline_model
        rows, summary = _run_track(
            track=track,
            model=model,
            ops=ops,
            api_key=api_key,
            optimize=args.optimize,
            max_attempts=args.max_attempts,
            repair_model=(args.repair_model or args.method_model),
        )

        payload = {
            "campaign": campaign_meta,
            "track": track,
            "model": model,
            "summary": summary,
            "results": rows,
        }

        out_file = out_dir / f"results_campaign_{track}_{_slug(model)}_{stamp}.json"
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
