"""
benchmark/compare.py

Load results JSON files and produce:
  - Descriptive statistics table
  - Per-category breakdown
  - McNemar test (binary: compilation / correctness)
  - Wilcoxon signed-rank test (attempts)
  - Bonferroni-corrected p-values for 4 planned comparisons

Usage:
    python benchmark/compare.py \
        --method  results_method.json \
        --b1      results_baseline_deepseek-v4-0.json \
        --b2      results_baseline_kimi-k2p5.json

    # Or in Colab:
    %run benchmark/compare.py
    # (auto-discovers results_*.json in current directory)
"""
import sys, json, glob, argparse
from pathlib import Path

# ── Scipy / numpy — optional but needed for stats ────────────────────────────
try:
    import numpy as np
    from scipy.stats import wilcoxon
    from statsmodels.stats.contingency_tables import mcnemar
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    print("⚠  scipy / statsmodels not installed — descriptive stats only.")
    print("   pip install scipy statsmodels\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def label(data: dict) -> str:
    if data.get("method") == "slm_pipeline":
        slug = data.get("slm_model", "").split("/")[-1]
        return f"SLM+GBNF+repair ({slug})"
    slug = data.get("model_slug", data.get("model", "baseline").split("/")[-1])
    return f"Baseline ({slug})"


def results_by_id(data: dict) -> dict:
    return {r["op_id"]: r for r in data["results"]}


def _pct(n, total):
    return f"{n}/{total} ({100*n/total:.1f}%)"


# ── Descriptive table ─────────────────────────────────────────────────────────

def print_descriptive(datasets: list[dict]):
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    headers = ["Method", "N", "Compile", "Correct", "Mean Att.", "Std Att."]
    col_w   = [30, 4, 18, 18, 10, 10]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print("-" * 70)

    for d in datasets:
        rs      = d["results"]
        total   = len(rs)
        n_comp  = sum(r["compilation_pass"] for r in rs)
        n_corr  = sum(r["correctness_pass"] for r in rs)
        attempts= [r["attempts"] for r in rs]
        mean_a  = sum(attempts) / total
        std_a   = (sum((a - mean_a)**2 for a in attempts) / total) ** 0.5

        row = [
            label(d)[:30],
            str(total),
            _pct(n_comp, total),
            _pct(n_corr, total),
            f"{mean_a:.2f}",
            f"{std_a:.2f}",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(row, col_w)))

    print()


# ── Per-category breakdown ────────────────────────────────────────────────────

def print_category_breakdown(datasets: list[dict]):
    print("=" * 70)
    print("CORRECTNESS BY CATEGORY")
    print("=" * 70)

    categories = ["elementwise", "reduction", "compound"]

    for d in datasets:
        print(f"\n  {label(d)}")
        rs = d["results"]
        for cat in categories:
            cat_rs  = [r for r in rs if r["category"] == cat]
            n_corr  = sum(r["correctness_pass"] for r in cat_rs)
            total   = len(cat_rs)
            bar_len = int(20 * n_corr / total) if total else 0
            bar     = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {cat:<12} [{bar}] {_pct(n_corr, total)}")
    print()


# ── Statistical tests ─────────────────────────────────────────────────────────

def run_stats(method: dict, baselines: list[dict]):
    if not STATS_AVAILABLE:
        return

    print("=" * 70)
    print("INFERENTIAL STATISTICS  (α = 0.05, Bonferroni n=4)")
    print("=" * 70)

    # 4 planned comparisons → Bonferroni threshold
    n_comparisons  = 2 * len(baselines)       # compile + correct per baseline
    alpha_adjusted = 0.05 / n_comparisons

    print(f"  Planned comparisons : {n_comparisons}")
    print(f"  α adjusted          : {alpha_adjusted:.4f}\n")

    m_by_id = results_by_id(method)

    header = f"  {'Comparison':<38} {'Metric':<12} {'Test':<10} {'p-value':<10} {'Sig?'}"
    print(header)
    print("  " + "-" * 68)

    for b in baselines:
        b_by_id = results_by_id(b)
        ops     = sorted(set(m_by_id) & set(b_by_id))   # common operations

        # ── McNemar: compilation ─────────────────────────────────────────────
        # Contingency table for paired binary outcomes
        # [[both_pass, only_method_pass],
        #  [only_baseline_pass, both_fail]]
        b00 = sum( m_by_id[o]["compilation_pass"] and     b_by_id[o]["compilation_pass"]  for o in ops)
        b01 = sum( m_by_id[o]["compilation_pass"] and not b_by_id[o]["compilation_pass"]  for o in ops)
        b10 = sum(not m_by_id[o]["compilation_pass"] and  b_by_id[o]["compilation_pass"]  for o in ops)
        b11 = sum(not m_by_id[o]["compilation_pass"] and not b_by_id[o]["compilation_pass"] for o in ops)

        table_compile = np.array([[b00, b01], [b10, b11]])
        try:
            res_compile = mcnemar(table_compile, exact=True)
            p_compile   = res_compile.pvalue
        except Exception:
            p_compile   = float("nan")

        # ── McNemar: correctness ─────────────────────────────────────────────
        c00 = sum( m_by_id[o]["correctness_pass"] and     b_by_id[o]["correctness_pass"]  for o in ops)
        c01 = sum( m_by_id[o]["correctness_pass"] and not b_by_id[o]["correctness_pass"]  for o in ops)
        c10 = sum(not m_by_id[o]["correctness_pass"] and  b_by_id[o]["correctness_pass"]  for o in ops)
        c11 = sum(not m_by_id[o]["correctness_pass"] and not b_by_id[o]["correctness_pass"] for o in ops)

        table_correct = np.array([[c00, c01], [c10, c11]])
        try:
            res_correct = mcnemar(table_correct, exact=True)
            p_correct   = res_correct.pvalue
        except Exception:
            p_correct   = float("nan")

        # ── Wilcoxon: attempts ───────────────────────────────────────────────
        m_att = [m_by_id[o]["attempts"] for o in ops]
        b_att = [b_by_id[o]["attempts"] for o in ops]
        diffs = [m - b for m, b in zip(m_att, b_att)]

        if any(d != 0 for d in diffs):
            try:
                _, p_att = wilcoxon(m_att, b_att)
            except Exception:
                p_att = float("nan")
        else:
            p_att = 1.0   # identical — no difference

        b_label = label(b)[:35]

        def sig_marker(p):
            if p != p:    return "N/A"
            return "✅ YES" if p < alpha_adjusted else "NO"

        print(f"  {'Method vs ' + b_label:<38} {'compile':<12} {'McNemar':<10} {p_compile:<10.4f} {sig_marker(p_compile)}")
        print(f"  {'Method vs ' + b_label:<38} {'correct':<12} {'McNemar':<10} {p_correct:<10.4f} {sig_marker(p_correct)}")
        print(f"  {'Method vs ' + b_label:<38} {'attempts':<12} {'Wilcoxon':<10} {p_att:<10.4f} {sig_marker(p_att)}")
        print()

    print(f"  Note: McNemar used for paired binary outcomes (same benchmark ops).")
    print(f"        Wilcoxon used for ordinal attempts (non-parametric, paired).")
    print(f"        Bonferroni correction applied across {n_comparisons} planned comparisons.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("--method", default=None,
                        help="Path to results_method.json")
    parser.add_argument("--b1",    default=None,
                        help="Path to first baseline results JSON")
    parser.add_argument("--b2",    default=None,
                        help="Path to second baseline results JSON (optional)")
    args = parser.parse_args()

    # Auto-discover if not provided
    if args.method is None:
        found = glob.glob("results_method.json")
        if not found:
            print("No results_method.json found. Run benchmark/run_method.py first.")
            sys.exit(1)
        args.method = found[0]

    if args.b1 is None:
        found = sorted(glob.glob("results_baseline_*.json"))
        if not found:
            print("No results_baseline_*.json found. Run benchmark/run_baseline.py first.")
            sys.exit(1)
        args.b1 = found[0]
        args.b2 = found[1] if len(found) > 1 else None

    method    = load(args.method)
    baselines = [load(args.b1)]
    if args.b2:
        baselines.append(load(args.b2))

    all_datasets = [method] + baselines

    print(f"\nLoaded:")
    print(f"  Method    : {args.method}  →  {label(method)}")
    for i, b in enumerate(baselines, 1):
        path = args.b1 if i == 1 else args.b2
        print(f"  Baseline {i}: {path}  →  {label(b)}")

    print_descriptive(all_datasets)
    print_category_breakdown(all_datasets)
    run_stats(method, baselines)


if __name__ == "__main__":
    main()
