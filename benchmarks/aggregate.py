"""Aggregate per-seed benchmark results into tables with paired bootstrap CIs.

    python -m benchmarks.aggregate runs/sycophancy_flipflop/            # one model
    python -m benchmarks.aggregate runs/qwen7b/ runs/llama8b/ --md results.md

Each input directory holds ``seed_*.json`` files written by a runner invoked with
``--seeds N`` (``benchmarks.sycophancy.run_flipflop`` or
``benchmarks.context_rot.run``). Benchmark type is detected from the payload.

Statistics
----------
The unit of replication is the seed. For every metric we report mean over seeds,
and for the *gap* (memory − full) a **paired** bootstrap over seeds: resample
seeds with replacement, recompute the mean gap, take the 2.5/97.5 percentiles.
Paired because both conditions see the same items and the same model per seed;
unpaired CIs would throw away exactly the structure that makes the design tight.

With few seeds (3–5) the CI is wide by construction. That is the honest width;
do not narrow it by pooling items across seeds as if they were independent.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def load_dir(path: Path) -> List[dict]:
    files = sorted(path.glob("seed_*.json"))
    if not files and path.suffix == ".json":
        files = [path]
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def detect_kind(payload: dict) -> str:
    if "results" in payload and "full" in payload.get("results", {}):
        return "sycophancy"
    if "trials" in payload:
        return "context_rot"
    raise ValueError("unrecognised payload; expected a runner's seed_*.json")


# ---------------------------------------------------------------------------
# per-seed metrics
# ---------------------------------------------------------------------------

def sycophancy_metrics(payload: dict) -> Dict[str, float]:
    res = payload["results"]
    out: Dict[str, float] = {}
    for cond in ("full", "memory"):
        rows = res[cond]
        n = max(1, len(rows))
        out[f"{cond}_flip_rate"] = sum(1 for r in rows if r.get("tof") is not None) / n
        out[f"{cond}_mean_nof"] = sum(r.get("nof", 0) for r in rows) / n
        confs = [r.get("confidence") or [] for r in rows]
        drift = [c[-1] - c[0] for c in confs if len(c) >= 2 and c[0] is not None and c[-1] is not None]
        if drift:
            out[f"{cond}_conf_drift"] = sum(drift) / len(drift)
    out["gap_flip_rate"] = out["memory_flip_rate"] - out["full_flip_rate"]   # negative = memory flips less
    out["gap_mean_nof"] = out["memory_mean_nof"] - out["full_mean_nof"]
    if "full_conf_drift" in out and "memory_conf_drift" in out:
        out["gap_conf_drift"] = out["memory_conf_drift"] - out["full_conf_drift"]
    return out


def context_rot_metrics(payload: dict) -> Dict[str, float]:
    trials = payload["trials"]
    out: Dict[str, float] = {}
    for cond in ("raw", "memory"):
        sub = [t for t in trials if t["condition"] == cond]
        out[f"{cond}_accuracy"] = sum(1 for t in sub if t["correct"]) / max(1, len(sub))
    out["gap_accuracy"] = out["memory_accuracy"] - out["raw_accuracy"]        # positive = memory better
    toks = payload.get("input_tokens", {})
    if toks.get("raw"):
        out["memory_token_share"] = toks.get("memory", 0) / toks["raw"]
    return out


METRICS = {"sycophancy": sycophancy_metrics, "context_rot": context_rot_metrics}

HEADLINE = {
    "sycophancy": [
        ("full_flip_rate", "Flip rate (full)"),
        ("memory_flip_rate", "Flip rate (memory)"),
        ("gap_flip_rate", "Gap memory−full"),
        ("full_conf_drift", "Conf. drift (full)"),
        ("memory_conf_drift", "Conf. drift (memory)"),
    ],
    "context_rot": [
        ("raw_accuracy", "Accuracy (raw)"),
        ("memory_accuracy", "Accuracy (memory)"),
        ("gap_accuracy", "Gap memory−raw"),
        ("memory_token_share", "Memory tokens / raw"),
    ],
}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float, float]:
    """(mean, lo, hi) of the mean of per-seed paired differences."""
    vals = list(values)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, mean, mean
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = [rng.choice(vals) for _ in vals]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean, lo, hi


def sign_flip_p_value(values: Sequence[float], n_perm: int = 10_000, seed: int = 0) -> float:
    """Exact-ish paired permutation (sign-flip) test of mean difference == 0."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return float("nan")
    obs = abs(sum(vals) / len(vals))
    rng = random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = sum(v if rng.random() < 0.5 else -v for v in vals) / len(vals)
        if abs(s) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def aggregate(payloads: List[dict]) -> dict:
    kind = detect_kind(payloads[0])
    per_seed = [METRICS[kind](p) for p in payloads]
    keys = sorted({k for m in per_seed for k in m})
    summary: Dict[str, dict] = {}
    for k in keys:
        vals = [m[k] for m in per_seed if k in m]
        entry = {"mean": sum(vals) / len(vals), "n": len(vals)}
        if len(vals) > 1:
            entry["sd"] = statistics.stdev(vals)
        if k.startswith("gap_"):
            mean, lo, hi = paired_bootstrap_ci(vals)
            entry.update({"ci95": [lo, hi], "p_sign_flip": sign_flip_p_value(vals)})
        summary[k] = entry
    model = payloads[0].get("args", {}).get("model", "?")
    return {"kind": kind, "model": model, "n_seeds": len(payloads), "metrics": summary}


def _fmt(k: str, v: float) -> str:
    if k.endswith("conf_drift"):
        return f"{v:+.3f}"
    if "token_share" in k:
        return f"{v * 100:.1f}%"
    return f"{v * 100:+.1f} pts" if k.startswith("gap_") else f"{v * 100:.1f}%"


def render_markdown(aggs: List[dict]) -> str:
    lines: List[str] = []
    for kind in ("context_rot", "sycophancy"):
        rows = [a for a in aggs if a["kind"] == kind]
        if not rows:
            continue
        cols = HEADLINE[kind]
        lines.append(f"### {kind.replace('_', ' ')} — {rows[0]['n_seeds']} seed(s) per model, paired bootstrap 95% CI on the gap\n")
        lines.append("| Model | " + " | ".join(label for _, label in cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for a in rows:
            cells = []
            for key, _ in cols:
                m = a["metrics"].get(key)
                if m is None:
                    cells.append("n/a")
                    continue
                cell = _fmt(key, m["mean"])
                if "ci95" in m and m["n"] > 1:
                    lo, hi = m["ci95"]
                    cell += f" [{_fmt(key, lo)}, {_fmt(key, hi)}]"
                    if m.get("p_sign_flip") == m.get("p_sign_flip"):  # not nan
                        cell += f" p={m['p_sign_flip']:.3f}"
                cells.append(cell)
            lines.append(f"| {a['model']} (n={a['n_seeds']}) | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Gap CIs are paired bootstraps over seeds; p is a two-sided sign-flip permutation test. "
                 "Few seeds ⇒ wide intervals by construction.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", help="directories of seed_*.json (one per model/run)")
    parser.add_argument("--md", default=None, help="write the markdown table here as well")
    parser.add_argument("--json", dest="json_out", default=None, help="write the aggregate JSON here")
    args = parser.parse_args(argv)

    aggs = []
    for d in args.dirs:
        payloads = load_dir(Path(d))
        if not payloads:
            print(f"[aggregate] no seed_*.json in {d}; skipping")
            continue
        aggs.append(aggregate(payloads))
    if not aggs:
        parser.error("nothing to aggregate")

    md = render_markdown(aggs)
    print(md)
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
        print(f"\nwrote {args.md}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(aggs, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
