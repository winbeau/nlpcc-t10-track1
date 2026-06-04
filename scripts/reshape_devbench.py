#!/usr/bin/env python3
"""GATE-B PHASE A2 — reshape the image-disjoint dev pool into testp1-shaped gold variants.

Background (notes/gate_b_plan.md, notes/eval_bench_design.md):
  PEM is paragraph-level; model behaviour differs wildly between testp1's "long + minority-dense"
  paragraphs and traindev's "short + minority-sparse" ones. The image-disjoint dev produced by
  `build_dataset --split-mode component_aware` kills the 32% image leak but still has the WRONG
  SHAPE (mean len 5.1, only ~34% records carry a minority). This script emits several gold VARIANTS
  spanning the power<->regime tradeoff so PHASE B's rank-corr can pick which one best predicts
  testp1 ranking. Each variant is a STRICT SUBSET of whole dev_gold records (every kept record stays
  intact, all its sentences), so PEM semantics hold and the file stays evaluate.py-valid.

Variants written to --out (default data/devbench/):
  dev_gold_raw.jsonl        copy of dev_gold (control). Max power, traindev shape (short/sparse).
  dev_gold_reshaped.jsonl   records with len >= --min-len, then keep ALL minority-carrying records +
                            the LONGEST minority-free ones until minority density ~= --minority-frac.
                            testp1 REGIME probe: right shape, low record count (pool-bounded).
  dev_gold_densematch.jsonl NO length floor; keep ALL minority-carrying + longest minority-free to
                            --minority-frac. Isolates the DENSITY effect at higher power.

NOTE on the pool ceiling (honest): the image-disjoint dev has only ~25 records that are BOTH long
(>=8) AND minority-carrying, so the strict reshaped variant is inherently ~30 records — a low-power
DIRECTIONAL probe, not a ranker of 50.0~50.3 union nuances. That ceiling is exactly why the clean
long-term fix is PHASE C (density-matched training data, never short/sparse). See the shape report.

Pure stdlib — safe to run locally (no torch / GPU / images). Selection is deterministic
(longest-first, id tie-break); --seed only perturbs the tie order for robustness checks.

CLI:
  python scripts/reshape_devbench.py \
      --dev-gold data/devbench/dev_gold.jsonl \
      --testp1   ../NLPCC-2026-Task10-Science/data/testp1-track-1.jsonl \
      --out      data/devbench \
      --report   notes/devbench_shape_report.md \
      --min-len 8 --minority-frac 0.84
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MINORITY = {
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
}


# ──────────────────────────────────────────────────────────────────────────────
# I/O + record helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def plen(rec: dict[str, Any]) -> int:
    return len(rec.get("sentence_label", []))


def minority_count(rec: dict[str, Any]) -> int:
    """Number of sentences carrying >=1 minority gold type (any-of, per evaluate.py)."""
    return sum(
        1 for s in rec.get("sentence_label", [])
        if any(t in MINORITY for t in s.get("types", []))
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reshape
# ──────────────────────────────────────────────────────────────────────────────

def select(
    pool: list[dict[str, Any]],
    minority_frac: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Keep ALL minority-carrying records in `pool` + enough LONGEST minority-free records to land
    at ~minority_frac density. Deterministic: minority-free sorted by (-len, id); --seed only jitters
    the tie order. If the pool lacks enough minority-free records, density ends up ABOVE the target
    (reported honestly by the caller)."""
    hard = [r for r in pool if minority_count(r) >= 1]
    easy = [r for r in pool if minority_count(r) == 0]
    H = len(hard)
    # density = H / (H + keep_easy)  ==  minority_frac  ->  keep_easy = H * (1 - frac) / frac
    keep_easy = round(H * (1.0 - minority_frac) / minority_frac) if minority_frac > 0 else len(easy)
    rng = random.Random(seed)
    # primary key: length desc; secondary: a seeded shuffle for tie robustness (stable id fallback)
    easy_shuffled = easy[:]
    rng.shuffle(easy_shuffled)
    easy_sorted = sorted(easy_shuffled, key=lambda r: -plen(r))
    kept_easy = easy_sorted[: min(keep_easy, len(easy))]
    out = hard + kept_easy
    out.sort(key=lambda r: r.get("id", ""))  # stable id order for reproducible files
    return out


def make_variants(
    dev: list[dict[str, Any]],
    min_len: int,
    minority_frac: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    raw = sorted(dev, key=lambda r: r.get("id", ""))
    reshaped = select([r for r in dev if plen(r) >= min_len], minority_frac, seed)
    densematch = select(dev, minority_frac, seed)
    return {"raw": raw, "reshaped": reshaped, "densematch": densematch}


# ──────────────────────────────────────────────────────────────────────────────
# Shape summary / report
# ──────────────────────────────────────────────────────────────────────────────

def _median(xs: list[int]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(xs[mid]) if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    lens = [plen(r) for r in records]
    mins = [minority_count(r) for r in records]
    n = len(records)
    n_sent = sum(lens)
    return {
        "n_rec": n,
        "n_sent": n_sent,
        "mean_len": (n_sent / n) if n else 0.0,
        "median_len": _median(lens),
        "max_len": max(lens) if lens else 0,
        "pct_ge8": (100.0 * sum(1 for x in lens if x >= 8) / n) if n else 0.0,
        "minority_density": (100.0 * sum(1 for m in mins if m >= 1) / n) if n else 0.0,
        "mean_min_per_rec": (sum(mins) / n) if n else 0.0,
        "pct_ge2_min": (100.0 * sum(1 for m in mins if m >= 2) / n) if n else 0.0,
    }


def testp1_summary(testp1: list[dict[str, Any]]) -> dict[str, Any]:
    """testp1 has NO gold labels -> only length stats are real; minority density is the ~84% anchor
    ESTIMATE (notes/submissions_log.md), not measured here."""
    lens = [plen(r) for r in testp1]
    n = len(testp1)
    n_sent = sum(lens)
    return {
        "n_rec": n,
        "n_sent": n_sent,
        "mean_len": n_sent / n,
        "median_len": _median(lens),
        "max_len": max(lens),
        "pct_ge8": 100.0 * sum(1 for x in lens if x >= 8) / n,
        "minority_density": 84.0,  # ANCHOR ESTIMATE (no gold), see submissions_log.md
        "mean_min_per_rec": float("nan"),
        "pct_ge2_min": float("nan"),
    }


# (key, header, width, decimals)
_COLS = [
    ("n_rec", "#rec", 6, 0),
    ("n_sent", "#sent", 6, 0),
    ("mean_len", "mean-len", 8, 2),
    ("median_len", "med-len", 7, 1),
    ("max_len", "max-len", 7, 0),
    ("pct_ge8", "%>=8", 6, 1),
    ("minority_density", "%minor", 7, 1),
    ("mean_min_per_rec", "min/rec", 7, 2),
    ("pct_ge2_min", "%>=2min", 7, 1),
]


def render_table(rows: list[tuple[str, dict[str, Any]]]) -> str:
    head = f"  {'variant':<12} " + " ".join(f"{h:>{w}}" for _, h, w, _ in _COLS)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for name, s in rows:
        cells = []
        for key, _h, w, dec in _COLS:
            v = s.get(key)
            cells.append(f"{'nan':>{w}}" if (isinstance(v, float) and v != v) else f"{v:>{w}.{dec}f}")
        lines.append(f"  {name:<12} " + " ".join(cells))
    return "\n".join(lines)


def trivial_baseline(gold_path: Path, data_root: Path) -> dict[str, float] | None:
    """Score the TRIVIAL all-Supported prediction against a gold variant via the OFFICIAL evaluator.
    The single most diagnostic number: a bench is recall-critical (like testp1, trivial~=19) only if
    all-Supported scores LOW. If it scores high (raw ~42, PEM ~65), the bench rewards suppression and
    is useless for selecting testp1 candidates. Returns {score,macro_f1,pem} *100, or None on failure."""
    ev = data_root / "offline_eval" / "evaluate.py"
    if not ev.exists():
        return None
    gold = load_jsonl(gold_path)
    pred = [{"id": r["id"], "labels": ["Supported"] * len(r["sentence_label"])} for r in gold]
    pf = Path(tempfile.mktemp(suffix=".jsonl"))
    of = Path(tempfile.mktemp(suffix=".json"))
    try:
        write_jsonl(pred, pf)
        r = subprocess.run(
            [sys.executable, str(ev), "--track", "1", "--gold", str(gold_path),
             "--pred", str(pf), "--output", str(of), "--match", "id"],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or not of.exists():
            return None
        res = json.loads(of.read_text())
        return {k: res[k] * 100 for k in ("score", "macro_f1", "pem")}
    finally:
        for p in (pf, of):
            if p.exists():
                p.unlink()


def len_histogram(records: list[dict[str, Any]], bins: list[tuple[int, int]]) -> dict[str, float]:
    lens = [plen(r) for r in records]
    n = len(lens) or 1
    out = {}
    for lo, hi in bins:
        c = sum(1 for x in lens if lo <= x <= hi)
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        out[label] = 100.0 * c / n
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reshape image-disjoint dev into testp1-shaped gold variants.")
    ap.add_argument("--dev-gold", default="data/devbench/dev_gold.jsonl")
    ap.add_argument("--testp1", default="../NLPCC-2026-Task10-Science/data/testp1-track-1.jsonl")
    ap.add_argument("--out", default="data/devbench")
    ap.add_argument("--report", default="notes/devbench_shape_report.md")
    ap.add_argument("--min-len", type=int, default=8, help="reshaped: keep records with >= this many sentences (testp1 median 8).")
    ap.add_argument("--minority-frac", type=float, default=0.84, help="target fraction of records carrying >=1 minority (testp1 anchor estimate).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="../NLPCC-2026-Task10-Science", help="for --validate: locate offline_eval/evaluate.py.")
    ap.add_argument("--no-validate", action="store_true", help="skip the official-evaluator trivial-baseline check (default: run it).")
    args = ap.parse_args(argv)

    dev = load_jsonl(Path(args.dev_gold))
    testp1 = load_jsonl(Path(args.testp1))
    out_dir = Path(args.out)

    variants = make_variants(dev, args.min_len, args.minority_frac, args.seed)

    # ---- summaries ----
    rows = [("testp1*", testp1_summary(testp1))]
    rows += [(name, summarize(recs)) for name, recs in variants.items()]
    table = render_table(rows)

    bins = [(2, 5), (6, 7), (8, 9), (10, 13), (14, 999)]
    hist_rows = [("testp1", len_histogram(testp1, bins))]
    hist_rows += [(name, len_histogram(recs, bins)) for name, recs in variants.items()]
    bin_labels = [f"{lo}-{hi}" if hi < 999 else f"{lo}+" for lo, hi in bins]
    hist_head = "  " + f"{'variant':<12} " + " ".join(f"{b:>7}" for b in bin_labels)
    hist_lines = [hist_head, "  " + "-" * (len(hist_head) - 2)]
    for name, h in hist_rows:
        hist_lines.append(f"  {name:<12} " + " ".join(f"{h[b]:>6.1f}" for b in bin_labels))
    hist_table = "\n".join(hist_lines)

    print("\n### Shape summary (testp1* minority/min-per-rec are the 84% ANCHOR ESTIMATE; testp1 has no gold)")
    print(table)
    print("\n### Paragraph-length distribution (% of records in each sentence-count bin)")
    print(hist_table)
    for name, recs in variants.items():
        write_jsonl(recs, out_dir / f"dev_gold_{name}.jsonl")
        print(f"\n  wrote {out_dir}/dev_gold_{name}.jsonl  ({len(recs)} records)")

    # ---- trivial all-Supported baseline per variant (the recall-vs-suppression litmus) ----
    trivial_table = ""
    if not args.no_validate:
        data_root = Path(args.data_root)
        tlines = [
            f"  {'variant':<12} {'score':>7} {'MF1':>7} {'PEM':>7}",
            "  " + "-" * 36,
            f"  {'testp1 (s05)':<12} {18.85:>7.2f} {22.01:>7.2f} {15.70:>7.2f}   <- target regime",
        ]
        for name, recs in variants.items():
            tb = trivial_baseline(out_dir / f"dev_gold_{name}.jsonl", data_root)
            if tb is None:
                tlines.append(f"  {name:<12}  (evaluator unavailable)")
            else:
                flag = "  <- suppression trap" if tb["score"] > 30 else "  ok: recall-critical"
                tlines.append(f"  {name:<12} {tb['score']:>7.2f} {tb['macro_f1']:>7.2f} {tb['pem']:>7.2f}{flag}")
        trivial_table = "\n".join(tlines)
        print("\n### Trivial all-Supported baseline (official evaluator) — recall-vs-suppression litmus")
        print(trivial_table)

    # ---- markdown report (committed; bench data itself gitignored) ----
    rep = Path(args.report)
    lines = [
        "# devbench shape report (GATE-B PHASE A)",
        "",
        f"> Generated by `scripts/reshape_devbench.py` (min-len={args.min_len}, minority-frac={args.minority_frac}, seed={args.seed}).",
        "> Bench data files live under `data/devbench/` and are gitignored; this report is the committed artifact.",
        "",
        "## Shape summary",
        "",
        "`testp1*` length stats are real; its `%minor` is the **84% anchor estimate** (testp1 has no gold).",
        "",
        "```",
        table,
        "```",
        "",
        "## Paragraph-length distribution (% records per sentence-count bin)",
        "",
        "```",
        hist_table,
        "```",
        "",
        "## Trivial all-Supported baseline (recall-vs-suppression litmus)",
        "",
        "The single most diagnostic number. testp1 is recall-critical: its trivial all-Supported score is",
        "~18.85 (s05). A bench is a valid testp1 proxy only if all-Supported scores LOW there too. `raw`",
        "scores all-Supported at ~42 (PEM ~65) — it would **reward** the minority-suppression that is",
        "falsified on testp1. `reshaped`/`densematch` land at ~17-18, matching testp1's regime.",
        "",
        "```",
        trivial_table or "  (--no-validate: skipped)",
        "```",
        "",
        "## How to read this / which variant to use",
        "",
        "- **raw** — image-disjoint dev untouched. Max statistical power (~500 rec), but traindev shape",
        "  (short, minority-sparse) — the regime where the model already looks fine. Control.",
        "- **reshaped** — testp1 REGIME probe (long + dense). Best shape-match but pool-bounded to ~30",
        "  records (only ~25 dev records are both long and minority-carrying). Low power: a directional",
        "  smell test, **cannot** rank 50.0~50.3 union nuances.",
        "- **densematch** — density-matched without a hard length floor. Higher power than reshaped,",
        "  isolates the minority-density effect; length still short of testp1.",
        "",
        "PHASE B picks the trustworthy variant by Spearman(dev_score, testp1_score) over the 24 scored",
        "anchors (`scripts/rankcorr_meta.py`). The pool ceiling is precisely why PHASE C (density-matched",
        "TRAINING data) is the clean long-term fix — a dev built from naturally long+dense paragraphs.",
        "",
    ]
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote shape report -> {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
