#!/usr/bin/env python3
"""GATE-B PHASE B5 — rank-correlate offline dev scores against known testp1 scores.

WRITE-AHEAD (notes/gate_b_plan.md B5): this is ready to run but NEEDS PHASE B's dev predictions,
which require the remote GPU (re-run the reproducible adapters on dev_sft, then synthesize the union
anchors locally). It does NOT run as part of PHASE A.

What it answers: "Is a given gold variant a trustworthy testp1 selector?" The domain shift (different
papers) makes ABSOLUTE dev scores meaningless, so we only trust ORDER: a bench is credible iff the
order it imposes on the 24 known-testp1 anchors matches their testp1 order (Spearman). The Mine-phase
finding (eval_bench_design.md) is that Spearman(MF1,score)=+0.980 and Spearman(PEM,score)=+0.945 on
testp1 — so a bench that merely reproduces the MF1+PEM ordering nearly pins the score ordering.

Inputs
  --gold       a gold variant from reshape_devbench.py (dev_gold_{raw,reshaped,densematch}.jsonl).
  --pred-dir   dir of per-anchor dev predictions named "<anchor>.jsonl" (aggregate.py {id,labels}).
               anchor stem must match a key in ANCHORS below (e.g. s01.jsonl, s15.jsonl).
  --data-root  to locate offline_eval/evaluate.py.

For each anchor present in BOTH --pred-dir and ANCHORS(scored): subset its dev pred to the gold id set
(evaluate.py requires identical id sets), score via the OFFICIAL evaluator, then Spearman the dev
(score / MF1 / PEM) vector against the embedded testp1 vector. Also reports the union-knee test (does
the bench rank s15 at the top of the union family?) and applies the plan's B6 decision gate.

  python scripts/rankcorr_meta.py --gold data/devbench/dev_gold_reshaped.jsonl \
      --pred-dir outputs/devbench_preds --data-root ../NLPCC-2026-Task10-Science
  python scripts/rankcorr_meta.py --selftest      # verify the hand-written Spearman
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ── Embedded testp1 anchor scores (notes/submissions_log.md). None = 待测/unscored (excluded). ──
# name: (type, testp1_score, testp1_macro_f1, testp1_pem)   — all on the 0-100 scale.
ANCHORS: dict[str, tuple[str, float | None, float | None, float | None]] = {
    "s01": ("m", 47.79897, 52.76518, 42.83276),
    "s02": ("m", 44.46574, 48.99973, 39.93174),
    "s03": ("pp", 43.85382, 48.28784, 39.41980),
    "s04": ("pp", 38.47759, 42.48420, 34.47099),
    "s05": ("pp", 18.85468, 22.00970, 15.69966),
    "s06": ("pp", 39.57432, 44.33635, 34.81229),
    "s07": ("m", 40.90984, 46.83674, 34.98294),
    "s08": ("m", 45.10490, 50.44872, 39.76109),
    "s09": ("m", 43.83218, 49.95104, 37.71331),
    "s10": ("m", 41.69745, 46.53484, 36.86007),
    "s11": ("u", 48.21778, 54.79733, 41.63823),
    "s12": ("m", 43.33171, 48.26752, 38.39590),
    "s13": ("m", 44.43184, 49.44389, 39.41980),
    "s14": ("u", 49.98915, 56.80424, 43.17406),
    "s15": ("u", 50.25660, 57.16849, 43.34471),  # ★ testp1 champion
    "s16": ("u", 47.28613, 52.59274, 41.97952),
    "s17": ("m", None, None, None),
    "s18": ("u", 49.71340, 57.27662, 42.15017),
    "s19": ("u", 48.48465, 56.35496, 40.61433),
    "s20": ("u", 49.35701, 56.73450, 41.97952),
    "s21": ("u", 47.77223, 55.95401, 39.59044),
    "s22": ("u", None, None, None),
    "s23": ("u", 49.82598, 57.67243, 41.97952),
    "s24": ("m", None, None, None),
    "s25": ("m", 40.04057, 45.26884, 34.81229),
    "s26": ("m", 44.69206, 51.84146, 37.54266),
    "s27": ("m", 29.27701, 33.98063, 24.57338),
    "s28": ("u", None, None, None),
    "s29": ("u", None, None, None),
}
# NOTE: the union-knee test in main() derives its family from the rows that actually have dev preds
# (`uni = [r for r in rows if r[1]=="u"]`), so there is no module-level union list to keep in sync.


# ──────────────────────────────────────────────────────────────────────────────
# Hand-written Spearman (scipy is not installed in the env)
# ──────────────────────────────────────────────────────────────────────────────

def _avg_ranks(xs: list[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean of the spanned ranks."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mean of positions i..j (0-based) -> +1 for 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den = (sum(x * x for x in da) * sum(y * y for y in db)) ** 0.5
    return num / den if den else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho = Pearson on average ranks. Handles ties."""
    if len(x) != len(y):
        raise ValueError("spearman: length mismatch")
    return _pearson(_avg_ranks(x), _avg_ranks(y))


def _selftest() -> int:
    cases = [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0),
        ([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], 0.8),   # d^2 sum=4 -> 1-6*4/(5*24)=0.8
        ([1, 1, 2, 3], [1, 2, 2, 3], None),          # tie smoke (just must run)
    ]
    ok = True
    for x, y, exp in cases:
        got = spearman(x, y)
        if exp is None:
            print(f"  spearman({x},{y}) = {got:+.4f}  (tie smoke)")
            continue
        good = abs(got - exp) < 1e-9
        ok = ok and good
        print(f"  spearman({x},{y}) = {got:+.4f}  expected {exp:+.4f}  {'OK' if good else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ──────────────────────────────────────────────────────────────────────────────
# Official-evaluator scoring of one anchor on one gold variant
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _mktemp(suffix: str) -> str:
    """Atomically create a temp file (mkstemp, no TOCTOU) and return its path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def score_on_gold(
    gold_path: Path,
    pred_path: Path,
    data_root: Path,
) -> dict[str, float] | None:
    """Subset the dev prediction to the gold variant's id set, then score via the official evaluator.
    Returns {score,macro_f1,pem} *100, or None if the pred does not cover every gold id."""
    gold = load_jsonl(gold_path)
    gold_ids = [r["id"] for r in gold]
    pred_list = load_jsonl(pred_path)
    pred_by_id = {r["id"]: r for r in pred_list}
    if len(pred_by_id) != len(pred_list):
        print(f"    !! {pred_path.name}: duplicate ids collapsed "
              f"({len(pred_list)} rows -> {len(pred_by_id)} ids); kept last each", file=sys.stderr)
    missing = [i for i in gold_ids if i not in pred_by_id]
    if missing:
        print(f"    !! {pred_path.name}: missing {len(missing)}/{len(gold_ids)} gold ids "
              f"(e.g. {missing[:3]}) -> skipped", file=sys.stderr)
        return None
    no_labels = [i for i in gold_ids if "labels" not in pred_by_id[i]]
    if no_labels:
        print(f"    !! {pred_path.name}: {len(no_labels)} records lack a 'labels' key "
              f"(raw infer.py output, not aggregate.py?) -> skipped", file=sys.stderr)
        return None
    pred_sub = [{"id": i, "labels": pred_by_id[i]["labels"]} for i in gold_ids]
    ev = data_root / "offline_eval" / "evaluate.py"
    pf = Path(_mktemp(".jsonl"))
    of = Path(_mktemp(".json"))
    try:
        with pf.open("w", encoding="utf-8") as f:
            for r in pred_sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        res = subprocess.run(
            [sys.executable, str(ev), "--track", "1", "--gold", str(gold_path),
             "--pred", str(pf), "--output", str(of), "--match", "id"],
            capture_output=True, text=True,
        )
        if res.returncode != 0 or not of.exists():
            print(f"    !! evaluator failed for {pred_path.name}: {res.stderr.strip()[:200]}", file=sys.stderr)
            return None
        d = json.loads(of.read_text())
        return {k: d[k] * 100 for k in ("score", "macro_f1", "pem")}
    finally:
        for p in (pf, of):
            if p.exists():
                p.unlink()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rank-correlate offline dev scores vs known testp1 scores.")
    ap.add_argument("--gold", help="gold variant jsonl (reshape_devbench.py output).")
    ap.add_argument("--pred-dir", help="dir of per-anchor dev preds named <anchor>.jsonl ({id,labels}).")
    ap.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    ap.add_argument("--min-anchors", type=int, default=5, help="minimum scored anchors for a meaningful rho.")
    ap.add_argument("--selftest", action="store_true", help="verify the hand-written Spearman and exit.")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.gold or not args.pred_dir:
        ap.error("--gold and --pred-dir are required (or use --selftest)")

    gold_path = Path(args.gold)
    pred_dir = Path(args.pred_dir)
    data_root = Path(args.data_root)

    # collect anchors that have BOTH a dev pred file and a scored testp1 entry
    found = []
    for pf in sorted(pred_dir.glob("*.jsonl")):
        name = pf.stem
        meta = ANCHORS.get(name)
        if meta is None:
            print(f"  (skip {pf.name}: not a known anchor)", file=sys.stderr)
            continue
        if meta[1] is None:
            print(f"  (skip {name}: testp1 unscored / 待测)", file=sys.stderr)
            continue
        found.append((name, pf, meta))

    if len(found) < args.min_anchors:
        print(f"error: only {len(found)} usable anchors (< --min-anchors {args.min_anchors}). "
              f"Need more reproduced dev preds; see gate_b_plan.md PHASE B.", file=sys.stderr)
        return 2

    print(f"\nScoring {len(found)} anchors on {gold_path.name} (n_rec={len(load_jsonl(gold_path))}) ...")
    rows = []  # (name, type, dev_score, dev_mf1, dev_pem, tp_score, tp_mf1, tp_pem)
    for name, pf, meta in found:
        dev = score_on_gold(gold_path, pf, data_root)
        if dev is None:
            continue
        _, ts, tm, tp = meta
        rows.append((name, meta[0], dev["score"], dev["macro_f1"], dev["pem"], ts, tm, tp))

    if len(rows) < args.min_anchors:
        print(f"error: only {len(rows)} anchors scored successfully (< {args.min_anchors}).", file=sys.stderr)
        return 2

    # ---- per-anchor table (sorted by testp1 score desc) ----
    rows.sort(key=lambda r: -r[5])
    print(f"\n  {'anchor':<8} {'T':<3} {'dev_sc':>7} {'dev_MF1':>8} {'dev_PEM':>8} | "
          f"{'tp_sc':>7} {'tp_MF1':>7} {'tp_PEM':>7}")
    print("  " + "-" * 74)
    for n, t, ds, dm, dp, ts, tm, tp in rows:
        print(f"  {n:<8} {t:<3} {ds:>7.2f} {dm:>8.2f} {dp:>8.2f} | {ts:>7.2f} {tm:>7.2f} {tp:>7.2f}")

    dev_sc = [r[2] for r in rows]; dev_mf1 = [r[3] for r in rows]; dev_pem = [r[4] for r in rows]
    tp_sc = [r[5] for r in rows]
    rho_sc = spearman(dev_sc, tp_sc)
    rho_mf1 = spearman(dev_mf1, tp_sc)
    rho_pem = spearman(dev_pem, tp_sc)
    print(f"\n  Spearman vs testp1 SCORE (n={len(rows)}):")
    print(f"    dev_score -> {rho_sc:+.3f}")
    print(f"    dev_MF1   -> {rho_mf1:+.3f}")
    print(f"    dev_PEM   -> {rho_pem:+.3f}")

    # ---- union-knee test: does the bench rank s15 at the top of the union family it has? ----
    uni = [r for r in rows if r[1] == "u"]
    if len(uni) >= 2 and any(r[0] == "s15" for r in uni):
        by_dev = sorted(uni, key=lambda r: -r[2])
        s15_rank = next(i for i, r in enumerate(by_dev, 1) if r[0] == "s15")
        print(f"\n  Union-knee: dev ranks s15 at #{s15_rank}/{len(uni)} of the union family "
              f"(testp1 truth: s15 is #1). "
              + ("OK" if s15_rank == 1 else "MISSES the knee"))
        print("    (eval_bench_design: the bench likely CANNOT resolve the 50.0~50.3 union top within "
              "its CI — treat union-knee as low-confidence regardless.)")
    else:
        print("\n  Union-knee: not enough union anchors with s15 present to test.")

    # ---- B6 decision gate ----
    if math.isnan(rho_sc):
        print("\n  DECISION: rho=nan — zero-variance dev scores (degenerate gold variant or all anchors "
              "tie on it). Cannot rank; inspect the per-anchor table above. NOT a true anticorrelation.")
        return 2
    print("\n  DECISION (plan B6, on dev_score rho):")
    if rho_sc >= 0.8:
        print(f"    rho={rho_sc:+.3f} >= 0.80 -> TRUSTWORTHY selection bench (but still each 1 Codabench "
              "shot for union-top ties).")
    elif rho_sc >= 0.6:
        print(f"    rho={rho_sc:+.3f} in [0.60,0.80) -> COARSE filter only: kill known bad directions, "
              "do not pick union winners.")
    elif rho_sc >= 0.5:
        print(f"    rho={rho_sc:+.3f} in [0.50,0.60) -> WEAK; use only as a smell test.")
    else:
        print(f"    rho={rho_sc:+.3f} < 0.50 -> ABANDON this variant; switch to the cumulative protocol "
              "(PHASE C).")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
