#!/usr/bin/env python3
"""Post-hoc PEM-recovery: suppress LOW-CONFIDENCE minority predictions -> Supported.

Rationale (notes/analysis.md + testp1 results): the score is PEM-dominated and the model
over-fires minority labels on OOD testp1 -> false positives shatter clean paragraphs. A purely
post-hoc, training-free lever: take an existing model's sentence-level raw predictions (which now
carry `min_logprob` = the weakest token's logprob over the whole {"label":"X"} response ~= label
confidence, length-invariant; see infer.py), and revert any MINORITY prediction whose confidence
is below a threshold τ back to "Supported" (the safe, PEM-protecting default).

Sweeps τ over chosen percentiles of the minority preds' min_logprob (we cannot tune τ on the
HIDDEN testp1 gold, so generate a few variants + compare on Codabench). For each τ it writes a
submission jsonl ({id, labels}, sentence order preserved) + prints suppression count and label
distribution. Higher τ (toward 0) = suppress MORE minority = more conservative.

Usage:
  python scripts/threshold_variants.py --raw outputs/testp1_gridckpt_lp_raw.jsonl \
      --out-prefix submissions/testp1_gridckpt_thr [--taus -0.05,-0.2,-0.5]
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

MINORITY = {
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
}
ALL_LABELS = ["Supported", *sorted(MINORITY)]


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_submission(rows: list[dict], tau: float | None) -> tuple[list[dict], int]:
    """Group sentence rows -> [{id, labels}] (order by sent_index). With tau set, revert any
    minority prediction whose min_logprob < tau to 'Supported'. Rows with null min_logprob are
    left untouched (we never fabricate confidence)."""
    by_id: dict[str, dict[int, str]] = collections.defaultdict(dict)
    n_suppressed = 0
    for r in rows:
        lab = r["label"]
        if tau is not None and lab in MINORITY:
            ml = r.get("min_logprob")
            if isinstance(ml, (int, float)) and ml < tau:
                lab = "Supported"
                n_suppressed += 1
        by_id[r["id"]][int(r["sent_index"])] = lab
    subs = [{"id": rid, "labels": [d[i] for i in sorted(d)]} for rid, d in by_id.items()]
    return subs, n_suppressed


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Threshold-suppress low-confidence minority -> Supported.")
    ap.add_argument("--raw", required=True, help="sentence-level raw jsonl (must carry min_logprob)")
    ap.add_argument("--out-prefix", required=True, help="output submission jsonl path prefix")
    ap.add_argument("--taus", default="", help="comma floats (min_logprob cutoffs); empty=auto percentiles")
    args = ap.parse_args(argv)

    rows = load_rows(args.raw)
    n_lp = sum(1 for r in rows if isinstance(r.get("min_logprob"), (int, float)))
    print(f"rows: {len(rows)} | with min_logprob: {n_lp}")
    if n_lp == 0:
        print("[ABORT] no min_logprob in raw — re-run infer.py (fixed) to populate confidence.")
        return 1

    mls = sorted(r["min_logprob"] for r in rows
                 if r["label"] in MINORITY and isinstance(r.get("min_logprob"), (int, float)))
    print(f"minority predictions (with logprob): {len(mls)}")
    if mls:
        qs = {p: _pct(mls, p) for p in (10, 25, 50, 75, 90)}
        print("  min_logprob percentiles of minority preds:",
              {p: round(v, 3) for p, v in qs.items()})

    if args.taus.strip():
        taus = [float(x) for x in args.taus.split(",")]
    else:  # auto: suppress the bottom 25/50/75% least-confident minority preds
        taus = [qs[p] for p in (25, 50, 75)] if mls else []

    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    for tau in [None, *taus]:
        subs, n_sup = to_submission(rows, tau)
        dist = collections.Counter(l for s in subs for l in s["labels"])
        mino_left = sum(dist[m] for m in MINORITY)
        tag = "base" if tau is None else f"tau{tau:+.2f}".replace(".", "p")
        outp = f"{args.out_prefix}_{tag}.jsonl"
        with open(outp, "w", encoding="utf-8") as f:
            for s in subs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[{tag}] suppressed {n_sup} minority->Supported | minority left {mino_left} "
              f"(S={dist['Supported']}) | -> {outp}")
    print("\nNext: zip a chosen variant with scripts/make_submission_zip.py and submit to Codabench.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
