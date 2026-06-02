#!/usr/bin/env python3
"""Lone-minority paragraph veto — the #1 precision lever (zero-GPU, zero-retrain).

The score is PEM-dominated & precision-critical: a single spurious minority sentence in an
otherwise all-Supported paragraph flips that paragraph's PEM 1->0. Analysis of the grid (47.80)
predictions found 207 of 229 minority-firing paragraphs are LONE-MINORITY (exactly one minority
sentence, all others Supported). Reverting that lone sentence to Supported recaptures the whole
paragraph's PEM at the cost of at most one minority recall — net-positive whenever the FP:TP
ratio among lone-minority predictions exceeds ~1 (very likely in this OOD precision-critical
regime). Confirmed only on Codabench (no testp1 gold).

Reads a sentence-level raw jsonl (id, sent_index, label, min_logprob) and writes a submission
jsonl ({id, labels}, sentence order preserved).

Modes:
  all   : veto EVERY lone-minority sentence (unconditional). The core hypothesis test.
  gated : veto a lone-minority sentence ONLY if its min_logprob < --gate (default -0.05) — i.e.
          KEEP very-confident lone predictions, revert only the shaky ones (conservative).

Usage:
  python scripts/lone_minority_veto.py --raw outputs/testp1_gridckpt_lp_raw.jsonl \
      --out submissions/testp1_veto_all.jsonl --mode all
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


def load_records(path: str) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["id"]].append(r)
    for rid in by_id:
        by_id[rid].sort(key=lambda r: int(r["sent_index"]))
    return by_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lone-minority paragraph veto -> Supported.")
    ap.add_argument("--raw", required=True, help="sentence-level raw jsonl (id, sent_index, label, min_logprob)")
    ap.add_argument("--out", required=True, help="output submission jsonl")
    ap.add_argument("--mode", choices=["all", "gated"], default="all")
    ap.add_argument("--gate", type=float, default=-0.05,
                    help="gated mode: veto a lone-minority sent only if min_logprob < gate (revert shaky ones)")
    args = ap.parse_args(argv)

    by_id = load_records(args.raw)

    # diagnostics over the whole file
    n_lone = n_multi = n_clean = 0
    subs = []
    n_vetoed = 0
    for rid, rows in by_id.items():
        labels = [r["label"] for r in rows]
        mino_idx = [i for i, l in enumerate(labels) if l in MINORITY]
        if not mino_idx:
            n_clean += 1
        elif len(mino_idx) == 1:
            n_lone += 1
            i = mino_idx[0]
            do = args.mode == "all"
            if args.mode == "gated":
                ml = rows[i].get("min_logprob")
                do = isinstance(ml, (int, float)) and ml < args.gate
            if do:
                labels[i] = "Supported"
                n_vetoed += 1
        else:
            n_multi += 1
        subs.append({"id": rid, "labels": labels})

    dist = collections.Counter(l for s in subs for l in s["labels"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in subs:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"records={len(subs)} | clean(all-Supported)={n_clean} | lone-minority={n_lone} | multi-minority={n_multi}")
    print(f"mode={args.mode}" + (f" gate={args.gate}" if args.mode == "gated" else "")
          + f" -> vetoed {n_vetoed} lone-minority paragraphs to all-Supported")
    print("label distribution:", {k: dist.get(k, 0) for k in
          ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
           "Scope Overgeneralization", "Contradiction"]})
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
