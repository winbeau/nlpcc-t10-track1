#!/usr/bin/env python3
"""Recall-additive UNION ensemble of per-sentence predictions across submissions.

testp1 is recall-critical (catch the gold minority sentence to win a paragraph's PEM). This unions
several models' submissions: a sentence is labelled minority if ANY input predicts a minority class;
among disagreeing minority labels at the same position, pick the most-voted (ties broken by a fixed
rarity priority Contradiction>SO>UCM>UE — rarer class wins, since those are the under-fired ones).
A sentence stays Supported only if ALL inputs say Supported.

Inputs: submission jsonls ({"id":..,"labels":[...]}), same id set + per-id length. Output: same format.
NOTE: union RAISES recall (and minority count). On recall-critical testp1 that can help, but extra
minorities at disagreements may be FPs — confirm on Codabench.

Usage:
  python scripts/ensemble_union.py --out submissions/testp1_ensU.jsonl \
      submissions/testp1_b5_l0.5_submission.jsonl \
      submissions/testp1_os4_b5_l0.5_submission.jsonl \
      submissions/testp1_perclass_b5_l0.5_submission.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json

MINORITY = ["Unsupported Causal Mechanistic", "Unsupported Entity",
            "Scope Overgeneralization", "Contradiction"]
# rarity priority for tie-break (rarer/under-fired first)
PRIORITY = {"Contradiction": 0, "Scope Overgeneralization": 1,
            "Unsupported Causal Mechanistic": 2, "Unsupported Entity": 3}


def load(path: str) -> dict[str, list[str]]:
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["id"]] = r["labels"]
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Union-ensemble per-sentence submissions (recall-additive).")
    ap.add_argument("--out", required=True)
    ap.add_argument("subs", nargs="+", help="2+ submission jsonls to union")
    args = ap.parse_args(argv)

    models = [load(p) for p in args.subs]
    ids = list(models[0].keys())
    for m in models[1:]:
        assert set(m.keys()) == set(ids), "id sets differ across submissions"

    out = []
    dist = collections.Counter()
    n_union_added = 0
    for rid in ids:
        label_lists = [m[rid] for m in models]
        n = len(label_lists[0])
        assert all(len(ll) == n for ll in label_lists), f"length mismatch at {rid}"
        merged = []
        for i in range(n):
            votes = [ll[i] for ll in label_lists]
            mino_votes = [v for v in votes if v in PRIORITY]
            if not mino_votes:
                merged.append("Supported")
            else:
                # most-voted minority; tie -> rarer class (lower PRIORITY value)
                cnt = collections.Counter(mino_votes)
                top = max(cnt.items(), key=lambda kv: (kv[1], -PRIORITY[kv[0]]))[0]
                merged.append(top)
                if not all(v in PRIORITY for v in votes):
                    n_union_added += 1  # at least one model said Supported -> union added a minority
            dist[merged[i]] += 1
        out.append({"id": rid, "labels": merged})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mino = sum(dist[m] for m in MINORITY)
    print(f"union of {len(models)} models -> {args.out}")
    print(f"  records={len(out)} minority={mino} (Supported {dist['Supported']}); "
          f"union-added minorities (>=1 model disagreed) = {n_union_added}")
    print("  per-class:", {m: dist[m] for m in MINORITY})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
