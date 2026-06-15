#!/usr/bin/env python3
"""Per-class union of member predictions (precision-aware).

Each member = a {id, labels:[...]} jsonl (aggregated per-sentence labels). The union, per
sentence, counts how many members vote each minority class; a class is ADOPTED only if its
vote count >= the (per-class) min-votes threshold. Among adopted minority classes pick the
most-voted (tie -> rarer class, better for macro-F1); else Supported.

This generalizes scripts/ensemble_union.py:
  --min-votes-default 1                      == naive add-only union (any member fires -> adopt)
  --min-votes-default 2                      == consensus (>=2 members)
  --per-class "Scope Overgeneralization:2"   == per-class threshold (liberal on high-precision
                                                classes, consensus on FP-prone ones)

Output id/order follows --ref (the testp1 jsonl), so sentence counts/ids align for the scorer.
Pure stdlib; runs anywhere.
"""
from __future__ import annotations
import argparse
import json

LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]
MINORITY = [l for l in LABELS if l != "Supported"]
# tie-break priority when multiple minority classes pass with equal votes: rarer first.
RARITY = {"Contradiction": 0, "Unsupported Causal Mechanistic": 1,
          "Unsupported Entity": 2, "Scope Overgeneralization": 3}


def load_member(path: str) -> dict[str, list[str]]:
    return {json.loads(l)["id"]: json.loads(l)["labels"]
            for l in open(path, encoding="utf-8") if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-class min-votes union of member predictions.")
    ap.add_argument("--members", nargs="+", required=True, help="member {id,labels} jsonls")
    ap.add_argument("--ref", required=True, help="reference jsonl (testp1) for id + sentence count/order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-votes-default", type=int, default=1)
    ap.add_argument("--per-class", default="",
                    help="'Label:K,Label:K' per-class min-votes overrides (label may contain spaces)")
    a = ap.parse_args()

    mv: dict[str, int] = {}
    for item in a.per_class.split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.rpartition(":")
        mv[k.strip()] = int(v)

    def minv(c: str) -> int:
        return mv.get(c, a.min_votes_default)

    members = [load_member(p) for p in a.members]
    ref = [json.loads(l) for l in open(a.ref, encoding="utf-8") if l.strip()]

    n_fire = {c: 0 for c in MINORITY}
    out = []
    for rec in ref:
        rid = rec["id"]
        n = len(rec.get("sentence_label", []))
        labels = []
        for i in range(n):
            votes = {c: 0 for c in MINORITY}
            for m in members:
                ml = m.get(rid, [])
                if i < len(ml) and ml[i] in votes:
                    votes[ml[i]] += 1
            passing = [c for c in MINORITY if votes[c] > 0 and votes[c] >= minv(c)]
            if passing:
                best = max(passing, key=lambda c: (votes[c], -RARITY[c]))
                labels.append(best)
                n_fire[best] += 1
            else:
                labels.append("Supported")
        out.append({"id": rid, "labels": labels})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tot = sum(n_fire.values())
    print(f"union -> {a.out}: {len(out)} records | min_votes default={a.min_votes_default} "
          f"per-class={mv} | minority fired={tot} {n_fire}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
