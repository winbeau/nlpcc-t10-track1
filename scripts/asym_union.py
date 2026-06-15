#!/usr/bin/env python3
"""Asymmetric union: a fixed BASE submission + gated additions from a member fleet.

Unlike scripts/per_class_union.py (symmetric voting over equal members), this treats one
submission (--base, e.g. s15's proven 50.26 union) as the authority: wherever the base fires
a minority class it is kept VERBATIM. Only where the base says "Supported" do we consider
ADDING a minority, and only if >= K fleet members agree on it (per-class K via --per-class).

This guarantees the output is a strict superset of the base's firing positions (base firings
are never dropped or overridden), so it can only ADD recall on top of the base — the union
lever applied conservatively around a strong base.

Output id/order follows --ref. Pure stdlib.
"""
from __future__ import annotations
import argparse
import json

LABELS = ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
          "Scope Overgeneralization", "Contradiction"]
MINORITY = [l for l in LABELS if l != "Supported"]
RARITY = {"Contradiction": 0, "Unsupported Causal Mechanistic": 1,
          "Unsupported Entity": 2, "Scope Overgeneralization": 3}


def ld(p: str) -> dict[str, list[str]]:
    return {json.loads(l)["id"]: json.loads(l)["labels"]
            for l in open(p, encoding="utf-8") if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="authority submission {id,labels}; its firings are kept verbatim")
    ap.add_argument("--members", nargs="+", required=True, help="fleet member {id,labels} jsonls")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-votes-default", type=int, default=2, help="K fleet members must agree to ADD")
    ap.add_argument("--per-class", default="", help="'Label:K,...' per-class add-threshold overrides")
    a = ap.parse_args()

    mv: dict[str, int] = {}
    for item in a.per_class.split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.rpartition(":")
        mv[k.strip()] = int(v)
    minv = lambda c: mv.get(c, a.min_votes_default)

    base = ld(a.base)
    members = [ld(p) for p in a.members]
    ref = [json.loads(l) for l in open(a.ref, encoding="utf-8") if l.strip()]

    kept = {c: 0 for c in MINORITY}   # base minority firings preserved
    added = {c: 0 for c in MINORITY}  # net-new firings added where base=Supported
    out = []
    for rec in ref:
        rid = rec["id"]
        n = len(rec.get("sentence_label", []))
        bl = base.get(rid, [])
        labels = []
        for i in range(n):
            blab = bl[i] if i < len(bl) else "Supported"
            if blab in MINORITY:
                labels.append(blab)            # keep base verbatim
                kept[blab] += 1
                continue
            votes = {c: 0 for c in MINORITY}
            for m in members:
                ml = m.get(rid, [])
                if i < len(ml) and ml[i] in votes:
                    votes[ml[i]] += 1
            passing = [c for c in MINORITY if votes[c] >= minv(c)]
            if passing:
                best = max(passing, key=lambda c: (votes[c], -RARITY[c]))
                labels.append(best)
                added[best] += 1
            else:
                labels.append("Supported")
        out.append({"id": rid, "labels": labels})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tk, ad = sum(kept.values()), sum(added.values())
    print(f"asym_union -> {a.out}: kept {tk} base firings {kept} | ADDED {ad} {added} "
          f"| total minority = {tk + ad} | K_default={a.min_votes_default} per-class={mv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
