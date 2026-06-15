#!/usr/bin/env python3
"""Confidence-gated asymmetric union: fixed BASE + additions from source RAW files, where each
added firing must clear a per-class min_logprob confidence threshold.

Motivation: ungated surgical union peaks (s75: s15 + s70/s56 UCM/Contra = 50.61); adding UE/SO
ungated (s78) loses because those classes are FP-prone. But the source models' RAW predictions
carry per-sentence min_logprob — high-confidence UE/SO firings may still be correct. This tool
adds a source minority firing (at a position where BASE says Supported) only if min_logprob >=
the per-class threshold. UCM/Contra default threshold = -inf (always add, they're proven good);
UE/SO get a confidence cutoff. Base firings are kept verbatim (strict superset, recall-safe).

Source = an infer.py RAW jsonl: rows {id, sent_index, label, min_logprob, ...}. Output id/order
follows --ref. Pure stdlib.
"""
from __future__ import annotations
import argparse
import json
import math

LABELS = ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
          "Scope Overgeneralization", "Contradiction"]
MINORITY = [l for l in LABELS if l != "Supported"]
RARITY = {"Contradiction": 0, "Unsupported Causal Mechanistic": 1,
          "Unsupported Entity": 2, "Scope Overgeneralization": 3}


def ld_sub(p):
    return {json.loads(l)["id"]: json.loads(l)["labels"] for l in open(p, encoding="utf-8") if l.strip()}


def ld_raw(p):
    """RAW jsonl -> {(id, sent_idx): (label, min_logprob)}."""
    d = {}
    for l in open(p, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        d[(r["id"], int(r["sent_index"]))] = (r["label"], float(r["min_logprob"]))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="authority submission {id,labels}; kept verbatim")
    ap.add_argument("--source-raws", nargs="+", required=True, help="infer.py RAW jsonls (with min_logprob)")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default="",
                    help="per-class min_logprob threshold 'Label:thr,...'; a firing is added only if "
                         "min_logprob >= thr. Classes absent here default to -inf (always add).")
    a = ap.parse_args()

    thr = {}
    for it in a.conf.split(","):
        it = it.strip()
        if not it:
            continue
        k, _, v = it.rpartition(":")
        thr[k.strip()] = float(v)
    minconf = lambda c: thr.get(c, -math.inf)

    base = ld_sub(a.base)
    sources = [ld_raw(p) for p in a.source_raws]
    ref = [json.loads(l) for l in open(a.ref, encoding="utf-8") if l.strip()]

    kept = {c: 0 for c in MINORITY}
    added = {c: 0 for c in MINORITY}
    out = []
    for rec in ref:
        rid = rec["id"]
        n = len(rec.get("sentence_label", []))
        bl = base.get(rid, [])
        labels = []
        for i in range(n):
            blab = bl[i] if i < len(bl) else "Supported"
            if blab in MINORITY:
                labels.append(blab)
                kept[blab] += 1
                continue
            # base says Supported -> consider confidence-gated additions from sources
            cands = {}  # class -> best (highest) min_logprob across sources firing it
            for s in sources:
                lab, mlp = s.get((rid, i), ("Supported", -math.inf))
                if lab in MINORITY and mlp >= minconf(lab):
                    cands[lab] = max(cands.get(lab, -math.inf), mlp)
            if cands:
                # pick most-confident class; tie -> rarer
                best = max(cands, key=lambda c: (cands[c], -RARITY[c]))
                labels.append(best)
                added[best] += 1
            else:
                labels.append("Supported")
        out.append({"id": rid, "labels": labels})

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tk, ad = sum(kept.values()), sum(added.values())
    print(f"conf_gated -> {a.out}: kept {tk} {kept} | ADDED {ad} {added} | total {tk+ad} | conf={thr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
