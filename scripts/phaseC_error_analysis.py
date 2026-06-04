#!/usr/bin/env python3
"""D1 error analysis — where does a model fail on the densematch bench? (diagnostic, CPU, stdlib).

Inputs: a model's devbench dev prediction ({id,labels}) + a densematch gold (sentence_label[].types,
evidence_bundle[].type). The official scorer stays authoritative; this is DIAGNOSTIC to decide whether
D2 (frontier chart-reading) earns a Wave-3 slot — i.e. are minority misses concentrated in IMAGE records?

Reports (printed + notes/phaseC/D1_<tag>.json):
  - 5x5 sentence confusion (gold primary = rarest minority in types, else Supported; vs pred).
  - per-class precision/recall/F1 (any-of match, mirrors evaluate.py's hit rule).
  - bucket by RECORD evidence type (all-image / all-table / mixed): minority recall + share of misses.
  - minority recall by paragraph-length bin (<=5 / 6-8 / >=9).

  python scripts/phaseC_error_analysis.py --pred outputs/devbench_preds/s01.jsonl \
      --gold data/devbench/dev_gold_densematch.jsonl --tag s01
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
          "Scope Overgeneralization", "Contradiction"]
MIN = set(LABELS) - {"Supported"}
# rarity rank for picking the "primary" gold of a multi-label sentence (rarer first; diagnostic only)
RARITY = {"Contradiction": 0, "Unsupported Causal Mechanistic": 1, "Unsupported Entity": 2,
          "Scope Overgeneralization": 3, "Supported": 4}


def primary(types: list[str]) -> str:
    mins = [t for t in types if t in MIN]
    if mins:
        return min(mins, key=lambda t: RARITY[t])
    return "Supported"


def rec_evidence_type(rec: dict) -> str:
    ts = {e.get("type", "image") for e in rec.get("evidence_bundle", [])}
    if ts == {"image"}:
        return "all-image"
    if ts == {"table"}:
        return "all-table"
    return "mixed"


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = {r["id"]: r for r in load(args.gold)}
    pred = {r["id"]: r["labels"] for r in load(args.pred)}

    conf = Counter()                      # (gold_primary, pred) -> n
    tp = Counter(); fp = Counter(); fn = Counter()   # any-of per class
    ev_total = Counter(); ev_minmiss = Counter(); ev_minsupport = Counter()
    len_minhit = Counter(); len_minsupport = Counter()

    def lenbin(n): return "<=5" if n <= 5 else ("6-8" if n <= 8 else ">=9")

    n_rec = 0
    for rid, grec in gold.items():
        if rid not in pred:
            continue
        n_rec += 1
        ev = rec_evidence_type(grec)
        sl = grec["sentence_label"]
        plabs = pred[rid]
        L = lenbin(len(sl))
        for i, s in enumerate(sl):
            gtypes = s.get("types", [])
            gp = primary(gtypes)
            pl = plabs[i] if i < len(plabs) else "Supported"
            conf[(gp, pl)] += 1
            hit = pl in gtypes                       # any-of match (evaluate.py rule)
            # per-class P/R/F1 (any-of)
            for c in LABELS:
                if c in gtypes and pl == c:
                    tp[c] += 1
                elif c in gtypes and pl != c and pl not in gtypes:
                    fn[c] += 1
                elif c not in gtypes and pl == c:
                    fp[c] += 1
            # minority-miss bucketing
            is_min_gold = any(t in MIN for t in gtypes)
            if is_min_gold:
                ev_total[ev] += 1; ev_minsupport[ev] += 1; len_minsupport[L] += 1
                if hit:
                    len_minhit[L] += 1
                else:
                    ev_minmiss[ev] += 1

    # ---- print ----
    print(f"\n### D1 error analysis: {args.tag}  ({n_rec} records, gold={args.gold}) ###")
    print("\nConfusion (rows=gold primary, cols=pred):")
    hdr = "  " + "gold\\pred".ljust(34) + "".join(f"{l.split()[-1][:7]:>9}" for l in LABELS)
    print(hdr)
    for g in LABELS:
        row = "".join(f"{conf[(g, p)]:>9}" for p in LABELS)
        print("  " + g.ljust(34) + row)

    print("\nPer-class (any-of match):")
    print("  " + "label".ljust(34) + f"{'P':>7}{'R':>7}{'F1':>7}{'gold':>7}{'pred':>7}")
    perclass = {}
    for c in LABELS:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        perclass[c] = {"p": p, "r": r, "f1": f1, "gold": tp[c] + fn[c], "pred": tp[c] + fp[c]}
        print("  " + c.ljust(34) + f"{p*100:>6.1f} {r*100:>6.1f} {f1*100:>6.1f} {tp[c]+fn[c]:>7} {tp[c]+fp[c]:>7}")

    print("\nMinority gold by RECORD evidence type (the D2 question: are misses image-heavy?):")
    print("  " + "bucket".ljust(12) + f"{'min-gold':>9}{'min-miss':>9}{'recall%':>9}{'miss-share%':>12}")
    tot_miss = sum(ev_minmiss.values()) or 1
    for ev in ("all-image", "all-table", "mixed"):
        sup = ev_minsupport[ev]; miss = ev_minmiss[ev]
        rec = 100 * (sup - miss) / sup if sup else 0.0
        print("  " + ev.ljust(12) + f"{sup:>9}{miss:>9}{rec:>8.1f} {100*miss/tot_miss:>11.1f}")

    print("\nMinority recall by paragraph length:")
    for L in ("<=5", "6-8", ">=9"):
        sup = len_minsupport[L]
        rec = 100 * len_minhit[L] / sup if sup else 0.0
        print(f"  {L:<6} min-gold={sup:>4}  recall={rec:>5.1f}%")

    out = Path(args.out or f"notes/phaseC/D1_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag, "n_records": n_rec,
        "confusion": {f"{g}|{p}": conf[(g, p)] for g in LABELS for p in LABELS if conf[(g, p)]},
        "per_class": perclass,
        "evidence_buckets": {ev: {"min_gold": ev_minsupport[ev], "min_miss": ev_minmiss[ev]}
                             for ev in ("all-image", "all-table", "mixed")},
        "minrecall_by_len": {L: {"support": len_minsupport[L], "hit": len_minhit[L]}
                             for L in ("<=5", "6-8", ">=9")},
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
