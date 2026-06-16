#!/usr/bin/env python3
"""precision_trim.py — PEM-oriented precision trim on a winning union.

Leaderboard reframe (2026-06-16): xjdx_001 is Macro-F1 #1 (57.54 > chuhaicai 56.59)
but PEM-limited (43.69 vs 47.10). The marginal lever has flipped recall->precision.
This trims FP-prone *low-member-support* minority predictions (default: UE only, the
documented over-firing/low-precision class) from a union submission, flipping them
back to Supported. Pure local post-processing; no GPU.

FP-detector = how many of the diverse member models independently fire the SAME
minority label at that sentence position. Low support == union "reach" == FP-prone
(s79/s80 proved *added* UE is FP on testp1; this inverts that to TRIM).

Usage:
  python scripts/precision_trim.py \
      --union submissions/prod/testp1_s75_u_s15base-s70s56ucmContra.jsonl \
      --members submissions/testp1_s01_m_q8b-grid.jsonl ... \
      --trim "Unsupported Entity:1" \
      --out outputs/testp1_s87_pp_ue1trim.jsonl
"""
import argparse, json, collections, sys

SUP = "Supported"
MIN = {"Scope Overgeneralization", "Unsupported Entity",
       "Unsupported Causal Mechanistic", "Contradiction"}


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["id"]] = r["labels"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--union", required=True)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--trim", nargs="+", required=True,
                    help='per-class "Label:maxSupport" — flip union preds of that '
                         'class to Supported when <= maxSupport members agree. '
                         'e.g. "Unsupported Entity:1" trims single-member UE.')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # parse trim rules: class -> max_support (flip if support <= max_support)
    rules = {}
    for t in args.trim:
        lab, k = t.rsplit(":", 1)
        rules[lab] = int(k)

    union = load(args.union)
    members = [load(p) for p in args.members]
    ids = list(union.keys())

    flips = collections.Counter()
    out_records = []
    for rid in ids:
        labs = list(union[rid])
        for i, l in enumerate(labs):
            if l in rules and l in MIN:
                support = sum(1 for m in members
                              if i < len(m.get(rid, [])) and m[rid][i] == l)
                if support <= rules[l]:
                    labs[i] = SUP
                    flips[l] += 1
        out_records.append({"id": rid, "labels": labs})

    with open(args.out, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # report
    def dist(d):
        c = collections.Counter()
        for labs in d.values():
            for l in labs:
                c[l] += 1
        return c
    before = dist(union)
    after = dist({r["id"]: r["labels"] for r in out_records})
    print(f"[precision_trim] rules={rules}", file=sys.stderr)
    print(f"  flips (minority->Supported): {dict(flips)}  total={sum(flips.values())}",
          file=sys.stderr)
    print("  label dist  BEFORE -> AFTER:", file=sys.stderr)
    for k in [SUP, "Unsupported Entity", "Unsupported Causal Mechanistic",
              "Scope Overgeneralization", "Contradiction"]:
        print(f"    {k:34s} {before.get(k,0):5d} -> {after.get(k,0):5d}", file=sys.stderr)
    print(f"  wrote {args.out} ({len(out_records)} records)", file=sys.stderr)


if __name__ == "__main__":
    main()
