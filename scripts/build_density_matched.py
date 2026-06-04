#!/usr/bin/env python3
"""B1 density-matching — synthesize LONG + minority-DENSE joint paragraphs from train' records.

Root cause (notes/breakthrough_plan.md): traindev has ~≤1 minority sentence per paragraph (long
paragraphs: only 0.46% carry ≥2), so the model learns a near-deterministic "one minority per
paragraph" prior and systematically under-recalls in testp1's long, minority-dense paragraphs.

Fix: build synthetic paragraphs that carry ≥2 REAL minority sentences, by concatenating train'
records that share the EXACT SAME evidence (identical image-sha set) — so every spliced sentence's
label still holds against the same figure(s). Records are kept CONTIGUOUS (not interleaved) so each
record's internal Scope-Overgeneralization context is preserved; cross-record SO leakage is the only
residual label risk (densematch will tell us if it hurts).

IRON LAW: uses ONLY train' records (devbench split.json == "train"); NEVER touches dev'. Output is a
JOINT dataset (build_dataset.make_sample_joint format) = all real train' joint samples + the
synthetic dense ones, for a plain-CE joint train (the B1 model).

  uv run python scripts/build_density_matched.py --data-root "$DATA_ROOT" \
      --split data/devbench/split.json --out data/devbench/train_b1_joint.jsonl \
      --min-minorities 2 --target-len 12 --max-len 16
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nlpcc_t10.build_dataset import (  # noqa: E402
    MINORITY_LABELS, _record_image_stems, compute_global_label_freq, load_jsonl,
    make_sample_joint, pick_rarest_label, write_jsonl,
)


def rec_minorities(rec: dict, label_freq: Counter) -> int:
    return sum(1 for s in rec.get("sentence_label", [])
               if pick_rarest_label(s.get("types", []), label_freq) in MINORITY_LABELS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    ap.add_argument("--split", default="data/devbench/split.json")
    ap.add_argument("--out", default="data/devbench/train_b1_joint.jsonl")
    ap.add_argument("--min-minorities", type=int, default=2, help="a synthetic paragraph must carry >= this many minorities")
    ap.add_argument("--target-len", type=int, default=12, help="stop adding records once paragraph reaches ~this many sentences")
    ap.add_argument("--max-len", type=int, default=16, help="hard cap on synthetic paragraph length")
    ap.add_argument("--max-figures", type=int, default=4, help="cap distinct evidence figures per synthetic (token/memory + label-risk bound)")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    records = load_jsonl(data_root / "data" / "traindev-track-1.jsonl")
    split = json.loads(Path(args.split).read_text())
    train_idx = [i for i in range(len(records)) if split.get(f"track1-{i:06d}") == "train"]
    label_freq = compute_global_label_freq([records[i] for i in train_idx])
    print(f"train' records: {len(train_idx)}  (image-disjoint; never dev')")

    # 1) all REAL train' joint samples (the base)
    out_samples = []
    for i in train_idx:
        rec = records[i]
        sl = rec.get("sentence_label", [])
        sents = [s.get("sentence", "") for s in sl]
        labs = [pick_rarest_label(s.get("types", []), label_freq) for s in sl]
        out_samples.append(make_sample_joint(f"track1-{i:06d}", rec.get("claim_text", ""),
                                             sents, labs, rec.get("evidence_bundle", []), data_root))
    n_real = len(out_samples)

    # 2) hard records (>=1 minority) — the splice material. Greedily pack 2-3 into each synthetic
    #    so it carries >=min_minorities. Prefer co-evidence partners (share >=1 sha -> lowest label
    #    risk); evidence = union, capped at --max-figures. Contradiction/UCM (our bottleneck) are
    #    safe across records (a random extra figure rarely resolves them); UE/SO carry mild risk
    #    -> densematch is the arbiter (notes/exploration_roadmap.md B1).
    hard = [i for i in train_idx if rec_minorities(records[i], label_freq) >= 1]
    hard.sort(key=lambda i: (-rec_minorities(records[i], label_freq), i))  # deterministic; dense-first
    sha_of = {i: set(_record_image_stems(records[i])) for i in train_idx}

    def figs(idxs):  # union sha count
        u = set()
        for j in idxs:
            u |= sha_of[j]
        return len(u)

    used = set()
    reuse = Counter()
    n_synth = 0
    synth_minhist = Counter()
    synth_lenhist = []
    sid = 0
    for seed in hard:
        if seed in used:
            continue
        chosen = [seed]
        n_sent = len(records[seed]["sentence_label"])
        n_min = rec_minorities(records[seed], label_freq)
        # candidate partners: co-evidence first, then any other unused hard record
        partners = ([j for j in hard if j not in used and j != seed and sha_of[j] & sha_of[seed]]
                    + [j for j in hard if j not in used and j != seed and not (sha_of[j] & sha_of[seed])])
        for j in partners:
            if n_min >= args.min_minorities and n_sent >= args.target_len:
                break
            sl = len(records[j]["sentence_label"])
            if n_sent + sl > args.max_len or figs(chosen + [j]) > args.max_figures:
                continue
            chosen.append(j)
            n_sent += sl
            n_min += rec_minorities(records[j], label_freq)
        if len(chosen) < 2 or n_min < args.min_minorities:
            continue
        sents, labs, claim_parts, ev = [], [], [], []
        seen_sha = set()
        for j in chosen:
            rec = records[j]
            claim_parts.append(rec.get("claim_text", ""))
            for s in rec.get("sentence_label", []):
                sents.append(s.get("sentence", ""))
                labs.append(pick_rarest_label(s.get("types", []), label_freq))
            for e in rec.get("evidence_bundle", []):
                stem = e.get("img_path", "")
                if stem and stem not in seen_sha:
                    ev.append(e); seen_sha.add(stem)
            used.add(j)
        out_samples.append(make_sample_joint(f"track1-b1synth-{sid:05d}", " ".join(claim_parts),
                                             sents, labs, ev, data_root))
        sid += 1
        n_synth += 1
        synth_minhist[n_min] += 1
        synth_lenhist.append(len(sents))

    write_jsonl(out_samples, Path(args.out))
    ml = sorted(synth_lenhist)
    med = ml[len(ml)//2] if ml else 0
    print(f"\nB1 dataset -> {args.out}")
    print(f"  real train' joint samples : {n_real}")
    print(f"  synthetic dense paragraphs: {n_synth}  (each >= {args.min_minorities} minorities, <= {args.max_len} sentences)")
    print(f"  synthetic minority/para histogram: {dict(sorted(synth_minhist.items()))}")
    print(f"  synthetic length: median {med}, max {max(synth_lenhist) if synth_lenhist else 0}")
    print(f"  TOTAL samples: {len(out_samples)}")
    print("  (vs traindev where only 0.46% of long paragraphs carry >=2 minorities — this injects the missing geometry)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
