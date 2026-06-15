#!/usr/bin/env python3
"""SIMI density-inject (Arm 2): synthesize LONG, minority-DENSE training paragraphs whose
labels stay VALID, to break the model's "<=1 minority per paragraph" false prior.

Construction (per the breakthrough workflow's label-clean SIMI subset):
  - HOST/anchor  = a train' record, len>=6, with EXACTLY ONE real minority sentence whose
                   class is in --anchor-classes (default UCM/Contradiction).
  - DONOR        = a train' minority SENTENCE, single-label, class in --donor-classes
                   (default UCM/Contradiction), with NO explicit "Table N / Figure N" ref.
  - PAIRING      = donor and anchor must be DOMAIN-DISJOINT (CV/NLP/RL/Graph/Speech/other,
                   tagged from evidence captions) AND share ZERO capitalized proper nouns
                   in their evidence captions (kills entity cross-contamination).
  - MERGE        = insert the donor sentence at a random interior position of the anchor
                   paragraph; the synthetic record's evidence_bundle = anchor evidence UNION
                   donor evidence. Result: a >=2-minority long paragraph.

Why labels survive (the make-or-break): UCM (causal-mechanism) and Contradiction (numbers)
claims are NOT satisfied/cancelled by adding an UNRELATED figure/table — so the donor's label
holds in the merged bundle (its own evidence is present), and the anchor's minority holds too
(restricting BOTH to UCM/Contra + domain-disjoint + no-figref removes the SO/UE "added-evidence
satisfies the scope" corruption that sinks the other constructions).

Output = PER-SENTENCE SFT samples (NOT joint -> avoids the measured -4.5 PEM joint tax),
reusing build_dataset.expand_record so the format is byte-identical to the base train_sft.
Built ONLY from devbench train' records (image-disjoint from the densematch gate). Concatenate
the output onto data/devbench/train_sft.jsonl, then train; the densematch eval is the GATE.
"""
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path

from nlpcc_t10.build_dataset import (
    load_jsonl, compute_global_label_freq, expand_record, write_jsonl,
)

MINORITY = {"Unsupported Causal Mechanistic", "Unsupported Entity",
            "Scope Overgeneralization", "Contradiction"}
FIGREF = re.compile(r"\b(?:Table|Figure|Fig|Tab)\.?\s*\d", re.IGNORECASE)
PROPER = re.compile(r"\b[A-Z][A-Za-z0-9\-]{2,}\b")  # benchmark/model/dataset names
DOMAIN_KW = {
    "CV": ["imagenet", "cifar", "coco", "resnet", "vit", "segmentation", "pixel",
           "image classification", "object detection", "convolution"],
    "NLP": ["bert", "glue", "squad", "bleu", "rouge", "summariz", "translation",
            "language model", "gpt", "token", "text classification"],
    "RL": ["reward", "policy", "agent", "environment", "atari", "mujoco", "episode",
           "q-learning", "reinforcement"],
    "Graph": ["graph", "gnn", "node", "edge", "gcn", "gin "],
    "Speech": ["speech", "audio", " asr", "phoneme", "acoustic"],
}


def ev_caption_text(rec: dict) -> str:
    parts = []
    for ev in rec.get("evidence_bundle", []):
        cap = ev.get("image_caption") or ev.get("table_caption") or []
        parts.append(" ".join(cap) if isinstance(cap, list) else str(cap))
    return " ".join(parts)


def domain_of(rec: dict) -> str:
    txt = (ev_caption_text(rec) + " " + rec.get("claim_text", "")).lower()
    best, best_c = "other", 0
    for d, kws in DOMAIN_KW.items():
        c = sum(txt.count(k) for k in kws)
        if c > best_c:
            best, best_c = d, c
    return best


def proper_nouns(rec: dict) -> set[str]:
    # proper nouns from evidence captions only (entity-overlap is what corrupts UE/Contra)
    return set(PROPER.findall(ev_caption_text(rec)))


def minority_sent_idxs(rec: dict) -> list[tuple[int, str]]:
    out = []
    for i, s in enumerate(rec.get("sentence_label", [])):
        ts = s.get("types", [])
        mins = [t for t in ts if t in MINORITY]
        if mins:
            out.append((i, mins[0]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", required=True,
                    help="data/devbench/split.json (train' boundary), or 'ALL' to use every "
                         "traindev record (submit-only final build, no dev holdout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-samples", type=int, default=420, help="stop after this many per-sentence samples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--donor-classes", default="Unsupported Causal Mechanistic,Contradiction")
    ap.add_argument("--anchor-classes", default="Unsupported Causal Mechanistic,Contradiction")
    ap.add_argument("--min-anchor-len", type=int, default=6)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    data_root = Path(a.data_root)
    records = load_jsonl(data_root / "data" / "traindev-track-1.jsonl")
    if a.split.upper() == "ALL":
        train_idx = list(range(len(records)))
    else:
        split = json.loads(Path(a.split).read_text(encoding="utf-8"))
        train_idx = [i for i in range(len(records)) if split.get(f"track1-{i:06d}") == "train"]
    label_freq = compute_global_label_freq([records[i] for i in train_idx])
    donor_cls = {c.strip() for c in a.donor_classes.split(",")}
    anchor_cls = {c.strip() for c in a.anchor_classes.split(",")}

    # ---- donor pool: single-label minority sentences, class in donor_cls, no fig/table ref ----
    donors = []  # (rec_idx, sent_idx, label)
    for i in train_idx:
        for si, s in enumerate(records[i].get("sentence_label", [])):
            ts = s.get("types", [])
            if len(ts) == 1 and ts[0] in donor_cls and not FIGREF.search(s.get("sentence", "")):
                donors.append((i, si, ts[0]))

    # ---- anchor pool: len>=min, EXACTLY 1 minority, that minority sentence SINGLE-label in
    # anchor_cls (single-label so expand_record's pick_rarest can't route it to UE/SO -> both
    # the anchor minority AND the injected donor minority are guaranteed clean UCM/Contra) ----
    anchors = []
    for i in train_idx:
        slabels = records[i].get("sentence_label", [])
        ms = minority_sent_idxs(records[i])
        if (len(slabels) >= a.min_anchor_len and len(ms) == 1
                and ms[0][1] in anchor_cls
                and len(slabels[ms[0][0]].get("types", [])) == 1):
            anchors.append(i)

    rng.shuffle(donors)
    rng.shuffle(anchors)
    dom = {i: domain_of(records[i]) for i in set([d[0] for d in donors] + anchors)}
    pn = {i: proper_nouns(records[i]) for i in set([d[0] for d in donors] + anchors)}

    from collections import Counter as _C
    dclass = _C(d[2] for d in donors)
    print(f"[pool] train' records={len(train_idx)} | donors={len(donors)} {dict(dclass)} | anchors={len(anchors)}")

    synth_samples = []
    used_anchor: set[int] = set()
    inj = _C()
    n_rec = 0
    for (ri, si, lab) in donors:
        if len(synth_samples) >= a.max_samples:
            break
        for ai in anchors:
            if ai in used_anchor or ai == ri:
                continue
            if dom[ai] == dom[ri]:
                continue
            if pn[ai] & pn[ri]:
                continue
            anchor, donor = records[ai], records[ri]
            asents = anchor.get("sentence_label", [])
            dsent = donor["sentence_label"][si]
            pos = rng.randint(1, len(asents))  # interior insert (not always last)
            new_sents = (asents[:pos]
                         + [{"sentence": dsent.get("sentence", ""), "types": [lab]}]
                         + asents[pos:])
            claim = " ".join(s.get("sentence", "") for s in new_sents)
            ev = anchor.get("evidence_bundle", []) + donor.get("evidence_bundle", [])
            synrec = {"claim_text": claim, "sentence_label": new_sents, "evidence_bundle": ev}
            samples = expand_record(synrec, 900000 + n_rec, label_freq, data_root, multi_target=False)
            synth_samples.extend([s for s, _ in samples])
            used_anchor.add(ai)
            inj[lab] += 1
            n_rec += 1
            break

    write_jsonl(synth_samples, Path(a.out))
    print(f"[done] synthetic records={n_rec} | per-sentence samples={len(synth_samples)} "
          f"| injected-minority dist={dict(inj)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
