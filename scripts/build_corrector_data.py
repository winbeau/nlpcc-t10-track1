#!/usr/bin/env python3
"""Build PARAGRAPH-CORRECTOR training data (V-A, lightweight): from the DEV records (OOF — no model
trained on them) + an ensemble's DEV candidate labels + gold, make joint-corrector samples
(evidence + numbered sentences + INITIAL label per sentence -> corrected gold label array).

The corrector learns "trust the ensemble, fix its errors" — targeting PEM (fix the marginal 1-2
wrong sentences/paragraph). Splits dev into train/val (--val-frac) so the corrector can be validated
on held-out dev records (vs the raw ensemble) BEFORE spending a Codabench submission.

Inputs:
  --gold        dev_gold.jsonl (id, claim_text, evidence_bundle, sentence_label[gold types])
  --candidates  ensemble DEV predictions {id, labels} (e.g. outputs/dev_ensU5_sub.jsonl)
Outputs (to --out-dir):
  corrector_train.jsonl   ms-swift messages samples (corrector prompt + gold labels), train split
  corrector_val_ids.json  held-out dev record ids (for offline validation)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nlpcc_t10.build_dataset import (  # noqa: E402
    SYSTEM_PROMPT_CORRECTOR,
    build_user_content_corrector,
    load_jsonl,
)

# rarity order (rarest first) from train primary freq: Contra 101 < SO 188 < UCM 276 < UE 544
RARITY = ["Contradiction", "Scope Overgeneralization", "Unsupported Causal Mechanistic", "Unsupported Entity"]


def gold_label(types: list[str]) -> str:
    """Pick the rarest minority gold type if any (matches the per-sentence training-target convention),
    else Supported."""
    for lbl in RARITY:
        if lbl in types:
            return lbl
    return "Supported"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build paragraph-corrector training data from dev OOF candidates.")
    ap.add_argument("--gold", default="data/dev_gold.jsonl")
    ap.add_argument("--candidates", required=True, help="ensemble dev predictions {id,labels}")
    ap.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    data_root = Path(args.data_root).resolve()
    gold = load_jsonl(Path(args.gold))
    cand = {r["id"]: r["labels"] for r in load_jsonl(Path(args.candidates))}

    rng = random.Random(args.seed)
    ids = [r["id"] for r in gold]
    rng.shuffle(ids)
    n_val = int(len(ids) * args.val_frac)
    val_ids = set(ids[:n_val])

    train_samples = []
    n_diff = 0  # how many sentences where candidate != gold (correction signal)
    for rec in gold:
        rid = rec["id"]
        if rid in val_ids:
            continue
        sl = rec.get("sentence_label", [])
        sentences = [s.get("sentence", "") for s in sl]
        golds = [gold_label(s.get("types", [])) for s in sl]
        cands = cand.get(rid, ["Supported"] * len(sentences))
        if len(cands) != len(sentences):
            cands = (cands + ["Supported"] * len(sentences))[:len(sentences)]
        n_diff += sum(1 for a, b in zip(cands, golds) if a != b)
        user_text, image_paths = build_user_content_corrector(
            rec.get("claim_text", ""), sentences, cands, rec.get("evidence_bundle", []), data_root
        )
        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CORRECTOR},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": json.dumps({"labels": golds}, ensure_ascii=False)},
            ],
        }
        if image_paths:
            sample["images"] = image_paths
        train_samples.append(sample)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corrector_train.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in train_samples) + "\n", encoding="utf-8")
    (out_dir / "corrector_val_ids.json").write_text(json.dumps(sorted(val_ids)), encoding="utf-8")
    print(f"corrector_train: {len(train_samples)} samples (train split) | val held-out: {len(val_ids)} records")
    print(f"correction signal: {n_diff} sentences where ensemble candidate != gold (in train split)")
    print(f"-> {out_dir}/corrector_train.jsonl , {out_dir}/corrector_val_ids.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
