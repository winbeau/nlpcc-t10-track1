#!/usr/bin/env python3
"""Finalize CoT teaching units WITHOUT running the Stage-C audit loop.

Use when Stage C was interrupted (e.g. proxy outage) but enough has been audited: this
directly calls cot_distill.merge_final using whatever `phase:"final"` markers already exist
in verify.raw.jsonl. Audited units keep their PASS/rewritten verdict; un-audited units are
kept as verified=False/not_audited (same as if never audited). Writes units.jsonl +
label_only.jsonl over the full train' scope. No API calls.

  python scripts/cot_finalize.py --out data/cot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cot_distill as cd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", default="data/devbench/split.json")
    ap.add_argument("--data-root", default=cd.os.environ.get("DATA_ROOT", "../NLPCC-2026-Task10-Science"))
    ap.add_argument("--out", default="data/cot")
    ap.add_argument("--model", default="gpt-5.5")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    split = cd.load_split(Path(args.split_json))
    records = cd.load_traindev(data_root)
    by_id = dict(records)
    freq = cd.label_freq_train(records, split)
    scope_ids = [rid for rid, _ in records if split.get(rid) == "train"]
    scope_rows = {rid: cd.sentence_rows(rid, by_id[rid], freq) for rid in scope_ids}
    ctx = {"split": split, "by_id": by_id, "scope_ids": scope_ids,
           "scope_rows": scope_rows, "data_root": data_root, "key": None}

    class A:
        pass
    a = A(); a.out = args.out; a.model = args.model
    a.base = cd.os.environ.get("GPT5_BASE", "https://aiapis.help/v1")
    a.max_images = 6; a.audit_supported_frac = 0.25; a.seed = 0; a.workers = 6

    cd.merge_final(Path(args.out), ctx, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
