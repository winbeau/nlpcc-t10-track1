#!/usr/bin/env python3
"""Validate a Track 1 submission JSONL BEFORE uploading to Codabench / committing.

Mirrors what the official offline_eval/evaluate.py requires, so a PASS here means the
file is well-formed for scoring:
  1. every non-blank line is valid JSON and an object;
  2. each record has a non-empty string "id" and a list "labels";
  3. EVERY label is EXACTLY one of the 5 official Track 1 labels (case- and space-sensitive);
  4. per record, len(labels) == that id's sentence count in --ref (testp1-track-1.jsonl);
  5. the submission id set EXACTLY matches the ref id set (no missing / extra / duplicate);
  6. record count == ref count.

Prints a PASS/FAIL report + label distribution. Exit code 0 iff all checks pass.

Usage:
  python scripts/validate_submission.py \
      --sub outputs/testp1_b5_l0.5_submission.jsonl \
      --ref "$DATA_ROOT/data/testp1-track-1.jsonl"
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Exact official label strings (offline_eval/evaluate.py TRACK1_LABELS).
TRACK1_LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]
LABELSET = set(TRACK1_LABELS)


def ref_sentence_counts(ref_path: Path) -> tuple[dict[str, int], list[str]]:
    """Return (id -> sentence_count, ordered_ids). Uses the record's 'id' (testp1 ships ids);
    falls back to track1-{idx:06d} if absent (matches build_dataset / aggregate conventions)."""
    counts: dict[str, int] = {}
    order: list[str] = []
    with ref_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("id") if isinstance(rec.get("id"), str) and rec.get("id") else f"track1-{idx:06d}"
            sl = rec.get("sentence_label")
            if isinstance(sl, list):
                n = len(sl)
            elif isinstance(rec.get("sentences"), list):
                n = len(rec["sentences"])
            else:
                n = -1  # unknown -> will surface as a mismatch
            counts[rid] = n
            order.append(rid)
    return counts, order


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate a Track 1 submission JSONL.")
    ap.add_argument("--sub", required=True, help="submission JSONL: {id, labels} per line")
    ap.add_argument("--ref", required=True, help="reference JSONL (testp1-track-1.jsonl)")
    args = ap.parse_args(argv)

    sub_path, ref_path = Path(args.sub), Path(args.ref)
    if not sub_path.exists():
        print(f"[FAIL] submission not found: {sub_path}")
        return 1

    ref_n, ref_order = ref_sentence_counts(ref_path)
    ref_ids = set(ref_order)

    errors: list[str] = []
    seen_ids: list[str] = []
    label_counter: Counter = Counter()
    n_records = n_labels = 0

    with sub_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {lineno}: invalid JSON: {e}")
                continue
            if not isinstance(rec, dict):
                errors.append(f"line {lineno}: not a JSON object")
                continue
            n_records += 1
            rid = rec.get("id")
            if not isinstance(rid, str) or not rid:
                errors.append(f"line {lineno}: missing/empty string 'id'")
                continue
            seen_ids.append(rid)
            labels = rec.get("labels")
            if not isinstance(labels, list):
                errors.append(f"line {lineno} (id={rid}): 'labels' is not a list")
                continue
            for j, lb in enumerate(labels):
                if not isinstance(lb, str) or lb not in LABELSET:
                    errors.append(f"line {lineno} (id={rid}) labels[{j}]: invalid label {lb!r}")
                else:
                    label_counter[lb] += 1
                    n_labels += 1
            exp = ref_n.get(rid)
            if exp is None:
                errors.append(f"line {lineno}: id {rid!r} not present in reference")
            elif len(labels) != exp:
                errors.append(f"id={rid}: labels count {len(labels)} != ref sentence count {exp}")

    # id set integrity
    dups = [k for k, v in Counter(seen_ids).items() if v > 1]
    if dups:
        errors.append(f"duplicate ids ({len(dups)}): {dups[:10]}{'...' if len(dups) > 10 else ''}")
    missing = sorted(ref_ids - set(seen_ids))
    extra = sorted(set(seen_ids) - ref_ids)
    if missing:
        errors.append(f"{len(missing)} ref ids MISSING from submission: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"{len(extra)} EXTRA ids not in reference: {extra[:10]}{'...' if len(extra) > 10 else ''}")
    if n_records != len(ref_ids):
        errors.append(f"record count {n_records} != reference count {len(ref_ids)}")

    print(f"submission : {sub_path}")
    print(f"reference  : {ref_path}")
    print(f"records    : {n_records}  (ref {len(ref_ids)})")
    print(f"labels     : {n_labels}")
    print("label distribution:")
    for lb in TRACK1_LABELS:
        print(f"  {lb}: {label_counter.get(lb, 0)}")

    if errors:
        print(f"\n[FAIL] {len(errors)} problem(s):")
        for e in errors[:60]:
            print(f"  - {e}")
        if len(errors) > 60:
            print(f"  ... and {len(errors) - 60} more")
        return 1

    print("\n[PASS] well-formed: valid JSON, all labels in the 5 official labels, per-record "
          "counts match the reference, id set complete & unique. Ready to upload/commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
