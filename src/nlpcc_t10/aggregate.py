"""Aggregate sentence-level raw predictions → official Track 1 submission format.

Design contract (CLAUDE.md §2, §8):

  - Reads infer.py's sentence-level raw JSONL: each line is
      {"id": <record_id>, "sent_index": <int>, "label": "<label>", "logprob"?: <float>}
    where <record_id> is the paragraph/record id (e.g. "track1-p1-test-000008"),
    <sent_index> is 0-based, and <label> is one of the 5 Track 1 labels.

  - Groups by id, sorts by sent_index, emits one line per record:
      {"id": <id>, "labels": [<label_for_sent_0>, ...]}
    The label count MUST equal the number of sentences in the reference record.
    - Missing sentences: filled with FALLBACK_LABEL ("Supported") and warned.
    - Extra sentences (index >= ref count): dropped and warned.

  - --ref <path>: reference JSONL (traindev or testp1 split) used to:
      a) establish the expected sentence count per id.
      b) define the output id order (same order as ref).
      If ref records have no "id" field, synthetic ids "track1-{i:06d}" are used
      (matching build_dataset.py's paragraph_id convention).

  - --postprocess: optional paragraph-level threshold biasing:
      Sentences whose logprob < --supported-threshold are reset to "Supported",
      reducing minority-class fire-rate and protecting PEM on uncertain paragraphs.
      Choose threshold on 9:1 dev via official score; keep 0.0 (disabled) for pure
      model output.

CLI:
  python -m nlpcc_t10.aggregate \\
      --pred outputs/testp1_raw.jsonl \\
      --ref  $DATA_ROOT/data/testp1-track-1.jsonl \\
      --out  outputs/testp1_submission.jsonl \\
      [--postprocess --supported-threshold 0.6]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

TRACK1_LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]
FALLBACK_LABEL = "Supported"
LABEL_SET = set(TRACK1_LABELS)


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            records.append(obj)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Reference parsing: id → sentence count, plus ordered id list
# ──────────────────────────────────────────────────────────────────────────────

def parse_ref(ref_records: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    """Return (ordered_ids, id_to_sent_count).

    If a record has a string "id" field it is used directly; otherwise the
    synthetic id "track1-{i:06d}" is generated (0-based, matching
    build_dataset.py's paragraph_id convention).
    """
    ordered_ids: list[str] = []
    id_to_count: dict[str, int] = {}
    for i, rec in enumerate(ref_records):
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            rid = f"track1-{i:06d}"
        if rid in id_to_count:
            raise ValueError(f"Duplicate id in reference file: {rid!r}")
        # sentence count: from sentence_label list (gold) or sentences list
        sent_label = rec.get("sentence_label")
        if isinstance(sent_label, list):
            cnt = len(sent_label)
        elif isinstance(rec.get("sentences"), list):
            cnt = len(rec["sentences"])
        else:
            raise ValueError(
                f"Reference record {rid!r} has no 'sentence_label' or 'sentences' list"
            )
        ordered_ids.append(rid)
        id_to_count[rid] = cnt
    return ordered_ids, id_to_count


# ──────────────────────────────────────────────────────────────────────────────
# Prediction grouping
# ──────────────────────────────────────────────────────────────────────────────

def group_preds(
    raw_preds: list[dict[str, Any]],
) -> dict[str, list[tuple[int, str, float | None]]]:
    """Group raw predictions by id.

    Returns {id: [(sent_index, label, logprob_or_None), ...]} unsorted.
    Warns on invalid / unknown labels and replaces with FALLBACK_LABEL.
    """
    groups: dict[str, list[tuple[int, str, float | None]]] = defaultdict(list)
    for i, row in enumerate(raw_preds):
        rid = row.get("id")
        if not isinstance(rid, str) or not rid:
            warnings.warn(f"Row {i}: missing or non-string 'id', skipping.")
            continue
        sent_index = row.get("sent_index")
        if not isinstance(sent_index, int):
            warnings.warn(
                f"Row {i} (id={rid!r}): 'sent_index' is not an int ({sent_index!r}), skipping."
            )
            continue
        label = row.get("label", "")
        if label not in LABEL_SET:
            warnings.warn(
                f"Row {i} (id={rid!r}, sent={sent_index}): "
                f"unknown label {label!r}, replaced with {FALLBACK_LABEL!r}."
            )
            label = FALLBACK_LABEL
        logprob = row.get("logprob")
        if logprob is not None and not isinstance(logprob, (int, float)):
            logprob = None
        groups[rid].append((sent_index, label, logprob))
    return dict(groups)


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing: threshold-based biasing toward Supported
# ──────────────────────────────────────────────────────────────────────────────

def apply_postprocess(
    labels: list[str],
    logprobs: list[float | None],
    threshold: float,
) -> list[str]:
    """For each sentence whose logprob is below threshold (or missing), override
    any minority prediction to Supported.  Sentences already predicted as
    Supported or whose logprob >= threshold are left unchanged.

    threshold=0.0 (or negative) effectively disables this (logprob is ≤ 0 in
    log-space; we compare against the NEGATED logprob so higher confidence =
    lower |logprob|).  Use positive threshold values, e.g. 0.5 means
    |logprob| > 0.5 → override to Supported.
    """
    result = []
    for label, lp in zip(labels, logprobs):
        if label == FALLBACK_LABEL:
            result.append(label)
            continue
        # lp is a log-probability (≤ 0); |lp| is the uncertainty.
        # Override if we lack a logprob or if |logprob| > threshold.
        if lp is None or (-lp) > threshold:
            result.append(FALLBACK_LABEL)
        else:
            result.append(label)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Core aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(
    raw_preds: list[dict[str, Any]],
    ref_records: list[dict[str, Any]],
    postprocess: bool = False,
    supported_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Aggregate sentence-level preds into record-level submission.

    Returns a list of {"id": ..., "labels": [...]} dicts in the same order as
    ref_records.
    """
    ordered_ids, id_to_count = parse_ref(ref_records)
    groups = group_preds(raw_preds)

    # Guard (review finding #15): --postprocess with all-None logprobs would override
    # EVERY minority prediction to Supported (lp is None -> treated as "uncertain"),
    # silently producing an all-Supported submission. Disable instead of footgun.
    if postprocess and supported_threshold > 0.0:
        any_lp = any(lp is not None for rows in groups.values() for (_, _, lp) in rows)
        if not any_lp:
            print(
                "[aggregate] ERROR: --postprocess requested but ALL logprobs are None "
                "(was infer.py run with --no-logprob?). It would convert every minority "
                "prediction to 'Supported'. DISABLING postprocess.",
                file=sys.stderr,
            )
            postprocess = False

    # Warn about ids in preds but not in ref
    pred_ids = set(groups)
    ref_ids = set(ordered_ids)
    extra = pred_ids - ref_ids
    if extra:
        warnings.warn(
            f"{len(extra)} prediction id(s) not found in reference (will be ignored): "
            + repr(sorted(extra)[:5]) + ("..." if len(extra) > 5 else "")
        )

    submission: list[dict[str, Any]] = []
    missing_count = 0
    extra_sent_count = 0

    for rid in ordered_ids:
        expected = id_to_count[rid]
        rows = groups.get(rid, [])

        # Sort by sent_index
        rows.sort(key=lambda t: t[0])

        # Check for duplicate sent_index
        seen_indices: set[int] = set()
        deduped: list[tuple[int, str, float | None]] = []
        for (si, lbl, lp) in rows:
            if si in seen_indices:
                warnings.warn(
                    f"id={rid!r}: duplicate sent_index={si}, keeping first occurrence."
                )
            else:
                seen_indices.add(si)
                deduped.append((si, lbl, lp))
        rows = deduped

        # Build dense array indexed 0..expected-1
        labels: list[str] = [FALLBACK_LABEL] * expected
        logprobs: list[float | None] = [None] * expected

        for (si, lbl, lp) in rows:
            if si >= expected:
                extra_sent_count += 1
                warnings.warn(
                    f"id={rid!r}: sent_index={si} >= expected count {expected}, dropping."
                )
                continue
            labels[si] = lbl
            logprobs[si] = lp

        # Count positions that were filled with fallback due to missing preds
        filled_indices = {si for (si, _, _) in rows if si < expected}
        n_missing = expected - len(filled_indices)
        if n_missing > 0:
            missing_count += n_missing
            warnings.warn(
                f"id={rid!r}: {n_missing} sentence(s) missing from predictions, "
                f"filled with {FALLBACK_LABEL!r}."
            )

        # Optional postprocessing
        if postprocess and supported_threshold > 0.0:
            labels = apply_postprocess(labels, logprobs, supported_threshold)

        submission.append({"id": rid, "labels": labels})

    if missing_count > 0:
        print(
            f"[aggregate] WARNING: {missing_count} total missing sentence predictions "
            f"filled with {FALLBACK_LABEL!r}.",
            file=sys.stderr,
        )
    if extra_sent_count > 0:
        print(
            f"[aggregate] WARNING: {extra_sent_count} out-of-range sentence predictions dropped.",
            file=sys.stderr,
        )

    return submission


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Aggregate sentence-level preds into Track 1 submission format."
    )
    p.add_argument(
        "--pred", required=True,
        help="Sentence-level raw JSONL from infer.py "
             "({id, sent_index, label, logprob?} per line)."
    )
    p.add_argument(
        "--ref", required=True,
        help="Reference JSONL (traindev or testp1 split) — used for id order and sentence counts."
    )
    p.add_argument("--out", required=True, help="Output submission JSONL path.")
    p.add_argument(
        "--postprocess", action="store_true",
        help="Enable paragraph-level threshold biasing toward Supported (protects PEM)."
    )
    p.add_argument(
        "--supported-threshold", type=float, default=0.0,
        help="When --postprocess: override any non-Supported prediction to Supported "
             "if |logprob| > this value (higher = more conservative; tune on dev SCORE). "
             "0.0 = disabled."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    pred_path = Path(args.pred)
    ref_path = Path(args.ref)
    out_path = Path(args.out)

    if not pred_path.exists():
        print(f"error: pred file not found: {pred_path}", file=sys.stderr)
        return 2
    if not ref_path.exists():
        print(f"error: ref file not found: {ref_path}", file=sys.stderr)
        return 2

    print(f"[aggregate] Loading predictions from {pred_path} ...", file=sys.stderr)
    raw_preds = load_jsonl(pred_path)
    print(f"[aggregate] Loaded {len(raw_preds)} sentence-level rows.", file=sys.stderr)

    print(f"[aggregate] Loading reference from {ref_path} ...", file=sys.stderr)
    ref_records = load_jsonl(ref_path)
    print(f"[aggregate] Loaded {len(ref_records)} reference records.", file=sys.stderr)

    submission = aggregate(
        raw_preds=raw_preds,
        ref_records=ref_records,
        postprocess=args.postprocess,
        supported_threshold=args.supported_threshold,
    )

    write_jsonl(submission, out_path)

    total_sents = sum(len(r["labels"]) for r in submission)
    print(
        f"[aggregate] Wrote {len(submission)} records / {total_sents} sentences → {out_path}",
        file=sys.stderr,
    )
    if args.postprocess and args.supported_threshold > 0.0:
        print(
            f"[aggregate] Postprocess enabled: threshold={args.supported_threshold}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
