"""Construct sentence-level train/dev datasets in ms-swift messages format.

Design contract (CLAUDE.md §3–§5 + softmin_loss_design.md §5.5):

Input:  $DATA_ROOT/data/traindev-track-1.jsonl (3333 records / 17547 sentences).

Steps:
  1) Split 9:1 BY RECORD (never by sentence: same-paragraph leakage + dev PEM meaningless
     if sentences are split across train/dev).  Stratified by whether a record contains any
     non-Supported sentence (keeps minority-class coverage in the 10% dev slice).
     Fixed seed; split written to split.json for reproducibility.

  2) Sentence-level expansion: each sentence -> 1 ms-swift sample.
     messages format (Qwen3-VL multimodal chat):
       system:    careful scientific claim verification prompt.
       user:      label definitions + evidence bundle images/captions + full claim_text
                  + "TARGET sentence: <this sentence>" (single sentence to classify).
                  Images are embedded as <image> tokens; `images` field holds paths.
       assistant: {"label": "<single label>"}   (training target).
     Multi-label sentence: pick the RAREST label by global frequency (benefits macro-F1).
     channel = paragraph_id (record index in the original JSONL, 0-based, formatted as
               "track1-{idx:06d}").  This field rides the ms-swift channel pipeline so the
               softmin PEM loss receives all per-sentence CEs of one paragraph together.

  3) Imbalance handling (TRAIN ONLY):
     Minority oversampling: each minority-class sample is duplicated up to
     `--minority-oversample` times (float; 2.0 = 2× = one extra copy).  Each duplicate
     gets a UNIQUE singleton channel id "<pid>#augK" so it CANNOT pollute the real
     paragraph's bottleneck in the softmin loss.
     Supported downsampling: drop entire all-Supported RECORDS with probability
     (1 - `--supported-downsample`) at the WHOLE-RECORD level (mixed/hard paragraphs are
     ALWAYS kept). This keeps each retained paragraph's channel group complete so the
     softmin bottleneck matches PEM's full-paragraph AND; per-sentence dropping would
     leave partial groups and break the loss semantic (review finding #3).
     DEV: never resampled (true distribution for honest PEM evaluation).

  4) dev_gold.jsonl: record-level, with id + full gold sentence_label (for evaluate.py).

Outputs (written to --out, default data/):
  train_sft.jsonl   sentence-level ms-swift samples, resampled
  dev_sft.jsonl     sentence-level ms-swift samples, true distribution
  dev_gold.jsonl    record-level {id, claim_text, evidence_bundle, sentence_label}
  split.json        {record_id -> "train"|"dev"} mapping

CLI:
  python -m nlpcc_t10.build_dataset \\
      --data-root ../NLPCC-2026-Task10-Science --out data/ \\
      --dev-ratio 0.1 --seed 42 \\
      --minority-oversample 1.0 --supported-downsample 1.0
  # --limit N: use only first N records (fast local smoke-test without torch/GPU)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

TRACK1_LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]
MINORITY_LABELS = set(TRACK1_LABELS) - {"Supported"}

SYSTEM_PROMPT = (
    "You are a careful scientific claim verification system. "
    "Given evidence (figures/tables with captions) and a claim paragraph, "
    "classify the target sentence into exactly one of five categories. "
    "Return valid JSON only, with no explanation or extra keys."
)

LABEL_DEFINITIONS = """\
Allowed labels:
- Supported: the sentence is fully supported by the evidence.
- Unsupported Causal Mechanistic: the sentence adds an unsupported causal or mechanistic explanation.
- Unsupported Entity: the sentence mentions an unsupported dataset, metric, model variant, baseline, or other scientific entity.
- Scope Overgeneralization: the sentence generalizes beyond the scope supported by the evidence.
- Contradiction: the sentence directly contradicts the evidence.

Output format: {"label": "<one of the five labels above>"}\
"""

# ── PARAGRAPH-JOINT mode (one forward per record, all N labels at once) ──────────────────
# Rationale (roadmap v2 rank-2): per-sentence isolation cannot see cross-sentence context that
# Scope Overgeneralization needs (a later sentence over-generalizes a scope set earlier); and one
# forward/record collapses the per-sentence image-prefix re-encode (up to 57x) -> makes 32B fit.
SYSTEM_PROMPT_JOINT = (
    "You are a careful scientific claim verification system. "
    "Given evidence (figures/tables with captions) and a claim paragraph, classify EACH numbered "
    "sentence into exactly one of five categories. Return valid JSON only, with no explanation or "
    "extra keys."
)

LABEL_DEFINITIONS_JOINT = """\
Allowed labels:
- Supported: the sentence is fully supported by the evidence.
- Unsupported Causal Mechanistic: the sentence adds an unsupported causal or mechanistic explanation.
- Unsupported Entity: the sentence mentions an unsupported dataset, metric, model variant, baseline, or other scientific entity.
- Scope Overgeneralization: the sentence generalizes beyond the scope supported by the evidence.
- Contradiction: the sentence directly contradicts the evidence.

Output format: {"labels": ["<label for sentence 1>", "<label for sentence 2>", ...]} \
— exactly one label per numbered sentence, in the same order, same count.\
"""

# ── CORRECTOR mode (paragraph-joint 2nd stage: review an ensemble's initial labels + fix) ────
# Targets PEM: most paragraphs are off by 1-2 sentences; correcting the marginal error flips the
# whole paragraph to PEM=1. The corrector sees the evidence + numbered sentences + the ensemble's
# INITIAL label per sentence, and outputs the corrected full label array (keep correct, fix wrong).
SYSTEM_PROMPT_CORRECTOR = (
    "You are a careful scientific claim verification system reviewing an initial labeling. "
    "Given evidence (figures/tables with captions), a claim paragraph with numbered sentences, and an "
    "INITIAL label guess for each sentence, output the CORRECTED label for every sentence: keep the "
    "ones that are right and fix the ones that are wrong, based strictly on the evidence. "
    "Return valid JSON only, no explanation or extra keys."
)


def build_user_content_corrector(
    claim_text: str,
    sentences: list[str],
    candidates: list[str],
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> tuple[str, list[str]]:
    """Corrector user text: evidence + claim + numbered sentences each tagged with its INITIAL label."""
    evidence_text, image_paths = evidence_to_text_and_images(evidence_bundle, data_root)
    lines = []
    for i, s in enumerate(sentences):
        cand = candidates[i] if i < len(candidates) else "Supported"
        lines.append(f"{i + 1}. {s}\n   [initial: {cand}]")
    numbered = "\n".join(lines)
    parts = [
        LABEL_DEFINITIONS_JOINT,
        "",
        evidence_text,
        "",
        f"Claim paragraph:\n{claim_text}",
        "",
        f"Sentences with initial labels (output the corrected label per sentence, in order):\n{numbered}",
    ]
    return "\n".join(parts), image_paths


# ──────────────────────────────────────────────────────────────────────────────
# I/O HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# LABEL-FREQUENCY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_label_freq(records: list[dict[str, Any]]) -> Counter:
    """Count primary label occurrences (types[0]) across all records."""
    freq: Counter = Counter()
    for rec in records:
        for sent in rec.get("sentence_label", []):
            types = sent.get("types", [])
            if types:
                freq[types[0]] += 1
    return freq


def pick_rarest_label(types: list[str], label_freq: Counter) -> str:
    """For a multi-label sentence, pick the label with lowest global frequency.
    Falls back to types[0] if the frequency counter is empty (shouldn't happen).
    """
    if not types:
        return "Supported"
    if len(types) == 1:
        return types[0]
    # Pick the label with the smallest count (rarest = most informative for macro-F1).
    return min(types, key=lambda lbl: label_freq.get(lbl, 0))


# ──────────────────────────────────────────────────────────────────────────────
# STRATIFIED RECORD SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def is_hard_record(rec: dict[str, Any]) -> bool:
    """True if any sentence has a non-Supported label (the ~33% PEM-hard paragraphs)."""
    for sent in rec.get("sentence_label", []):
        for lbl in sent.get("types", []):
            if lbl != "Supported":
                return True
    return False


def stratified_split(
    records: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return (train_indices, dev_indices) stratified by hard/easy.

    We stratify on `is_hard_record` so the 10% dev slice gets the same ~33% hard
    proportion as the full dataset, ensuring minority-class coverage in dev.
    """
    rng = random.Random(seed)

    hard = [i for i, r in enumerate(records) if is_hard_record(r)]
    easy = [i for i, r in enumerate(records) if not is_hard_record(r)]

    rng.shuffle(hard)
    rng.shuffle(easy)

    n_hard_dev = max(1, round(len(hard) * dev_ratio))
    n_easy_dev = max(1, round(len(easy) * dev_ratio))

    hard_dev, hard_train = hard[:n_hard_dev], hard[n_hard_dev:]
    easy_dev, easy_train = easy[:n_easy_dev], easy[n_easy_dev:]

    train_idx = sorted(hard_train + easy_train)
    dev_idx = sorted(hard_dev + easy_dev)
    return train_idx, dev_idx


def _record_image_stems(rec: dict[str, Any]) -> set[str]:
    """Image identifiers (sha stems) referenced by a record's evidence_bundle."""
    stems: set[str] = set()
    for ev in rec.get("evidence_bundle", []):
        ip = ev.get("img_path", "")
        if ip:
            stems.add(Path(ip).stem)
    return stems


def build_image_components(records: list[dict[str, Any]]) -> list[list[int]]:
    """Connected components of records linked when they SHARE an image sha (union-find).
    Records sharing no image are singleton components. This is the leakage boundary: an
    image-disjoint split must keep each whole component on one side."""
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    img_first: dict[str, int] = {}  # image stem -> first record index seen
    for i, rec in enumerate(records):
        for stem in _record_image_stems(rec):
            if stem in img_first:
                union(i, img_first[stem])
            else:
                img_first[stem] = i

    comps: dict[int, list[int]] = {}
    for i in range(len(records)):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def component_aware_split(
    records: list[dict[str, Any]],
    dev_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Image-DISJOINT 9:1-style split: whole image-sharing components go to ONE side, so NO
    image sha appears in both train and dev (kills the ~32% cross-split image leak that made
    the old dev read 88 vs testp1 50.26). Greedy-fills dev with whole components (deterministic
    seed shuffle) until ~round(n*dev_ratio) records, then HARD-ASSERTS zero sha overlap."""
    comps = build_image_components(records)
    rng = random.Random(seed)
    # Deterministic order: largest components first (so a giant component can't silently
    # blow past the target by landing last), then seeded shuffle within for reproducibility.
    comps.sort(key=lambda c: (-len(c), min(c)))
    rng.shuffle(comps)

    n = len(records)
    target_dev = round(n * dev_ratio)
    dev_idx: list[int] = []
    train_idx: list[int] = []
    for comp in comps:
        if len(dev_idx) < target_dev:
            dev_idx.extend(comp)
        else:
            train_idx.extend(comp)

    train_stems: set[str] = set().union(*(_record_image_stems(records[i]) for i in train_idx)) if train_idx else set()
    dev_stems: set[str] = set().union(*(_record_image_stems(records[i]) for i in dev_idx)) if dev_idx else set()
    overlap = train_stems & dev_stems
    assert not overlap, f"component_aware_split BUG: {len(overlap)} image shas overlap across the split"

    return sorted(train_idx), sorted(dev_idx)


# ──────────────────────────────────────────────────────────────────────────────
# EVIDENCE BUNDLE -> USER MESSAGE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def evidence_to_text_and_images(
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> tuple[str, list[str]]:
    """Return (evidence_text_block, image_paths_list).

    For each evidence item:
      - Appends an <image> token to the user text (Qwen3-VL multimodal convention).
      - Collects the absolute image path in the images list.
    Both image and table items have img_path (path relative to data_root).
    Caption is a list of strings; join them.
    """
    text_parts: list[str] = ["Evidence:"]
    image_paths: list[str] = []

    for idx, item in enumerate(evidence_bundle, 1):
        item_type = item.get("type", "image")
        caption_raw = item.get("image_caption") or item.get("table_caption") or []
        caption = " ".join(caption_raw).strip() if isinstance(caption_raw, list) else str(caption_raw)
        img_path_rel = item.get("img_path", "")
        if img_path_rel:
            # Images live under <data_root>/data/ (same dir as the jsonl). The jsonl img_path is
            # like "images/<sha>.jpg", so the absolute path is <data_root>/data/images/<sha>.jpg
            # (matches the official `unzip data/images.zip -d data/images/` layout). The earlier
            # data_root/img_path was missing the 'data/' segment -> images would not resolve.
            abs_path = str(data_root / "data" / img_path_rel)
            image_paths.append(abs_path)
            text_parts.append(f"[{item_type.upper()} {idx}] <image>")
        else:
            text_parts.append(f"[{item_type.upper()} {idx}]")
        if caption:
            text_parts.append(f"Caption: {caption}")

    return "\n".join(text_parts), image_paths


def build_user_content(
    claim_text: str,
    target_sentence: str,
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> tuple[str, list[str]]:
    """Build the user message text and image path list for one sentence sample."""
    evidence_text, image_paths = evidence_to_text_and_images(evidence_bundle, data_root)

    parts = [
        LABEL_DEFINITIONS,
        "",
        evidence_text,
        "",
        f"Claim paragraph:\n{claim_text}",
        "",
        f"TARGET sentence to classify:\n{target_sentence}",
    ]
    return "\n".join(parts), image_paths


# ──────────────────────────────────────────────────────────────────────────────
# SAMPLE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def make_sample(
    record_id: str,
    channel: str,
    claim_text: str,
    target_sentence: str,
    target_label: str,
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    """Build one ms-swift sentence-level sample in messages format."""
    user_text, image_paths = build_user_content(
        claim_text, target_sentence, evidence_bundle, data_root
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": json.dumps({"label": target_label}, ensure_ascii=False)},
    ]
    sample: dict[str, Any] = {
        "messages": messages,
        "channel": channel,  # paragraph_id; rides ms-swift channel pipeline for softmin loss
    }
    if image_paths:
        sample["images"] = image_paths
    return sample


def expand_record(
    rec: dict[str, Any],
    record_idx: int,
    label_freq: Counter,
    data_root: Path,
) -> list[dict[str, Any]]:
    """Expand one record into per-sentence samples. All get the same channel (paragraph_id).
    Returns list of (sample, label) tuples — label used downstream for resampling."""
    paragraph_id = f"track1-{record_idx:06d}"
    claim_text = rec.get("claim_text", "")
    evidence_bundle = rec.get("evidence_bundle", [])
    sentence_labels = rec.get("sentence_label", [])

    samples = []
    for sent_item in sentence_labels:
        sentence = sent_item.get("sentence", "")
        types = sent_item.get("types", [])
        target_label = pick_rarest_label(types, label_freq)
        sample = make_sample(
            record_id=paragraph_id,
            channel=paragraph_id,
            claim_text=claim_text,
            target_sentence=sentence,
            target_label=target_label,
            evidence_bundle=evidence_bundle,
            data_root=data_root,
        )
        samples.append((sample, target_label))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# PARAGRAPH-JOINT SAMPLE CONSTRUCTION (one sample per RECORD; plain-CE training)
# ──────────────────────────────────────────────────────────────────────────────

def build_user_content_joint(
    claim_text: str,
    sentences: list[str],
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> tuple[str, list[str]]:
    """User text + images for a paragraph-joint sample: evidence + claim + NUMBERED sentences."""
    evidence_text, image_paths = evidence_to_text_and_images(evidence_bundle, data_root)
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    parts = [
        LABEL_DEFINITIONS_JOINT,
        "",
        evidence_text,
        "",
        f"Claim paragraph:\n{claim_text}",
        "",
        f"Sentences to classify (output exactly one label per sentence, in order):\n{numbered}",
    ]
    return "\n".join(parts), image_paths


def make_sample_joint(
    record_id: str,
    claim_text: str,
    sentences: list[str],
    labels: list[str],
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> dict[str, Any]:
    """One ms-swift sample for a whole paragraph; assistant target = {"labels": [...N...]}."""
    user_text, image_paths = build_user_content_joint(claim_text, sentences, evidence_bundle, data_root)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_JOINT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": json.dumps({"labels": labels}, ensure_ascii=False)},
    ]
    sample: dict[str, Any] = {"messages": messages, "channel": record_id}
    if image_paths:
        sample["images"] = image_paths
    return sample


def expand_record_joint(
    rec: dict[str, Any],
    record_idx: int,
    label_freq: Counter,
    data_root: Path,
) -> tuple[dict[str, Any], bool]:
    """One record -> one joint sample. Returns (sample, has_minority)."""
    pid = f"track1-{record_idx:06d}"
    sent_items = rec.get("sentence_label", [])
    sentences = [s.get("sentence", "") for s in sent_items]
    labels = [pick_rarest_label(s.get("types", []), label_freq) for s in sent_items]
    has_minority = any(l != "Supported" for l in labels)
    sample = make_sample_joint(
        pid, rec.get("claim_text", ""), sentences, labels, rec.get("evidence_bundle", []), data_root
    )
    return sample, has_minority


def build_dataset_joint(
    data_root: Path,
    out_dir: Path,
    dev_ratio: float,
    seed: int,
    hard_oversample: float = 2.0,
    limit: int | None = None,
    split_mode: str = "random",
) -> dict[str, Any]:
    """Paragraph-JOINT pipeline: one sample per record (all N labels as a JSON array). Train-time
    imbalance = duplicate WHOLE minority-containing records `hard_oversample`x (fresh channel id),
    which lifts the minority fraction WITHOUT the per-sentence singleton trick (plain-CE, no softmin).
    Reuses the same 9:1 split + train-only label_freq as the per-sentence build (split_mode:
    'random' stratified, or 'component_aware' image-disjoint)."""
    data_path = data_root / "data" / "traindev-track-1.jsonl"
    records = load_jsonl(data_path)
    if limit is not None:
        records = records[:limit]
    print(f"[JOINT] Loaded {len(records)} records")

    if split_mode == "component_aware":
        train_idx, dev_idx = component_aware_split(records, dev_ratio, seed)
    else:
        train_idx, dev_idx = stratified_split(records, dev_ratio, seed)
    print(f"[JOINT] Split [{split_mode}]: {len(train_idx)} train / {len(dev_idx)} dev records")
    split_map: dict[str, str] = {}
    for i in train_idx:
        split_map[f"track1-{i:06d}"] = "train"
    for i in dev_idx:
        split_map[f"track1-{i:06d}"] = "dev"

    label_freq = compute_global_label_freq([records[i] for i in train_idx])

    rng = random.Random(seed)
    train_samples: list[dict[str, Any]] = []
    n_hard = 0
    for i in train_idx:
        sample, has_minority = expand_record_joint(records[i], i, label_freq, data_root)
        train_samples.append(sample)
        if has_minority:
            n_hard += 1
            n_extra_float = hard_oversample - 1.0
            n_extra = int(math.floor(n_extra_float)) + (1 if rng.random() < (n_extra_float - math.floor(n_extra_float)) else 0)
            for k in range(n_extra):
                dup = dict(sample)
                dup["messages"] = [dict(m) for m in sample["messages"]]
                if "images" in sample:
                    dup["images"] = list(sample["images"])
                dup["channel"] = f"{sample['channel']}#dup{k}"
                train_samples.append(dup)
    rng.shuffle(train_samples)

    dev_samples = [expand_record_joint(records[i], i, label_freq, data_root)[0] for i in dev_idx]
    dev_gold = [make_dev_gold_record(records[i], i) for i in dev_idx]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_samples, out_dir / "train_sft.jsonl")
    write_jsonl(dev_samples, out_dir / "dev_sft.jsonl")
    write_jsonl(dev_gold, out_dir / "dev_gold.jsonl")
    (out_dir / "split.json").write_text(json.dumps(split_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # label distribution across all sentence positions in train (informational)
    lc: Counter = Counter()
    for s in train_samples:
        lc.update(json.loads(s["messages"][2]["content"])["labels"])
    print(f"[JOINT] hard_oversample={hard_oversample}x on {n_hard} minority-containing train records")
    print(f"[JOINT] train samples (records): {len(train_samples)} | dev: {len(dev_samples)}")
    print("[JOINT] train per-sentence-position label distribution:")
    for lbl in TRACK1_LABELS:
        print(f"  {lbl}: {lc[lbl]}")
    print(f"Wrote to {out_dir}: train_sft.jsonl({len(train_samples)}) dev_sft.jsonl({len(dev_samples)}) "
          f"dev_gold.jsonl({len(dev_gold)}) split.json")
    return {"n_train_samples": len(train_samples), "n_dev_samples": len(dev_samples)}


# ──────────────────────────────────────────────────────────────────────────────
# RESAMPLING (TRAIN ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def resample_train(
    sample_label_pairs: list[tuple[dict[str, Any], str]],
    minority_oversample: float,
    rng: random.Random,
    per_class: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Apply MINORITY OVERSAMPLING to training samples.

    NOTE: Supported *downsampling* is intentionally NOT done here. Dropping
    individual Supported sentences would leave their minority siblings under the
    same `channel`, producing a PARTIAL paragraph group whose softmin bottleneck
    no longer matches PEM's full-paragraph AND (review finding #3, CRITICAL).
    Supported is instead downsampled at the WHOLE-RECORD level in build_dataset
    (drop entire all-Supported paragraphs), keeping every retained paragraph's
    channel group complete.

    Minority oversampling: each minority-class sample is duplicated up to
    floor(minority_oversample - 1) extra times, plus a fractional final copy.
    E.g. minority_oversample=2.5 -> each minority sample appears 2 or 3 times.
    Each duplicate gets a UNIQUE singleton channel "<pid>#augK" so it reduces to
    its own CE term and cannot pollute its real paragraph's bottleneck.

    The result is shuffled.
    """
    result: list[dict[str, Any]] = []
    aug_counter: dict[str, int] = {}  # pid -> number of augmented copies so far

    for sample, label in sample_label_pairs:
        # Original copy: always kept with its real paragraph channel.
        result.append(sample)
        if label == "Supported":
            continue

        # Augmented minority copies: each gets a unique singleton channel.
        # Per-class rate overrides the global float when provided (target the MEASURED
        # under-fired classes — testp1 SO/Contradiction starved — without inflating UE).
        orig_channel = sample["channel"]
        rate = per_class.get(label, minority_oversample) if per_class else minority_oversample
        n_extra_float = rate - 1.0
        if n_extra_float > 0:
            n_extra_full = int(math.floor(n_extra_float))
            frac = n_extra_float - n_extra_full
            n_extra = n_extra_full + (1 if rng.random() < frac else 0)
            for _ in range(n_extra):
                k = aug_counter.get(orig_channel, 0)
                aug_counter[orig_channel] = k + 1
                # Deep-copy the nested lists so a downstream in-place collator edit
                # cannot corrupt every shared copy (review finding #11).
                aug_sample = dict(sample)
                aug_sample["messages"] = [dict(m) for m in sample["messages"]]
                if "images" in sample:
                    aug_sample["images"] = list(sample["images"])
                aug_sample["channel"] = f"{orig_channel}#aug{k}"
                result.append(aug_sample)

    rng.shuffle(result)
    return result


def downsample_supported_records(
    train_idx: list[int],
    records: list[dict[str, Any]],
    supported_downsample: float,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Drop WHOLE all-Supported train records with prob (1 - supported_downsample).

    Mixed/hard records (containing any non-Supported sentence) are ALWAYS kept so
    the softmin PEM loss sees every sentence of each retained paragraph (channel
    group integrity). Returns (kept_indices, n_dropped).
    """
    if supported_downsample >= 1.0:
        return list(train_idx), 0
    kept: list[int] = []
    dropped = 0
    for i in train_idx:
        if is_hard_record(records[i]):
            kept.append(i)            # mixed paragraph: keep intact for the bottleneck
        elif rng.random() < supported_downsample:
            kept.append(i)            # all-Supported: keep with prob
        else:
            dropped += 1
    return kept, dropped


# ──────────────────────────────────────────────────────────────────────────────
# DEV GOLD CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def make_dev_gold_record(rec: dict[str, Any], record_idx: int) -> dict[str, Any]:
    """Build a gold record for the official evaluator.
    The evaluator's extract_track1_gold_labels expects sentence_label[i].types (non-empty list).
    """
    paragraph_id = f"track1-{record_idx:06d}"
    return {
        "id": paragraph_id,
        "claim_text": rec.get("claim_text", ""),
        "evidence_bundle": rec.get("evidence_bundle", []),
        "sentence_label": rec.get("sentence_label", []),
    }


# ──────────────────────────────────────────────────────────────────────────────
# MAIN BUILD FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    data_root: Path,
    out_dir: Path,
    dev_ratio: float,
    seed: int,
    minority_oversample: float,
    supported_downsample: float,
    limit: int | None = None,
    per_class_oversample: dict[str, float] | None = None,
    split_mode: str = "random",
) -> dict[str, Any]:
    """Full pipeline: load, split, expand, resample, write. Returns summary dict."""
    data_path = data_root / "data" / "traindev-track-1.jsonl"
    records = load_jsonl(data_path)
    if limit is not None:
        records = records[:limit]
    print(f"Loaded {len(records)} records from {data_path}")

    # 1) 9:1 split by record (BEFORE computing label frequency, so dev sentences do not
    #    contaminate multi-label target selection — review finding #9). split_mode:
    #    'random' = stratified-by-hard (leaky: shares images across split);
    #    'component_aware' = image-DISJOINT (kills the leak; for a trustworthy testp1-tracking dev).
    if split_mode == "component_aware":
        train_idx, dev_idx = component_aware_split(records, dev_ratio, seed)
        print(f"Split [component_aware / image-disjoint]: {len(train_idx)} train / {len(dev_idx)} dev records")
    else:
        train_idx, dev_idx = stratified_split(records, dev_ratio, seed)
        print(f"Split [random / stratified]: {len(train_idx)} train records / {len(dev_idx)} dev records")
    print(f"  train hard (any minority): "
          f"{sum(is_hard_record(records[i]) for i in train_idx)}/{len(train_idx)}")
    print(f"  dev   hard (any minority): "
          f"{sum(is_hard_record(records[i]) for i in dev_idx)}/{len(dev_idx)}")

    # Build split.json from the canonical 9:1 split (the leakage boundary; the
    # Supported downsampling below is a train-time filter, logged separately).
    split_map: dict[str, str] = {}
    for i in train_idx:
        split_map[f"track1-{i:06d}"] = "train"
    for i in dev_idx:
        split_map[f"track1-{i:06d}"] = "dev"

    # 2) Global label frequency from TRAIN records ONLY (drives pick_rarest_label).
    label_freq = compute_global_label_freq([records[i] for i in train_idx])
    print("Global label frequency (TRAIN-only, primary types[0]):")
    for lbl in TRACK1_LABELS:
        print(f"  {lbl}: {label_freq[lbl]}")

    # 3) Supported downsampling at the WHOLE-RECORD level (drop entire all-Supported
    #    paragraphs only) to keep every retained paragraph's channel group complete.
    ds_rng = random.Random(seed + 1)
    train_use_idx, n_dropped = downsample_supported_records(
        train_idx, records, supported_downsample, ds_rng
    )
    if n_dropped:
        print(f"Supported downsample: dropped {n_dropped} all-Supported train records "
              f"-> {len(train_use_idx)}/{len(train_idx)} kept")

    # 4) Expand: sentence-level samples
    train_pairs: list[tuple[dict[str, Any], str]] = []
    for i in train_use_idx:
        train_pairs.extend(expand_record(records[i], i, label_freq, data_root))

    dev_pairs: list[tuple[dict[str, Any], str]] = []
    for i in dev_idx:
        dev_pairs.extend(expand_record(records[i], i, label_freq, data_root))

    print(f"Expanded: {len(train_pairs)} train sentences / {len(dev_pairs)} dev sentences")

    # Label distribution in train (before resampling)
    train_label_counter: Counter = Counter(lbl for _, lbl in train_pairs)
    print("Train label distribution (before resampling):")
    for lbl in TRACK1_LABELS:
        print(f"  {lbl}: {train_label_counter[lbl]}")

    # 5) Resample train (minority oversampling only; Supported already downsampled
    #    at the record level in step 3 to preserve paragraph channel integrity).
    rng = random.Random(seed)
    if per_class_oversample:
        print(f"Per-class oversample rates: {per_class_oversample} "
              f"(others default to {minority_oversample}x)")
    train_samples = resample_train(train_pairs, minority_oversample, rng, per_class_oversample)
    print(f"After resampling: {len(train_samples)} train samples")

    # Label distribution after resampling
    resampled_counter: Counter = Counter(
        json.loads(s["messages"][2]["content"])["label"] for s in train_samples
    )
    print("Train label distribution (after resampling):")
    for lbl in TRACK1_LABELS:
        print(f"  {lbl}: {resampled_counter[lbl]}")

    # 5) Dev samples (no resampling)
    dev_samples = [s for s, _ in dev_pairs]

    # 6) Dev gold records
    dev_gold = [make_dev_gold_record(records[i], i) for i in dev_idx]

    # 7) Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_samples, out_dir / "train_sft.jsonl")
    write_jsonl(dev_samples, out_dir / "dev_sft.jsonl")
    write_jsonl(dev_gold, out_dir / "dev_gold.jsonl")
    (out_dir / "split.json").write_text(
        json.dumps(split_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote to {out_dir}:")
    print(f"  train_sft.jsonl  ({len(train_samples)} rows)")
    print(f"  dev_sft.jsonl    ({len(dev_samples)} rows)")
    print(f"  dev_gold.jsonl   ({len(dev_gold)} rows)")
    print(f"  split.json       ({len(split_map)} record ids)")

    return {
        "n_records": len(records),
        "n_train_records": len(train_idx),
        "n_train_records_used": len(train_use_idx),
        "n_train_records_dropped_supported": n_dropped,
        "n_dev_records": len(dev_idx),
        "n_train_sentences_before_resample": len(train_pairs),
        "n_train_sentences_after_resample": len(train_samples),
        "n_dev_sentences": len(dev_samples),
        "n_dev_gold_records": len(dev_gold),
        "train_label_freq_before": dict(train_label_counter),
        "train_label_freq_after": dict(resampled_counter),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build sentence-level ms-swift dataset for NLPCC Track 1."
    )
    p.add_argument(
        "--data-root",
        default="../NLPCC-2026-Task10-Science",
        help="Path to the official data repo root (contains data/ + images/).",
    )
    p.add_argument(
        "--out",
        default="data/",
        help="Output directory for train_sft.jsonl / dev_sft.jsonl / dev_gold.jsonl / split.json.",
    )
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-mode",
        choices=["random", "component_aware"],
        default="random",
        help="'random' = stratified-by-hard (default; back-compat — SHARES images across split = leaky). "
             "'component_aware' = IMAGE-DISJOINT split via union-find over image shas (no sha in both "
             "train and dev) -> a trustworthy testp1-tracking dev (kills the ~32%% cross-split image leak).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Override --dev-ratio for the val/dev fraction (used with --split-mode component_aware, "
             "e.g. 0.15 for a bigger, more stable devhard). Falls back to --dev-ratio when unset.",
    )
    p.add_argument(
        "--minority-oversample",
        type=float,
        default=1.0,
        help=(
            "Float >= 1.0. Minority class sentences are duplicated to this multiplier. "
            "1.0 = no oversampling; 2.0 = each minority sentence appears twice; "
            "2.5 = appears 2 or 3 times (fractional probability). "
            "Tune by 9:1 dev SCORE, not minority recall."
        ),
    )
    p.add_argument(
        "--supported-downsample",
        type=float,
        default=1.0,
        help=(
            "Float in (0, 1]. Fraction of all-Supported RECORDS to KEEP (whole-record "
            "drop; mixed/hard paragraphs are always kept to preserve channel-group "
            "integrity for the softmin loss). 1.0 = keep all; 0.5 = drop ~50%% of "
            "all-Supported records. Tune by 9:1 dev SCORE, not minority recall."
        ),
    )
    p.add_argument(
        "--minority-oversample-per-class",
        default=None,
        help=(
            "Optional PER-CLASS oversample rates overriding --minority-oversample for the named "
            "labels. Format: 'Label:rate,Label:rate' e.g. "
            "'Contradiction:6,Scope Overgeneralization:5,Unsupported Causal Mechanistic:3,Unsupported Entity:2'. "
            "Target the MEASURED under-fired classes (testp1 SO/Contradiction starved) without inflating UE. "
            "Tune by Codabench leaderboard, not the leakage-inflated dev."
        ),
    )
    p.add_argument(
        "--joint",
        action="store_true",
        help="PARAGRAPH-JOINT mode: one sample per record (all N labels as a JSON array), plain-CE. "
             "Enables cross-sentence context (helps Scope Overgeneralization) and collapses the "
             "per-sentence image re-encode (makes 32B feasible). Uses build_dataset_joint().",
    )
    p.add_argument(
        "--joint-hard-oversample",
        type=float,
        default=2.0,
        help="JOINT mode only: duplicate WHOLE minority-containing records this many times "
             "(fresh channel id) to lift the minority fraction. 1.0 = no oversample.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N records (for fast local smoke-testing).",
    )
    return p


def _parse_per_class(spec: str | None) -> dict[str, float] | None:
    """Parse 'Label:rate,Label:rate' -> {label: rate}. Labels may contain spaces (no colons),
    so split each item on the LAST colon. Validates labels against TRACK1_LABELS."""
    if not spec:
        return None
    valid = set(TRACK1_LABELS)
    out: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, val = item.rpartition(":")
        key = key.strip()
        if key not in valid:
            raise ValueError(
                f"--minority-oversample-per-class: unknown label {key!r}; must be one of {TRACK1_LABELS}"
            )
        out[key] = float(val)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out).resolve()
    ratio = args.val_ratio if args.val_ratio is not None else args.dev_ratio
    if args.joint:
        build_dataset_joint(
            data_root=data_root,
            out_dir=out_dir,
            dev_ratio=ratio,
            seed=args.seed,
            hard_oversample=args.joint_hard_oversample,
            limit=args.limit,
            split_mode=args.split_mode,
        )
        return 0
    build_dataset(
        data_root=data_root,
        out_dir=out_dir,
        dev_ratio=ratio,
        seed=args.seed,
        minority_oversample=args.minority_oversample,
        supported_downsample=args.supported_downsample,
        limit=args.limit,
        per_class_oversample=_parse_per_class(args.minority_oversample_per_class),
        split_mode=args.split_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
