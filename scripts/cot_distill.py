#!/usr/bin/env python3
"""CoT teaching-unit distillation for Track-1 (plan: notes/cot_distill_plan.md).

Three stages (run in order; each is resumable and idempotent):
  --stage blind   Stage A: per-record k=2 blind sampling with the proven gpt5_infer
                  prompt (SYSTEM + FEWSHOT). Parse the per-sentence `i) ...; verdict: X`
                  lines; a record sample whose numbered lines do not align 1..N exactly
                  (or has no well-formed FINAL line) is DISCARDED and resampled
                  (--max-tries), never best-effort aligned. A sentence's reasoning line
                  becomes a unit iff canon(verdict) == target_label (pick_rarest over
                  train'-only primary freq, mirroring build_dataset). Both correct k
                  samples are kept (downstream rotates rationales to avoid templating).
                  NOTE: plan §2 "(或 ∈ gold_types)" is intentionally narrowed to strict
                  == target_label — a rationale arguing a different gold type than the
                  training target would pair inconsistently downstream; any-of hits are
                  still counted in --stats agreement.
  --stage gold    Stage B: per-sentence gold-conditioned rationale for sentences not
                  covered by blind units. CANNOT_JUSTIFY escape hatch -> label_only.
                  Malformed/overlong output: 1 reworded retry, then label_only.
  --stage verify  Stage C: grounding audit of ALL minority units + a seeded
                  --audit-supported-frac (default 10%) sample of Supported units.
                  FAIL on a minority unit -> inline Stage-B rewrite (1x) -> re-verify;
                  FAIL again -> dropped to label_only. Finishes by merging the final
                  units.jsonl + label_only.jsonl.
  --stats         No API calls: coverage / agreement / CJ rate / PASS rate / throughput
                  / call counts -> <out>/stats.md (reporting duty: model+version+calls).

Red lines (notes/cot_distill_plan.md §1):
  * train'-only, twice-asserted: every scoped record AND every written unit must have
    split.json[record_id] == "train". dev' is never read into scope.
  * rationale: plain English text, <=45 words, no trailing `-> <label>` / verdict tail.

Rationales: every unit row carries BOTH `target_label` (training target) and the raw
text; raw model outputs are always persisted next to the structured files.

Files in --out (default data/cot/; pilot uses data/cot/pilot/):
  sample_records.json     pilot scope (created by --stage blind w/ --records-sample)
  blind.raw.jsonl         every Stage A call (status ok|misaligned|api_error|gave_up)
  units_blind.jsonl       derived from blind.raw.jsonl (rebuilt each run)
  gold_cond.raw.jsonl     every Stage B call (status ok|cannot_justify|bad_output|...)
  units_gold.jsonl        derived from gold_cond.raw.jsonl (rebuilt each run)
  verify.raw.jsonl        every Stage C call + one terminal `phase:"final"` marker/unit
  units.jsonl             FINAL merged teaching units (written by --stage verify)
  label_only.jsonl        scoped sentences with no surviving unit (+ reason)
  stats.md                written by --stats

Env (same proxy chain as gpt5_infer.py): GPT5_KEY (required), GPT5_BASE, GPT5_MODEL
(default gpt-5.5), GPT5_REASONING_EFFORT (optional).

Usage (pilot, on the H200 box where images are unzipped):
  GPT5_KEY=... uv run --no-project python scripts/cot_distill.py --stage blind \
      --out data/cot/pilot --records-sample 50 --min-minority-sents 30 --workers 6
  ... --stage gold   --out data/cot/pilot --workers 6
  ... --stage verify --out data/cot/pilot --workers 6
  ... --stats        --out data/cot/pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpt5_infer as g5  # reuse SYSTEM / FEWSHOT_BLOCK / call_api / b64_image / canon

LABELS = g5.LABELS
MINORITY = [l for l in LABELS if l != "Supported"]

# Provenance of the 6 FEWSHOT_BLOCK examples (verified 2026-06-10, all split=='train'):
# Ex1<-track1-000000#1  Ex2<-track1-000010#0  Ex3<-track1-000069#1 (exact)
# Ex4<-track1-000023#4  Ex5<-track1-000077#5  Ex6<-track1-000606#4
FEWSHOT_SOURCE_IDS = ["track1-000000", "track1-000010", "track1-000069",
                      "track1-000023", "track1-000077", "track1-000606"]

MAX_WORDS = 45
MIN_WORDS = 5

LABEL_DEF = {
    "Supported": "the sentence is consistent with / backed by the evidence",
    "Contradiction": "a specific number / direction / comparison in the sentence "
                     "conflicts with a value visible in the evidence",
    "Unsupported Entity": "the sentence cites a dataset / metric / model / baseline / "
                          "method or a result for it that does not appear in the evidence",
    "Unsupported Causal Mechanistic": "the sentence asserts a specific causal mechanism "
                                      "('because', 'due to', ...) the evidence does not establish",
    "Scope Overgeneralization": "the sentence generalizes beyond what the evidence shows "
                                "('all datasets', 'always', 'in general', uncovered settings)",
}

GOLD_SYSTEM = (
    "You are a careful scientific figure/table reader. You write SHORT, GROUNDED "
    "rationales used to train a verifier model. You never invent numbers or entities: "
    "every specific value/entity you mention must be visible in the provided evidence "
    "images or captions. If you cannot ground the requested label in the evidence, you "
    "say so instead of making something up."
)

VERIFY_SYSTEM = (
    "You are a meticulous auditor of evidence-grounded rationales. You check every "
    "specific value / entity / axis range a rationale cites against the provided "
    "figure/table evidence, and you fail rationales that cite things which are not "
    "actually there, misread values, or cite nothing specific at all."
)


# ───────────────────────────── data loading / scope ──────────────────────────


def load_split(path: Path) -> dict[str, str]:
    return json.load(open(path, encoding="utf-8"))


def load_traindev(data_root: Path) -> list[tuple[str, dict]]:
    """[(record_id, record)] with record_id = track1-{line:06d} (devbench convention)."""
    p = data_root / "data" / "traindev-track-1.jsonl"
    out = []
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            if line.strip():
                out.append((f"track1-{lineno:06d}", json.loads(line)))
    return out


def label_freq_train(records: list[tuple[str, dict]], split: dict[str, str]) -> Counter:
    """Primary (types[0]) frequency over train' ONLY — mirrors build_dataset."""
    freq: Counter = Counter()
    for rid, rec in records:
        if split.get(rid) != "train":
            continue
        for s in rec.get("sentence_label", []):
            types = s.get("types", [])
            if types:
                freq[types[0]] += 1
    return freq


def pick_rarest(types: list[str], freq: Counter) -> str:
    if not types:
        return "Supported"
    if len(types) == 1:
        return types[0]
    return min(types, key=lambda l: freq.get(l, 0))


def sentence_rows(rid: str, rec: dict, freq: Counter) -> list[dict]:
    rows = []
    for si, s in enumerate(rec.get("sentence_label", [])):
        types = s.get("types", []) or ["Supported"]
        rows.append({"record_id": rid, "sent_idx": si, "sentence": s["sentence"],
                     "gold_types": types, "target_label": pick_rarest(types, freq)})
    return rows


def select_scope(args, records, split) -> list[str]:
    """Pilot sampling (--records-sample, forced minority floor) or full train'.
    Persisted to <out>/sample_records.json so gold/verify/stats reuse the same scope."""
    sample_path = Path(args.out) / "sample_records.json"
    train_ids = [rid for rid, _ in records if split.get(rid) == "train"]
    if sample_path.exists():
        meta = json.load(open(sample_path, encoding="utf-8"))
        ids = meta["ids"]
        assert all(split.get(r) == "train" for r in ids), "scope leaked out of train'"
        return ids
    if not args.records_sample:
        return train_ids
    freq = label_freq_train(records, split)
    by_id = dict(records)

    def n_minor(rid: str) -> int:
        return sum(1 for r in sentence_rows(rid, by_id[rid], freq)
                   if r["target_label"] != "Supported")

    rng = random.Random(args.seed)
    shuffled = train_ids[:]
    rng.shuffle(shuffled)
    chosen = shuffled[: args.records_sample]
    rest = shuffled[args.records_sample:]
    # greedily swap minority-poorest picks for minority-bearing remainder until floor met
    ri = 0
    while sum(map(n_minor, chosen)) < args.min_minority_sents and ri < len(rest):
        cand = rest[ri]; ri += 1
        if n_minor(cand) == 0:
            continue
        worst = min(range(len(chosen)), key=lambda i: n_minor(chosen[i]))
        if n_minor(chosen[worst]) < n_minor(cand):
            chosen[worst] = cand
    total_minor = sum(map(n_minor, chosen))
    if total_minor < args.min_minority_sents:
        print(f"WARNING: only {total_minor} minority sentences in sample "
              f"(< {args.min_minority_sents})", file=sys.stderr)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    json.dump({"seed": args.seed, "n_records": len(chosen),
               "n_minority_sents": total_minor, "ids": chosen},
              open(sample_path, "w", encoding="utf-8"), indent=1)
    print(f"sampled {len(chosen)} train' records ({total_minor} minority sentences) "
          f"-> {sample_path}")
    return chosen


# ───────────────────────────── shared helpers ────────────────────────────────


def caption_text(rec: dict) -> str:
    lines = ["EVIDENCE captions:"]
    for j, ev in enumerate(rec.get("evidence_bundle", [])):
        cap = ev.get("image_caption") or ev.get("table_caption") or ev.get("caption") or []
        if isinstance(cap, list):
            cap = " ".join(map(str, cap))
        lines.append(f"[evidence {j + 1} | {ev.get('type', '?')}]: {cap}")
    return "\n".join(lines)


def image_parts(rec: dict, data_root: Path, max_images: int) -> list[dict]:
    parts = []
    for ev in rec.get("evidence_bundle", []):
        ip = ev.get("img_path")
        if ip and len(parts) < max_images:
            cand = data_root / "data" / ip
            if not cand.exists():
                cand = data_root / ip
            if not cand.exists():
                cand = Path(ip)
            uri = g5.b64_image(cand)
            if uri:
                parts.append({"type": "image_url", "image_url": {"url": uri}})
    return parts


LABEL_ALT = "|".join(re.escape(l) for l in LABELS)
# Strip ONLY delimiter-anchored structural label tails (`-> X`, `verdict: X`, `(X)`).
# A label phrase that is the natural last words of the prose ("...is an unsupported
# entity") is genuine reasoning text and must be kept — the red line targets the
# structural tail, not English mentioning a category.
TAIL_PATTERNS = [
    re.compile(r"[;,.\s]*\b(verdict|label)\s*:?\s*(" + LABEL_ALT + r")[\s).\"']*$", re.IGNORECASE),
    re.compile(r"[;,.\s]*(->|→|=>|⇒|—|–)\s*(" + LABEL_ALT + r")[\s).\"']*$", re.IGNORECASE),
    re.compile(r"[;,.\s]*\(\s*→?\s*(" + LABEL_ALT + r")\s*\)[\s.\"']*$", re.IGNORECASE),
    re.compile(r"[;,.\s]*\bverdict\s*:\s*[A-Za-z ]{0,35}$", re.IGNORECASE),
]


def clean_rationale(text: str) -> str:
    """Collapse whitespace; strip leading 'RATIONALE:'/'claim:' scaffold and any
    trailing structural verdict/arrow/label tail (red line: no label tail)."""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"^(RATIONALE\s*:\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^claim\s*:\s*", "claim ", t, flags=re.IGNORECASE)
    prev = None
    while prev != t:
        prev = t
        for pat in TAIL_PATTERNS:
            t = pat.sub("", t).strip()
        t = re.sub(r"(->|→|=>)\s*$", "", t).strip()
        t = re.sub(r"[(\[\s]+$", "", t).strip()  # orphan opener left by a stripped tail
    return t.rstrip(";,. ").strip()


def n_words(t: str) -> int:
    return len(t.split())


def unit_key(u: dict) -> str:
    h = hashlib.sha1(u["rationale"].encode()).hexdigest()[:10]
    return f"{u['record_id']}|{u['sent_idx']}|{u['source']}|{h}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


class RawLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")
        self.lock = Lock()

    def write(self, row: dict):
        row.setdefault("ts", time.time())
        with self.lock:
            self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.f.flush()

    def close(self):
        self.f.close()


def write_units(path: Path, units: list[dict], split: dict[str, str]):
    """Rewrite a derived units file. RED LINE assert #2: every unit must be train'."""
    for u in units:
        assert split.get(u["record_id"]) == "train", \
            f"unit {u['record_id']} not in train' — refusing to write"
        assert u["rationale"] and n_words(u["rationale"]) <= MAX_WORDS
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"wrote {len(units)} units -> {path}")


# ───────────────────────────── Stage A: blind ────────────────────────────────

NUM_LINE = re.compile(r"^\s*(\d+)\)\s*(.*)$")


def parse_blind(text: str, n: int):
    """Strict parse of the gpt5_infer output shape. Returns (entries 1..n, final
    n labels) or (None, None) when misaligned — caller discards & resamples."""
    entries: dict[int, str] = {}
    cur = None
    final = None
    for ln in text.splitlines():
        if re.match(r"\s*FINAL\s*:", ln, re.IGNORECASE):
            seg = ln.split(":", 1)[1]
            if "|||" not in seg:
                return None, None
            final = [g5.canon(x) for x in seg.split("|||")]
            break
        m = NUM_LINE.match(ln)
        if m:
            i = int(m.group(1))
            if i in entries:
                return None, None  # duplicate index = misaligned
            entries[i] = m.group(2).strip()
            cur = i
        elif ln.strip() and cur is not None:
            entries[cur] += " " + ln.strip()
    if final is None or len(final) != n or set(entries) != set(range(1, n + 1)):
        return None, None
    return entries, final


def split_verdict(entry: str):
    """-> (rationale_part, verdict_label|None) from one `i)` line.
    The real verdict tail is at the END of the line — anchor on the LAST 'verdict'
    occurrence so prose like 'the verdict is clear that ...' never truncates the
    rationale. A long tail after the match is prose, not a verdict — keep it all."""
    ms = list(re.finditer(r"verdict\s*:?", entry, re.IGNORECASE))
    if not ms:
        return entry.strip(), None
    m = ms[-1]
    verd = entry[m.end():].strip(" :;,.")
    if not verd or len(verd) > 40:
        return entry.strip(), None
    return entry[: m.start()].strip(), g5.canon(verd)


def blind_sent_units(entries: dict[int, str], final: list[str], srows: list[dict]):
    """Per-sentence verdict + cleaned rationale + acceptance, for one parsed sample."""
    out = []
    for r in srows:
        i = r["sent_idx"] + 1
        rat_part, verd = split_verdict(entries[i])
        verdict = verd or final[r["sent_idx"]]
        rat = clean_rationale(rat_part)
        wc = n_words(rat)
        accepted, reason = False, ""
        if verdict != r["target_label"]:
            reason = "verdict!=target"
        elif wc > MAX_WORDS:
            reason = "overlength"
        elif wc < MIN_WORDS:
            reason = "too_short"
        else:
            accepted = True
        out.append({"sent_idx": r["sent_idx"], "verdict": verdict, "rationale": rat,
                    "n_words": wc, "accepted": accepted, "reason": reason,
                    "hit_target": verdict == r["target_label"],
                    "hit_anyof": verdict in r["gold_types"]})
    return out


def rebuild_units_blind(out_dir: Path, scope_rows: dict[str, list[dict]],
                        split: dict[str, str], model: str):
    units = []
    seen = set()
    for row in read_jsonl(out_dir / "blind.raw.jsonl"):
        if row.get("status") != "ok":
            continue
        rid = row["record_id"]
        if rid not in scope_rows:
            continue
        for su in row.get("sent_units", []):
            if not su["accepted"]:
                continue
            r = scope_rows[rid][su["sent_idx"]]
            u = {"record_id": rid, "sent_idx": su["sent_idx"], "sentence": r["sentence"],
                 "gold_types": r["gold_types"], "target_label": r["target_label"],
                 "rationale": su["rationale"], "source": "blind", "verified": False,
                 "verifier_note": "", "model": model, "ts": row.get("ts"),
                 "n_tries": row.get("try", 1)}
            k = unit_key(u)
            if k not in seen:  # dedupe identical rationale text per sentence
                seen.add(k)
                units.append(u)
    write_units(out_dir / "units_blind.jsonl", units, split)
    return units


def run_blind(args, ctx):
    out_dir, key = Path(args.out), ctx["key"]
    raw = RawLog(out_dir / "blind.raw.jsonl")
    terminal = {(r["record_id"], r["k"]) for r in read_jsonl(out_dir / "blind.raw.jsonl")
                if r.get("status") in ("ok", "gave_up")}
    tasks = [(rid, k) for rid in ctx["scope_ids"] for k in range(args.k)
             if (rid, k) not in terminal]
    print(f"Stage A blind: scope={len(ctx['scope_ids'])} records x k={args.k}; "
          f"done={len(terminal)} todo={len(tasks)} model={args.model} workers={args.workers}")

    def work(rid: str, k: int):
        rec = ctx["by_id"][rid]
        srows = ctx["scope_rows"][rid]
        msgs, n = g5.build_messages(rec, ctx["data_root"], args.max_images)
        msgs[0]["content"] = g5.SYSTEM + "\n\n" + g5.FEWSHOT_BLOCK
        for t in range(1, args.max_tries + 1):
            t0 = time.time()
            try:
                text = g5.call_api(msgs, args.model, args.base, key)
            except Exception as e:  # noqa: BLE001
                raw.write({"record_id": rid, "k": k, "try": t, "status": "api_error",
                           "err": f"{type(e).__name__}: {e}", "dur_s": round(time.time() - t0, 1)})
                continue
            entries, final = parse_blind(text, n)
            if entries is None:
                raw.write({"record_id": rid, "k": k, "try": t, "status": "misaligned",
                           "raw": text, "dur_s": round(time.time() - t0, 1)})
                continue
            raw.write({"record_id": rid, "k": k, "try": t, "status": "ok", "raw": text,
                       "sent_units": blind_sent_units(entries, final, srows),
                       "dur_s": round(time.time() - t0, 1)})
            return "ok"
        raw.write({"record_id": rid, "k": k, "try": args.max_tries, "status": "gave_up"})
        return "gave_up"

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, rid, k) for rid, k in tasks]
        for fut in as_completed(futs):
            fut.result()
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(tasks)} samples done")
    raw.close()
    rebuild_units_blind(out_dir, ctx["scope_rows"], ctx["split"], args.model)


# ───────────────────────────── Stage B: gold-conditioned ─────────────────────


def gold_user_text(rec: dict, row: dict, attempt: int, fail_note: str = "") -> str:
    lab = row["target_label"]
    parts = [caption_text(rec), "\nFULL CLAIM:\n" + rec.get("claim_text", ""),
             "\nTARGET SENTENCE:\n" + row["sentence"],
             f"\nEXPERT GOLD LABEL for the target sentence: {lab}\n"
             f"(label meaning: {LABEL_DEF[lab]})"]
    if fail_note:
        parts.append("\nA previous rationale for this sentence FAILED an evidence audit: "
                     + fail_note + "\nWrite a NEW rationale strictly grounded in what is "
                     "actually visible in the evidence.")
    strictness = ("STRICT: at most 40 words, one sentence preferred. " if attempt > 1 else "")
    parts.append(
        "\nTask: write ONE short rationale (<=40 words, plain text) that connects the "
        "evidence to this gold label by pointing at SPECIFIC values, entities, or axis "
        "ranges visible in the evidence images/captions. " + strictness +
        "Rules: do NOT write the label name at the end; do NOT invent numbers or "
        "entities; cite concrete numbers / named rows / axis limits where relevant.\n"
        "If you cannot find concrete grounds in the evidence for this label, output "
        "exactly: CANNOT_JUSTIFY\n"
        "Output format (one line): RATIONALE: <text>   — or — CANNOT_JUSTIFY")
    return "\n".join(parts)


def gold_messages(rec: dict, row: dict, data_root: Path, max_images: int,
                  attempt: int, fail_note: str = "") -> list:
    content = [{"type": "text", "text": gold_user_text(rec, row, attempt, fail_note)}]
    content += image_parts(rec, data_root, max_images)
    return [{"role": "system", "content": GOLD_SYSTEM},
            {"role": "user", "content": content}]


def parse_gold(text: str):
    """-> ("ok", rationale) | ("cannot_justify", "") | ("bad_output", why)."""
    if re.search(r"CANNOT[_ ]?JUSTIFY", text, re.IGNORECASE):
        return "cannot_justify", ""
    rat = clean_rationale(text)
    wc = n_words(rat)
    if wc > MAX_WORDS:
        return "bad_output", f"overlength({wc}w)"
    if wc < MIN_WORDS:
        return "bad_output", f"too_short({wc}w)"
    return "ok", rat


def gold_call_once(rid: str, row: dict, ctx, args, key, attempt: int,
                   fail_note: str = ""):
    """One Stage-B attempt -> (status, rationale_or_why, raw_text)."""
    rec = ctx["by_id"][rid]
    msgs = gold_messages(rec, row, ctx["data_root"], args.max_images, attempt, fail_note)
    text = g5.call_api(msgs, args.model, args.base, key)
    status, payload = parse_gold(text)
    return status, payload, text


def run_gold(args, ctx):
    out_dir, key = Path(args.out), ctx["key"]
    covered = {(u["record_id"], u["sent_idx"])
               for u in read_jsonl(out_dir / "units_blind.jsonl")}
    terminal = {(r["record_id"], r["sent_idx"])
                for r in read_jsonl(out_dir / "gold_cond.raw.jsonl")
                if r.get("status") in ("ok", "cannot_justify", "gave_up")}
    todo = [(rid, row) for rid in ctx["scope_ids"]
            for row in ctx["scope_rows"][rid]
            if (rid, row["sent_idx"]) not in covered
            and (rid, row["sent_idx"]) not in terminal]
    n_min = sum(1 for _, r in todo if r["target_label"] != "Supported")
    print(f"Stage B gold: uncovered sentences todo={len(todo)} (minority {n_min}); "
          f"blind-covered={len(covered)} resumed={len(terminal)}")
    raw = RawLog(out_dir / "gold_cond.raw.jsonl")

    def work(rid: str, row: dict):
        for attempt in (1, 2):
            t0 = time.time()
            try:
                status, payload, text = gold_call_once(rid, row, ctx, args, key, attempt)
            except Exception as e:  # noqa: BLE001
                raw.write({"record_id": rid, "sent_idx": row["sent_idx"], "attempt": attempt,
                           "status": "api_error", "err": f"{type(e).__name__}: {e}",
                           "dur_s": round(time.time() - t0, 1)})
                continue
            base = {"record_id": rid, "sent_idx": row["sent_idx"], "attempt": attempt,
                    "raw": text, "dur_s": round(time.time() - t0, 1),
                    "target_label": row["target_label"]}
            if status == "ok":
                raw.write({**base, "status": "ok", "rationale": payload})
                return
            if status == "cannot_justify":
                raw.write({**base, "status": "cannot_justify"})
                return
            raw.write({**base, "status": "bad_output", "why": payload})
        raw.write({"record_id": rid, "sent_idx": row["sent_idx"], "status": "gave_up"})

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, rid, row) for rid, row in todo]
        for fut in as_completed(futs):
            fut.result()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(todo)} sentences done")
    raw.close()
    rebuild_units_gold(out_dir, ctx, args.model)


def rebuild_units_gold(out_dir: Path, ctx, model: str):
    units, seen = [], set()
    for row in read_jsonl(out_dir / "gold_cond.raw.jsonl"):
        if row.get("status") != "ok" or "rationale" not in row:
            continue
        rid = row["record_id"]
        if rid not in ctx["scope_rows"]:
            continue
        r = ctx["scope_rows"][rid][row["sent_idx"]]
        u = {"record_id": rid, "sent_idx": row["sent_idx"], "sentence": r["sentence"],
             "gold_types": r["gold_types"], "target_label": r["target_label"],
             "rationale": row["rationale"], "source": "gold_cond", "verified": False,
             "verifier_note": "", "model": model, "ts": row.get("ts"),
             "n_tries": row.get("attempt", 1)}
        k = unit_key(u)
        if k not in seen:
            seen.add(k)
            units.append(u)
    write_units(out_dir / "units_gold.jsonl", units, ctx["split"])
    return units


# ───────────────────────────── Stage C: verify ───────────────────────────────


def verify_messages(rec: dict, u: dict, data_root: Path, max_images: int) -> list:
    txt = (caption_text(rec)
           + "\n\nTARGET SENTENCE (from a claim about this evidence):\n" + u["sentence"]
           + "\n\nRATIONALE under audit:\n" + u["rationale"]
           + "\n\nTask: list every specific value / entity / axis range the RATIONALE "
             "cites, and check EACH against the evidence images and captions above. "
             "The rationale FAILS if it cites a number / entity / trend that is not "
             "actually present, misreads a value, or cites nothing specific at all.\n"
             "Output exactly:\nCHECK: <itemized checks>\nVERDICT: PASS or FAIL\n"
             "REASON: <one line>")
    content = [{"type": "text", "text": txt}] + image_parts(rec, data_root, max_images)
    return [{"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": content}]


def parse_verify(text: str):
    m = re.search(r"VERDICT\s*:?\s*(PASS|FAIL)", text, re.IGNORECASE)
    if not m:
        return None, ""
    reason = ""
    mr = re.search(r"REASON\s*:?\s*(.+)", text)
    if mr:
        reason = mr.group(1).strip().splitlines()[0][:300]
    return m.group(1).upper(), reason


def audit_selection(units: list[dict], frac: float, seed: int) -> set[str]:
    """All minority units + seeded frac of Supported units (deterministic)."""
    sel = {unit_key(u) for u in units if u["target_label"] != "Supported"}
    sup = sorted(unit_key(u) for u in units if u["target_label"] == "Supported")
    rng = random.Random(seed)
    n_pick = max(1, int(round(len(sup) * frac))) if sup else 0
    sel |= set(rng.sample(sup, min(n_pick, len(sup))))
    return sel


def run_verify(args, ctx):
    out_dir, key = Path(args.out), ctx["key"]
    units = read_jsonl(out_dir / "units_blind.jsonl") + read_jsonl(out_dir / "units_gold.jsonl")
    by_key = {}
    for u in units:
        by_key.setdefault(unit_key(u), u)  # dedupe across files
    units = list(by_key.values())
    audit = audit_selection(units, args.audit_supported_frac, args.seed)
    finals = {r["unit_key"]: r for r in read_jsonl(out_dir / "verify.raw.jsonl")
              if r.get("phase") == "final"}
    todo = [by_key[k] for k in sorted(audit) if k not in finals]
    n_min = sum(1 for u in todo if u["target_label"] != "Supported")
    print(f"Stage C verify: units={len(units)} audit={len(audit)} "
          f"(minority {sum(1 for k in audit if by_key[k]['target_label'] != 'Supported')}) "
          f"todo={len(todo)} (minority {n_min}) resumed={len(finals)}")
    raw = RawLog(out_dir / "verify.raw.jsonl")

    def verify_once(u: dict, phase: str):
        rec = ctx["by_id"][u["record_id"]]
        msgs = verify_messages(rec, u, ctx["data_root"], args.max_images)
        for attempt in (1, 2):  # reparse-retry on unparseable auditor output
            text = g5.call_api(msgs, args.model, args.base, key)
            verdict, reason = parse_verify(text)
            raw.write({"unit_key": unit_key(u), "phase": phase, "attempt": attempt,
                       "verdict": verdict, "reason": reason, "raw": text})
            if verdict is not None:
                return verdict, reason
        return "FAIL", "auditor output unparseable"

    def work(u: dict):
        k = unit_key(u)
        try:
            verdict, reason = verify_once(u, "verify")
            if verdict == "PASS":
                raw.write({"unit_key": k, "phase": "final", "outcome": "pass",
                           "note": "audit_pass"})
                return
            if u["target_label"] == "Supported":
                raw.write({"unit_key": k, "phase": "final", "outcome": "dropped_fail",
                           "note": reason})
                return
            # minority FAIL -> inline Stage-B rewrite (1x) -> re-verify
            row = ctx["scope_rows"][u["record_id"]][u["sent_idx"]]
            status, payload, text = gold_call_once(u["record_id"], row, ctx, args, key,
                                                   attempt=2, fail_note=reason)
            raw.write({"unit_key": k, "phase": "rewrite", "status": status, "raw": text})
            if status != "ok":
                raw.write({"unit_key": k, "phase": "final",
                           "outcome": "dropped_cannot_justify" if status == "cannot_justify"
                           else "dropped_fail", "note": f"rewrite {status}"})
                return
            u2 = {**u, "rationale": payload, "source": "gold_cond"}
            verdict2, reason2 = verify_once(u2, "reverify")
            if verdict2 == "PASS":
                raw.write({"unit_key": k, "phase": "final", "outcome": "rewritten_pass",
                           "new_rationale": payload, "note": reason2})
            else:
                raw.write({"unit_key": k, "phase": "final", "outcome": "dropped_fail",
                           "note": f"reverify FAIL: {reason2}"})
        except Exception as e:  # noqa: BLE001
            raw.write({"unit_key": k, "phase": "error", "err": f"{type(e).__name__}: {e}"})

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, u) for u in todo]
        for fut in as_completed(futs):
            fut.result()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(todo)} audits done")
    raw.close()
    merge_final(out_dir, ctx, args)


def merge_final(out_dir: Path, ctx, args):
    """Compose final units.jsonl (+ label_only.jsonl) from derived units + audit finals."""
    units = read_jsonl(out_dir / "units_blind.jsonl") + read_jsonl(out_dir / "units_gold.jsonl")
    by_key = {}
    for u in units:
        by_key.setdefault(unit_key(u), u)
    finals = {r["unit_key"]: r for r in read_jsonl(out_dir / "verify.raw.jsonl")
              if r.get("phase") == "final"}
    merged, dropped = [], {}
    for k, u in by_key.items():
        f = finals.get(k)
        if f is None:
            merged.append({**u, "verified": False, "verifier_note": "not_audited"})
        elif f["outcome"] == "pass":
            merged.append({**u, "verified": True, "verifier_note": "audit_pass"})
        elif f["outcome"] == "rewritten_pass":
            merged.append({**u, "rationale": f["new_rationale"], "source": "gold_cond",
                           "verified": True, "verifier_note": "rewritten_after_fail"})
        else:
            dropped[(u["record_id"], u["sent_idx"])] = f.get("note", f["outcome"])
    covered = {(u["record_id"], u["sent_idx"]) for u in merged}
    cj = {(r["record_id"], r["sent_idx"]): "cannot_justify"
          for r in read_jsonl(out_dir / "gold_cond.raw.jsonl")
          if r.get("status") == "cannot_justify"}
    gu = {(r["record_id"], r["sent_idx"]): "stage_b_gave_up"
          for r in read_jsonl(out_dir / "gold_cond.raw.jsonl")
          if r.get("status") == "gave_up"}
    label_only = []
    for rid in ctx["scope_ids"]:
        for row in ctx["scope_rows"][rid]:
            kk = (rid, row["sent_idx"])
            if kk in covered:
                continue
            reason = dropped.get(kk) or cj.get(kk) or gu.get(kk) or "no_unit"
            label_only.append({**row, "reason": reason})
    write_units(out_dir / "units.jsonl", merged, ctx["split"])
    with open(out_dir / "label_only.jsonl", "w", encoding="utf-8") as f:
        for r in label_only:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_min_lo = sum(1 for r in label_only if r["target_label"] != "Supported")
    print(f"label_only: {len(label_only)} sentences ({n_min_lo} minority) "
          f"-> {out_dir / 'label_only.jsonl'}")


# ───────────────────────────── --stats ───────────────────────────────────────


def stage_throughput(rows: list[dict]) -> str:
    ts = [r["ts"] for r in rows if "ts" in r]
    if len(ts) < 2 or max(ts) == min(ts):
        return "n/a"
    cpm = len(rows) / ((max(ts) - min(ts)) / 60)
    return f"{cpm:.1f} calls/min"


def run_stats(args, ctx):
    out_dir = Path(args.out)
    scope_rows = ctx["scope_rows"]
    all_rows = [r for rid in ctx["scope_ids"] for r in scope_rows[rid]]
    per_class_total = Counter(r["target_label"] for r in all_rows)

    braw = read_jsonl(out_dir / "blind.raw.jsonl")
    graw = read_jsonl(out_dir / "gold_cond.raw.jsonl")
    vraw = read_jsonl(out_dir / "verify.raw.jsonl")
    b_calls = [r for r in braw if r.get("status") != "gave_up" or r.get("raw")]
    g_calls = [r for r in graw if r.get("status") != "gave_up"]
    v_calls = [r for r in vraw if r.get("phase") in ("verify", "reverify", "rewrite")]

    ok_samples = [r for r in braw if r.get("status") == "ok"]
    sus = [su for r in ok_samples for su in r.get("sent_units", [])]
    agree_any = sum(1 for su in sus if su.get("hit_anyof")) / len(sus) if sus else 0
    agree_strict = (sum(1 for su in sus if su.get("hit_target", su.get("accepted")))
                    / len(sus) if sus else 0)
    per_class_strict = Counter()
    for r in ok_samples:
        for su in r.get("sent_units", []):
            if su.get("accepted"):
                row = scope_rows[r["record_id"]][su["sent_idx"]]
                per_class_strict[row["target_label"]] += 1

    g_term = [r for r in graw if r.get("status") in ("ok", "cannot_justify", "gave_up")]
    g_sent = {}
    for r in g_term:
        g_sent[(r["record_id"], r["sent_idx"])] = r["status"]
    n_cj = sum(1 for v in g_sent.values() if v == "cannot_justify")
    cj_rate = n_cj / len(g_sent) if g_sent else 0

    finals = [r for r in vraw if r.get("phase") == "final"]
    first_verify = {}
    for r in vraw:
        if r.get("phase") == "verify" and r.get("verdict") in ("PASS", "FAIL"):
            first_verify.setdefault(r["unit_key"], r["verdict"])
    units_all = read_jsonl(out_dir / "units_blind.jsonl") + read_jsonl(out_dir / "units_gold.jsonl")
    ukey_label = {unit_key(u): u["target_label"] for u in units_all}
    fp_min = [v for k, v in first_verify.items() if ukey_label.get(k) != "Supported"]
    fp_sup = [v for k, v in first_verify.items() if ukey_label.get(k) == "Supported"]
    pass_min = fp_min.count("PASS") / len(fp_min) if fp_min else 0
    pass_sup = fp_sup.count("PASS") / len(fp_sup) if fp_sup else 0
    n_rewr_ok = sum(1 for r in finals if r["outcome"] == "rewritten_pass")

    units = read_jsonl(out_dir / "units.jsonl")
    lo = read_jsonl(out_dir / "label_only.jsonl")
    cov_sents = defaultdict(set)
    for u in units:
        cov_sents[u["target_label"]].add((u["record_id"], u["sent_idx"]))
    wc = [n_words(u["rationale"]) for u in units]
    multi = Counter((u["record_id"], u["sent_idx"]) for u in units)
    n_multi = sum(1 for c in multi.values() if c > 1)

    total_calls = len(b_calls) + len(g_calls) + len(v_calls)
    L = []
    L.append(f"# CoT distill stats — {out_dir}")
    L.append(f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}; model={args.model} "
             f"base={args.base} (report duty: model/version/call counts)_\n")
    L.append(f"## Scope\nrecords={len(ctx['scope_ids'])} sentences={len(all_rows)} | "
             + " ".join(f"{l}:{per_class_total[l]}" for l in LABELS) + "\n")
    L.append("## Calls")
    L.append(f"- Stage A blind: {len(b_calls)} calls ({stage_throughput(b_calls)}); "
             f"ok={len(ok_samples)} misaligned={sum(1 for r in braw if r.get('status') == 'misaligned')} "
             f"api_error={sum(1 for r in braw if r.get('status') == 'api_error')} "
             f"gave_up={sum(1 for r in braw if r.get('status') == 'gave_up')}")
    L.append(f"- Stage B gold: {len(g_calls)} calls ({stage_throughput(g_calls)})")
    L.append(f"- Stage C verify: {len(v_calls)} calls ({stage_throughput(v_calls)})")
    L.append(f"- TOTAL: {total_calls} calls\n")
    L.append("## Gate metrics (plan §3)")
    L.append(f"- Stage A sentence agreement (any-of gold_types): **{agree_any:.1%}** "
             f"(strict ==target: {agree_strict:.1%}) over {len(sus)} "
             f"sentence-verdicts [gate >=55%]")
    L.append("  - accepted units per class: "
             + " ".join(f"{l}:{per_class_strict[l]}" for l in LABELS))
    L.append(f"- Stage B CANNOT_JUSTIFY rate: **{cj_rate:.1%}** ({n_cj}/{len(g_sent)}) "
             f"[gate <15%]")
    L.append(f"- Stage C first-pass PASS rate: minority **{pass_min:.1%}** "
             f"({len(fp_min)} audited) [gate >=80%]; Supported sample {pass_sup:.1%} "
             f"({len(fp_sup)} audited)"
             + (" ⚠️ Supported FAIL>10% — expand audit" if fp_sup and (1 - pass_sup) > 0.10 else ""))
    L.append(f"- rewrites recovered: {n_rewr_ok}; dropped after audit: "
             f"{sum(1 for r in finals if r['outcome'].startswith('dropped'))}\n")
    L.append("## Teaching units (final units.jsonl)")
    L.append(f"- units={len(units)} (verified={sum(1 for u in units if u['verified'])}, "
             f"multi-rationale sentences={n_multi}); avg rationale words="
             f"{(sum(wc) / len(wc)):.1f}" if wc else "- units=0")
    L.append("- per-class sentence coverage (>=1 unit / total):")
    for l in LABELS:
        tot = per_class_total[l]
        cov = len(cov_sents[l])
        L.append(f"  - {l}: {cov}/{tot} ({cov / tot:.1%})" if tot else f"  - {l}: 0/0")
    n_lo_min = sum(1 for r in lo if r["target_label"] != "Supported")
    L.append(f"- label_only: {len(lo)} sentences ({n_lo_min} minority; "
             f"{(len(lo) / len(all_rows)):.1%} of scope)\n" if all_rows else "")
    L.append("## Stage A acceptance breakdown (per parsed sample sentence)")
    rc = Counter(su.get("reason") for su in sus if not su.get("accepted"))
    L.append(f"- accepted={sum(1 for su in sus if su['accepted'])}; rejects: "
             + " ".join(f"{k}:{v}" for k, v in rc.most_common()))
    txt = "\n".join(L) + "\n"
    (out_dir / "stats.md").write_text(txt, encoding="utf-8")
    print(txt)


# ───────────────────────────── main ──────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["blind", "gold", "verify"])
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--split-json", default="data/devbench/split.json")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "../NLPCC-2026-Task10-Science"))
    ap.add_argument("--out", default="data/cot")
    ap.add_argument("--model", default=os.environ.get("GPT5_MODEL", "gpt-5.5"))
    ap.add_argument("--base", default=os.environ.get("GPT5_BASE", "https://aiapis.help/v1"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-images", type=int, default=6)
    ap.add_argument("--k", type=int, default=2, help="blind samples per record")
    ap.add_argument("--max-tries", type=int, default=3, help="blind resamples on misalignment")
    ap.add_argument("--audit-supported-frac", type=float, default=0.10)
    ap.add_argument("--records-sample", type=int, default=0, help="pilot: sample N train' records")
    ap.add_argument("--min-minority-sents", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap scoped records (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="print stage prompts for 1 record, no API")
    args = ap.parse_args(argv)
    if not args.stage and not args.stats and not args.dry_run:
        ap.error("need --stage blind|gold|verify, --stats, or --dry-run")

    data_root = Path(args.data_root)
    split = load_split(Path(args.split_json))
    records = load_traindev(data_root)
    for fid in FEWSHOT_SOURCE_IDS:  # red line 2: few-shot provenance must be train'
        assert split.get(fid) == "train", f"few-shot source {fid} not in train'!"

    scope_ids = select_scope(args, records, split)
    if args.limit:
        scope_ids = scope_ids[: args.limit]
    # RED LINE assert #1 (entry): the whole scope is train'; dev' is never loaded.
    bad = [r for r in scope_ids if split.get(r) != "train"]
    assert not bad, f"scope contains non-train' ids: {bad[:5]}"

    by_id = dict(records)
    freq = label_freq_train(records, split)
    scope_rows = {rid: sentence_rows(rid, by_id[rid], freq) for rid in scope_ids}
    ctx = {"split": split, "by_id": by_id, "scope_ids": scope_ids,
           "scope_rows": scope_rows, "data_root": data_root, "key": None}

    if args.dry_run:
        rid = scope_ids[0]
        rec = by_id[rid]
        row = next((r for r in scope_rows[rid] if r["target_label"] != "Supported"),
                   scope_rows[rid][0])
        msgs, n = g5.build_messages(rec, data_root, args.max_images)
        print(f"=== Stage A user text ({rid}, {n} sents) ===\n"
              + msgs[1]["content"][0]["text"][:1200])
        print(f"\n=== Stage B user text (sent {row['sent_idx']}, {row['target_label']}) ===\n"
              + gold_user_text(rec, row, 1)[:1200])
        u = {"record_id": rid, "sent_idx": row["sent_idx"], "sentence": row["sentence"],
             "target_label": row["target_label"], "source": "blind",
             "rationale": "claim cites X; evidence table lists Y=1.0"}
        print("\n=== Stage C user text ===\n"
              + verify_messages(rec, u, data_root, 0)[1]["content"][0]["text"][:1200])
        return 0

    if args.stats:
        run_stats(args, ctx)
        return 0

    key = os.environ.get("GPT5_KEY") or os.environ.get("K")
    if not key:
        print("ERROR: set GPT5_KEY env", file=sys.stderr)
        return 2
    ctx["key"] = key
    {"blind": run_blind, "gold": run_gold, "verify": run_verify}[args.stage](args, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
