#!/usr/bin/env python3
"""Run a frontier API model (gpt-5.5 via OpenAI-compatible proxy) on Track-1.

Per RECORD (joint): feed the evidence image(s) + caption(s) + full claim + the numbered
sentences; the model reasons (CoT) then emits one label per sentence; we parse -> aggregate
-> {id, labels} submission jsonl. Designed to run on H200 (images unzipped there).

Pipeline validated locally 2026-06-04: gpt-5.5 text+vision OK via https://aiapis.help (needs a
browser User-Agent to pass Cloudflare 1010). Rules允许商业 API(指南 Q:GPT-5/Claude via API = Yes,
须报告 model/version/cost)。

Env: GPT5_KEY (required), GPT5_BASE (default https://aiapis.help/v1), GPT5_MODEL (default gpt-5.5).
Usage:
  GPT5_KEY=sk-... python scripts/gpt5_infer.py --records data/devbench/dev_gold.jsonl \
      --data-root "$DATA_ROOT" --out outputs/gpt5_dev_pred.jsonl --workers 6 --limit 30
  GPT5_KEY=sk-... python scripts/gpt5_infer.py --split testp1 --data-root "$DATA_ROOT" \
      --out submissions/testp1_gpt5_raw.jsonl --workers 6
Then score (dev): uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 \
      --gold data/devbench/dev_gold.jsonl --pred <out> --match id
"""
from __future__ import annotations
import argparse, base64, json, os, re, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LABELS = ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
          "Scope Overgeneralization", "Contradiction"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

SYSTEM = """You are a meticulous verifier of scientific-claim faithfulness. You are given figure/table EVIDENCE (images + captions) and a CLAIM split into numbered sentences. For EACH sentence, assign exactly ONE label, judging the sentence ONLY against the provided evidence:

- Supported: the sentence is fully supported by the evidence.
- Unsupported Causal Mechanistic: it asserts a causal / mechanistic explanation (a "why"/"because"/"due to" / mechanism) that the evidence does not establish.
- Unsupported Entity: it mentions a dataset / metric / model variant / baseline / method / scientific entity that does NOT appear in the evidence.
- Scope Overgeneralization: it extrapolates the conclusion BEYOND the scope the evidence supports (e.g. "all"/"always"/"in general"/other datasets/settings the evidence does not cover).
- Contradiction: it directly conflicts with the evidence (e.g. a number, direction, or comparison that disagrees with the figure/table).

CRITICAL guidance:
- READ THE NUMBERS in the figures/tables carefully (axes, bars, table cells). Many sentences hinge on a precise value or comparison that is only in the image, not the caption. Check the claimed number/direction against the evidence.
- Non-Supported labels are COMMON here. Do NOT default to Supported. Label a sentence Supported only if the evidence genuinely and fully backs it; otherwise pick the matching error type. If multiple apply, pick the single most specific/severe.
- Judge each sentence in the context of the whole claim, but the label is about THAT sentence.

You MUST reason explicitly before answering. For EACH sentence i, write ONE line in this form:
  i) claim asserts: <what the sentence claims>; evidence shows: <cite the SPECIFIC value/entity/scope from the figure/table, or state it is ABSENT>; verdict: <label>
Actually read the numbers off the figures/tables to fill "evidence shows" — do not guess.
Only AFTER reasoning through ALL N sentences, output the final answer as ONE line:
FINAL: <label#1> ||| <label#2> ||| ... ||| <label#N>
using the exact label strings above, one per sentence, in order, N total."""


def canon(s: str) -> str:
    """Map a free-text label to one of the 5 canonical strings (fallback Supported)."""
    t = s.strip().lower()
    if not t:
        return "Supported"
    if "contradict" in t:
        return "Contradiction"
    if "causal" in t or "mechanist" in t:
        return "Unsupported Causal Mechanistic"
    if "entity" in t:
        return "Unsupported Entity"
    if "scope" in t or "overgeneral" in t or "over-general" in t:
        return "Scope Overgeneralization"
    if "support" in t:
        return "Supported"
    return "Supported"


def parse_labels(text: str, n: int) -> list[str]:
    """Extract n labels. Prefer the FINAL: line; else scan label mentions in order."""
    labs: list[str] = []
    m = re.search(r"FINAL\s*:?\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        seg = m.group(1).strip().splitlines()[0] if "|||" in m.group(1) else m.group(1)
        if "|||" in seg:
            labs = [canon(x) for x in seg.split("|||")]
    if len(labs) != n:
        # fallback: find canonical labels in order of appearance
        found = re.findall(r"(Contradiction|Unsupported Causal Mechanistic|Unsupported Entity|Scope Overgeneralization|Supported)", text)
        labs = [canon(x) for x in found]
    if len(labs) < n:
        labs += ["Supported"] * (n - len(labs))
    return labs[:n]


def b64_image(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    ext = path.suffix.lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64," + base64.b64encode(data).decode()


def build_messages(rec: dict, data_root: Path, max_images: int) -> tuple[list, int]:
    sents = [sl["sentence"] for sl in rec["sentence_label"]]
    parts = ["EVIDENCE captions:"]
    for j, ev in enumerate(rec.get("evidence_bundle", [])):
        cap = ev.get("image_caption") or ev.get("table_caption") or ev.get("caption") or []
        if isinstance(cap, list):
            cap = " ".join(map(str, cap))
        parts.append(f"[evidence {j + 1} | {ev.get('type', '?')}]: {cap}")
    parts.append("\nFULL CLAIM:\n" + rec.get("claim_text", ""))
    parts.append(f"\nSENTENCES to label (assign exactly {len(sents)} labels, in order):")
    for i, s in enumerate(sents):
        parts.append(f"{i + 1}. {s}")
    content: list = [{"type": "text", "text": "\n".join(parts)}]
    imgs = 0
    for ev in rec.get("evidence_bundle", []):
        ip = ev.get("img_path")
        if ip and imgs < max_images:
            # Official layout: <data_root>/data/images/<sha>.jpg (img_path is "images/<sha>.jpg",
            # missing the 'data/' segment). Fall back to <data_root>/<ip> and bare <ip>.
            cand = data_root / "data" / ip
            if not cand.exists():
                cand = data_root / ip
            if not cand.exists():
                cand = Path(ip)
            uri = b64_image(cand)
            if uri:
                content.append({"type": "image_url", "image_url": {"url": uri}})
                imgs += 1
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}], len(sents)


def call_api(messages: list, model: str, base: str, key: str, retries: int = 4) -> str:
    payload = {"model": model, "messages": messages}
    eff = os.environ.get("GPT5_REASONING_EFFORT")
    if eff:
        payload["reasoning_effort"] = eff  # try to enable deep reasoning (proxy may ignore)
    body = json.dumps(payload).encode()
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(
                base.rstrip("/") + "/chat/completions", data=body,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": UA})
            r = json.load(urllib.request.urlopen(req, timeout=240))
            return r["choices"][0]["message"]["content"] or ""
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (a + 1))
    raise last


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", help="path to a {id,claim_text,evidence_bundle,sentence_label} jsonl")
    ap.add_argument("--split", choices=["testp1"], help="shorthand: $DATA_ROOT/data/testp1-track-1.jsonl")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "../NLPCC-2026-Task10-Science"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.environ.get("GPT5_MODEL", "gpt-5.5"))
    ap.add_argument("--base", default=os.environ.get("GPT5_BASE", "https://aiapis.help/v1"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-images", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build+print prompt for 1 record, no API/images")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root)
    recs_path = args.records or (data_root / "data" / "testp1-track-1.jsonl")
    records = [json.loads(l) for l in open(recs_path, encoding="utf-8") if l.strip()]
    if args.limit:
        records = records[: args.limit]

    if args.dry_run:
        msgs, n = build_messages(records[0], data_root, args.max_images)
        txt = msgs[1]["content"][0]["text"]
        print(f"id={records[0]['id']} n_sentences={n} n_images_in_content={sum(1 for c in msgs[1]['content'] if c['type']=='image_url')}")
        print("--- USER TEXT ---\n" + txt[:1600])
        print("\n--- parse_labels selftest ---")
        print(parse_labels("reasoning...\nFINAL: Supported ||| Contradiction ||| Scope Overgeneralization", 3))
        return 0

    key = os.environ.get("GPT5_KEY") or os.environ.get("K")
    if not key:
        print("ERROR: set GPT5_KEY env", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():  # resume
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
    todo = [r for r in records if r["id"] not in done]
    print(f"records={len(records)} done={len(done)} todo={len(todo)} model={args.model} workers={args.workers}")

    lock = __import__("threading").Lock()
    fout = open(out_path, "a", encoding="utf-8")
    fraw = open(out_path.with_suffix(".raw.jsonl"), "a", encoding="utf-8")  # full model output (reasoning) for inspection
    n_done = [0]; n_fb = [0]

    def work(rec):
        msgs, n = build_messages(rec, data_root, args.max_images)
        out = ""
        try:
            out = call_api(msgs, args.model, args.base, key)
            labs = parse_labels(out, n)
        except Exception as e:  # noqa: BLE001
            labs = ["Supported"] * n
            with lock:
                n_fb[0] += 1
            print(f"[fallback] {rec['id']}: {type(e).__name__} {e}", file=sys.stderr)
        with lock:
            fraw.write(json.dumps({"id": rec["id"], "labels": labs, "raw": out}, ensure_ascii=False) + "\n")
            fraw.flush()
        return {"id": rec["id"], "labels": labs}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in as_completed([ex.submit(work, r) for r in todo]):
            r = res.result()
            with lock:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                fout.flush()
                n_done[0] += 1
                if n_done[0] % 25 == 0:
                    print(f"  {n_done[0]}/{len(todo)} done (fallback {n_fb[0]})")
    fout.close()
    fraw.close()
    print(f"### GPT5 INFER DONE: {out_path} ({len(done)+n_done[0]} records, {n_fb[0]} fallbacks); raw -> {out_path.with_suffix('.raw.jsonl')} ###")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
