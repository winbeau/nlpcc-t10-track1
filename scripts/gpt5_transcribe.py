#!/usr/bin/env python3
"""Use GPT-5.5 as a VISION/OCR reader (NOT a classifier): transcribe each evidence image into
exact structured numeric text, then write an AUGMENTED records jsonl whose evidence captions
carry the transcription. The Qwen-8B classifier then re-infers on the digit-grounded evidence.

Rationale: the gpt5 probe showed GPT-5.5 is a WEAK Track-1 classifier (over-flags, ignores label
conventions) but its READING of dense tables is strong; Qwen-8B is convention-aligned but misreads
numbers. Split the labor: GPT-5.5 reads -> Qwen classifies. Reuses gpt5_infer.py's OpenAI-compat call.

Env: GPT5_KEY (required), GPT5_BASE (default https://aiapis.help/v1), GPT5_MODEL (default gpt-5.5).
Out: --out augmented records jsonl (evidence captions += transcription); --cache img->text (resume).
"""
from __future__ import annotations
import argparse, base64, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
SYSTEM = "You are a precise scientific figure/table transcriber. You extract data exactly; you never interpret, judge, or add commentary."
PROMPT = ("Transcribe this scientific figure or table into compact structured text for a downstream "
          "fact-checker. Include: (1) the type (table / line plot / bar chart / etc.); (2) all axis "
          "labels and every column & row header; (3) EVERY numeric value, each paired with which "
          "method/model/dataset and which metric it belongs to; (4) for each comparison, which entry "
          "is highest/lowest/best. Be exhaustive and EXACT with numbers and signs. Output only the "
          "extracted data, no interpretation.")


def b64_image(path: Path):
    try:
        data = path.read_bytes()
    except Exception:
        return None
    ext = path.suffix.lstrip(".").lower() or "jpeg"
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64," + base64.b64encode(data).decode()


def call_api(messages, model, base, key, retries=4):
    body = json.dumps({"model": model, "messages": messages}).encode()
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


def img_abspath(data_root: Path, ip: str) -> Path:
    cand = data_root / "data" / ip
    if cand.exists():
        return cand
    cand2 = data_root / ip
    return cand2 if cand2.exists() else Path(ip)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="testp1-track-1.jsonl (id/claim_text/evidence_bundle/sentence_label)")
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "../NLPCC-2026-Task10-Science"))
    ap.add_argument("--out", required=True, help="augmented records jsonl (captions += transcription)")
    ap.add_argument("--cache", required=True, help="img_path -> transcription jsonl (resume)")
    ap.add_argument("--model", default=os.environ.get("GPT5_MODEL", "gpt-5.5"))
    ap.add_argument("--base", default=os.environ.get("GPT5_BASE", "https://aiapis.help/v1"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    key = os.environ.get("GPT5_KEY")
    if not key:
        print("ERROR: set GPT5_KEY", file=sys.stderr)
        return 2
    data_root = Path(a.data_root)
    records = [json.loads(l) for l in open(a.records, encoding="utf-8") if l.strip()]
    if a.limit:
        records = records[: a.limit]

    # unique images
    imgs = []
    seen = set()
    for r in records:
        for ev in r.get("evidence_bundle", []):
            ip = ev.get("img_path", "")
            if ip and ip not in seen:
                seen.add(ip); imgs.append(ip)

    # resume from cache
    cache = {}
    cpath = Path(a.cache)
    if cpath.exists():
        for l in open(cpath, encoding="utf-8"):
            try:
                d = json.loads(l); cache[d["img_path"]] = d["text"]
            except Exception:
                pass
    todo = [ip for ip in imgs if ip not in cache]
    print(f"records={len(records)} unique_images={len(imgs)} cached={len(cache)} todo={len(todo)} model={a.model}", flush=True)

    lock = __import__("threading").Lock()
    cf = open(cpath, "a", encoding="utf-8")
    n = [0]

    def work(ip):
        uri = b64_image(img_abspath(data_root, ip))
        if not uri:
            return ip, ""
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": uri}}]}]
        try:
            txt = call_api(msgs, a.model, a.base, key).strip()
        except Exception as e:  # noqa: BLE001
            txt = ""
            print(f"[fail] {ip}: {type(e).__name__}", file=sys.stderr)
        return ip, txt

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(work, ip) for ip in todo]):
            ip, txt = fut.result()
            with lock:
                cache[ip] = txt
                cf.write(json.dumps({"img_path": ip, "text": txt}, ensure_ascii=False) + "\n"); cf.flush()
                n[0] += 1
                if n[0] % 25 == 0:
                    print(f"  transcribed {n[0]}/{len(todo)}", flush=True)
    cf.close()

    # write augmented records: append transcription into each evidence's caption
    nf = 0
    with open(a.out, "w", encoding="utf-8") as g:
        for r in records:
            for ev in r.get("evidence_bundle", []):
                ip = ev.get("img_path", "")
                tx = cache.get(ip, "")
                if not tx:
                    continue
                key_cap = "image_caption" if "image_caption" in ev or ev.get("type") == "image" else "table_caption"
                cap = ev.get(key_cap) or ev.get("image_caption") or ev.get("table_caption") or []
                if not isinstance(cap, list):
                    cap = [str(cap)]
                ev[key_cap] = list(cap) + ["[GPT5-VISION TRANSCRIPTION] " + tx]
                nf += 1
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"### TRANSCRIBE DONE: {a.out} ({len(records)} records, {nf} evidence augmented); cache {a.cache} ###", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
