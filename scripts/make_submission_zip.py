#!/usr/bin/env python3
"""Validate a Track 1 submission and package it into a Codabench-ready .zip.

Codabench's "result submission" wants a .zip; the platform unzips it and feeds the prediction
file to the official scorer (offline_eval/evaluate.py, matched by `id`). The official repo does
NOT state the exact filename expected INSIDE the zip, so it is configurable here.

  --inner-name  filename placed at the ROOT of the zip. Default 'testp1-track-1.jsonl'
                (mirrors the released input file name — the most common convention). If
                Codabench rejects it, check the competition's submission page / starting kit
                for the expected name (e.g. 'track1.jsonl' or 'predictions.jsonl') and re-run.

Always validates first (scripts/validate_submission.py); refuses to zip an invalid file.
Pure stdlib (uses zipfile — no `zip` binary needed).

Usage:
  python scripts/make_submission_zip.py \
      --sub submissions/testp1_b5_l0.5_submission.jsonl \
      --ref "$DATA_ROOT/data/testp1-track-1.jsonl" \
      [--inner-name testp1-track-1.jsonl] [--out submissions/<name>.zip]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate + zip a Track 1 submission for Codabench.")
    ap.add_argument("--sub", required=True, help="submission JSONL ({id, labels} per line)")
    ap.add_argument("--ref", required=True, help="reference JSONL (testp1-track-1.jsonl)")
    ap.add_argument("--inner-name", default="testp1-track-1.jsonl",
                    help="filename inside the zip (default mirrors the input file name)")
    ap.add_argument("--out", default=None, help="output .zip path (default: <sub>.zip)")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    sub = Path(args.sub)
    if not sub.exists():
        print(f"[ABORT] submission not found: {sub}")
        return 1

    print("=== VALIDATE (scripts/validate_submission.py) ===")
    rc = subprocess.run(
        [sys.executable, str(here / "validate_submission.py"), "--sub", str(sub), "--ref", args.ref]
    ).returncode
    if rc != 0:
        print("\n[ABORT] validation FAILED — not packaging the zip.")
        return rc

    out = Path(args.out) if args.out else sub.with_suffix(".zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub, arcname=args.inner_name)

    print(f"\n=== PACKAGED ===")
    print(f"[OK] {out}")
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"  contains: {info.filename}  ({info.file_size} bytes)")
    print(f"\nUpload {out} to Codabench competition 16666 (Track 1).")
    print(f"NOTE: inner filename = {args.inner_name!r}. The repo doesn't specify the exact name;")
    print(f"      if Codabench rejects it, re-run with --inner-name <expected> (e.g. track1.jsonl).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
