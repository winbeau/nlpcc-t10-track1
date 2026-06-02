#!/usr/bin/env python3
"""Validate a Track 1 submission and package it into a Codabench-ready .zip.

OFFICIAL Codabench spec (competition 16666): upload a FLAT .zip containing one or both
prediction files AT THE TOP LEVEL of the archive (do NOT zip a folder):
    track1_pred.jsonl   ->  {"id": ..., "labels": [...]}
    track2_pred.jsonl   ->  {"id": ..., "label": ..., "evidence_para_ids": [...]}
A missing track file is skipped (not scored as failed), so a Track-1-only zip is fine.

  --inner-name  filename placed at the ROOT of the zip. Default 'track1_pred.jsonl' (the
                official Track 1 name). zipfile arcname keeps it flat (top level, no folder).

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
    ap.add_argument("--inner-name", default="track1_pred.jsonl",
                    help="filename inside the zip (official Track 1 name: track1_pred.jsonl)")
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
    print(f"Official spec: FLAT zip, top-level 'track1_pred.jsonl' (this file is {args.inner_name!r}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
