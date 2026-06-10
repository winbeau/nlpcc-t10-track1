#!/usr/bin/env python3
"""Sync data artifacts with the unified HF DATASET repo (winbeau/nlpcc2026-task10, private).

Repo layout (everything under track1/ — see notes/cot_distill_plan.md §1.5):
  track1/
  ├── raw/                  # official track-1 jsonls + manifest (NO images — pull those
  │                         #   from the official git-lfs repo on each machine)
  └── processed/
      ├── devbench/         # split.json / dev_gold*.jsonl / train_sft.jsonl / dev_sft.jsonl
      └── cot/              # CoT teaching units (units.jsonl / *.raw.jsonl / stats.md / pilot/)

Token: env HF_TOKEN or ~/.cache/huggingface/token. Runs anywhere (processing box, GPU box,
local) — no project deps needed:
  uv run --no-project --with huggingface_hub python scripts/hf_data.py push --what raw devbench
  uv run --no-project --with huggingface_hub python scripts/hf_data.py push --what cot --dry-run
  uv run --no-project --with huggingface_hub python scripts/hf_data.py pull --what devbench cot
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

RID = "winbeau/nlpcc2026-task10"
REPO_TYPE = "dataset"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "../NLPCC-2026-Task10-Science"))

# what -> (local_dir, hf_prefix, filename globs)
WHAT = {
    "raw": (DATA_ROOT / "data", "track1/raw",
            ["traindev-track-1.jsonl", "testp1-track-1.jsonl", "testp1-manifest.json"]),
    "devbench": (Path("data/devbench"), "track1/processed/devbench", ["*"]),
    "cot": (Path("data/cot"), "track1/processed/cot", ["**/*"]),
}

README = """# nlpcc2026-task10 — unified data repo
Private archive for NLPCC-2026 Shared Task 10. Competition use only — do not redistribute.
All content lives under `track1/`: `raw/` (official track-1 jsonls; images stay on the
official git-lfs repo) and `processed/{devbench,cot}/` (derived artifacts).
Sync tool: `scripts/hf_data.py` in the solution repo.
"""


def is_lfs_pointer(p: Path) -> bool:
    """Official zips may be unpulled git-lfs pointers (~130 bytes) — never upload those."""
    return p.stat().st_size < 200 and p.read_bytes().startswith(b"version https://git-lfs")


def gather(what: str) -> list[tuple[Path, str]]:
    local, prefix, globs = WHAT[what]
    if not local.is_dir():
        print(f"  ⚠️ {what}: local dir {local} missing — skipped")
        return []
    pairs = []
    for g in globs:
        for p in sorted(local.glob(g)):
            if not p.is_file():
                continue
            if p.stat().st_size == 0:
                print(f"  ⚠️ skip empty {p}")
                continue
            if is_lfs_pointer(p):
                print(f"  ⚠️ skip lfs-pointer {p}")
                continue
            pairs.append((p, f"{prefix}/{p.relative_to(local)}"))
    return pairs


def push(whats: list[str], dry: bool) -> int:
    api = HfApi()
    api.create_repo(RID, repo_type=REPO_TYPE, private=True, exist_ok=True)
    ops = []
    existing = set(api.list_repo_files(RID, repo_type=REPO_TYPE))
    if "README.md" not in existing:
        Path("/tmp/_hfdata_readme.md").write_text(README)
        ops.append(CommitOperationAdd("README.md", "/tmp/_hfdata_readme.md"))
    for w in whats:
        pairs = gather(w)
        for p, rp in pairs:
            ops.append(CommitOperationAdd(rp, str(p)))
        print(f"  {w}: {len(pairs)} files ({sum(p.stat().st_size for p, _ in pairs)/1e6:.1f} MB)")
    print(f"\n{'DRY-RUN: would commit' if dry else 'committing'} {len(ops)} files -> {RID} ({REPO_TYPE})")
    if not dry and ops:
        api.create_commit(RID, ops, repo_type=REPO_TYPE,
                          commit_message=f"sync {'+'.join(whats)} via hf_data.py")
        print(f"DONE -> https://huggingface.co/datasets/{RID}/tree/main/track1")
    return 0


def pull(whats: list[str], dry: bool) -> int:
    api = HfApi()
    files = api.list_repo_files(RID, repo_type=REPO_TYPE)
    n = 0
    for w in whats:
        local, prefix, _ = WHAT[w]
        for rp in files:
            if not rp.startswith(prefix + "/"):
                continue
            dst = local / Path(rp).relative_to(prefix)
            print(f"  {rp} -> {dst}")
            n += 1
            if dry:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(RID, rp, repo_type=REPO_TYPE)
            shutil.copyfile(cached, dst)
    print(f"{'DRY-RUN: would fetch' if dry else 'fetched'} {n} files")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["push", "pull"])
    ap.add_argument("--what", nargs="+", choices=list(WHAT), required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.mode == "push" and "raw" in args.what and not DATA_ROOT.is_dir():
        print(f"ERROR: DATA_ROOT {DATA_ROOT} not found (set env DATA_ROOT)", file=sys.stderr)
        return 2
    return push(args.what, args.dry_run) if args.mode == "push" else pull(args.what, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
