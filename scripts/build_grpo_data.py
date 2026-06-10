#!/usr/bin/env python3
"""Build the GRPO prompt set from the E1 CoT training set — grpo_plan.md §3.

GRPO trains on PROMPTS (no assistant target); reward comes from scripts/grpo_reward.py reading a
`gold_types` column. We take E1's resampled sentence-level set (os3.0/ds0.6 train', the SAME
prompt distribution the policy was SFT'd on — §3 red line: the pseudo-prior is fixed HERE), strip
the assistant turn, and attach gold_types.

gold_types source: the E1 target label (the pick_rarest single label) -> a 1-element gold list.
This credits exactly the training target. For the ~3% multi-label sentences the official "any of
types" is looser, but the rarest IS the chosen target, so single-element gold is the right RL
signal here (and avoids re-deriving full types). Documented simplification.

Usage:
  uv run python scripts/build_grpo_data.py --in data/cot/e1/train_sft.jsonl \
      --out data/cot/grpo/prompts.jsonl
Output rows: {"messages": [system, user], "gold_types": ["<label>"], "channel": "<pid>"}
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def last_line_label(assistant_content: str) -> str | None:
    lines = [l for l in assistant_content.strip().splitlines() if l.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1]).get("label")
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/cot/e1/train_sft.jsonl")
    ap.add_argument("--out", default="data/cot/grpo/prompts.jsonl")
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dist: Counter = Counter()
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            msgs = r.get("messages", [])
            if len(msgs) < 3:
                continue
            label = last_line_label(msgs[2].get("content", ""))
            if label is None:
                continue
            prompt = [m for m in msgs if m.get("role") != "assistant"]
            rec = {"messages": prompt, "gold_types": [label], "channel": r.get("channel", "")}
            if "images" in r:
                rec["images"] = r["images"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dist[label] += 1
            n += 1
    print(f"wrote {n} GRPO prompts -> {out_path}")
    for lbl, c in dist.most_common():
        print(f"  {lbl}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
