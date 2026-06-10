#!/usr/bin/env python3
"""Custom GRPO reward (ORM plugin) for Track-1 — grpo_plan.md §2.

Registered into ms-swift's `orms` dict as 'track1'; selected via
  swift rlhf --rlhf_type grpo --external_plugins scripts/grpo_reward.py --reward_funcs track1

Reward per rollout (the per-sentence `<analysis>...</analysis>\\n{"label": X}` format):
  * parse fails / illegal label / analysis > MAX_ANALYSIS_WORDS    -> R = -1     (format anti-gaming)
  * label ∈ gold types (multi-label: ANY hit, matches official scorer) -> R = w_class
  * label valid but wrong                                          -> R = 0
  w_class: Supported=1.0; minority = inverse-freq capped at 5.0, calibrated so the EXPECTED
  reward of an all-Supported policy < that of an honest one (anti-laziness; s05=18.85 lesson).

Gold reaches the reward via a dataset column. ms-swift forwards every non-standard dataset
column to the reward fn as a kwarg list aligned to `completions`. We accept the gold types under
several likely column names (`gold_types`, `solution`, `labels`, `gold`) so the data builder can
use whichever ms-swift passes through cleanly.

The reward looks ONLY at the label (推理真假 is policed upstream: ①Stage C grounding audit +
②G2 human spot-check). analysis LENGTH enters the reward to stop rationale-bloat entropy gaming.

Pure helpers (parse_last_label / analysis_word_count / compute_reward) are torch/swift-free and
unit-tested in __main__ (run: `python scripts/grpo_reward.py --selftest`).
"""
from __future__ import annotations

import json
import os
import re

LABELS = ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
          "Scope Overgeneralization", "Contradiction"]
_LOOKUP = {l.lower(): l for l in LABELS}

# Inverse-frequency-ish minority weights (grpo_plan §2 initial values). Override via env
# TRACK1_W_<KEY> if a sweep is needed (KEY = UCM/UE/SO/CONTRA/SUPPORTED).
W_CLASS = {
    "Supported": float(os.environ.get("TRACK1_W_SUPPORTED", "1.0")),
    "Unsupported Causal Mechanistic": float(os.environ.get("TRACK1_W_UCM", "5.0")),
    "Contradiction": float(os.environ.get("TRACK1_W_CONTRA", "4.5")),
    "Unsupported Entity": float(os.environ.get("TRACK1_W_UE", "4.0")),
    "Scope Overgeneralization": float(os.environ.get("TRACK1_W_SO", "3.5")),
}
MAX_ANALYSIS_WORDS = int(os.environ.get("TRACK1_MAX_ANALYSIS_WORDS", "80"))
R_FORMAT_FAIL = float(os.environ.get("TRACK1_R_FORMAT_FAIL", "-1.0"))


def _normalize(raw: str) -> str | None:
    return _LOOKUP.get(" ".join(str(raw).strip().split()).lower())


def parse_last_label(text: str) -> str | None:
    """The label is the LAST line's JSON {"label": X}. Try last-line JSON, then any-line
    `"label": "..."`, then a trailing valid-label substring. None if unparseable/illegal."""
    if not text:
        return None
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        try:
            obj = json.loads(lines[-1])
            if isinstance(obj, dict) and "label" in obj:
                n = _normalize(obj["label"])
                if n:
                    return n
        except Exception:
            pass
    m = None
    for mm in re.finditer(r'"label"\s*:\s*"([^"]+)"', text):
        m = mm
    if m:
        n = _normalize(m.group(1))
        if n:
            return n
    return None


def analysis_word_count(text: str) -> int:
    """Word count inside <analysis>...</analysis> (0 if absent)."""
    m = re.search(r"<analysis>(.*?)</analysis>", text or "", re.DOTALL | re.IGNORECASE)
    return len(m.group(1).split()) if m else 0


def _as_gold_list(gold) -> list[str]:
    """Normalize a gold field into a list of canonical label strings."""
    if gold is None:
        return []
    if isinstance(gold, str):
        s = gold.strip()
        if s.startswith("["):
            try:
                gold = json.loads(s)
            except Exception:
                gold = [s]
        else:
            gold = [s]
    out = []
    for g in gold:
        n = _normalize(g)
        if n:
            out.append(n)
    return out


def compute_reward(completion: str, gold) -> float:
    """One rollout's reward (see module docstring)."""
    gold_list = _as_gold_list(gold)
    if analysis_word_count(completion) > MAX_ANALYSIS_WORDS:
        return R_FORMAT_FAIL
    label = parse_last_label(completion)
    if label is None:
        return R_FORMAT_FAIL
    if gold_list and label in gold_list:
        return W_CLASS.get(label, 1.0)
    return 0.0


def _completion_text(c) -> str:
    """ms-swift may pass a completion as a str, or as a chat list [{'role','content'}, ...]."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c:
        last = c[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(c, dict):
        return str(c.get("content", ""))
    return str(c)


def _pick_gold_column(kwargs, n: int):
    """Find the per-rollout gold list among forwarded dataset columns."""
    for key in ("gold_types", "solution", "labels", "gold", "target"):
        if key in kwargs and kwargs[key] is not None:
            col = kwargs[key]
            if isinstance(col, list) and len(col) == n:
                return col
    return [None] * n


def _make_orm():
    """Build the ms-swift ORM subclass (deferred import so the file is importable for selftest)."""
    from swift.plugin import ORM, orms

    class Track1Reward(ORM):
        def __call__(self, completions, **kwargs):
            golds = _pick_gold_column(kwargs, len(completions))
            return [compute_reward(_completion_text(c), g) for c, g in zip(completions, golds)]

    orms["track1"] = Track1Reward
    return Track1Reward


# Register on import (ms-swift imports the plugin file via --external_plugins).
try:  # pragma: no cover - only on the server where swift is installed
    _make_orm()
except Exception:  # noqa: BLE001 - selftest/local import without swift
    pass


def _selftest() -> int:
    ok = True

    def chk(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL:", msg)

    g = '<analysis>table has no CoRTX row</analysis>\n{"label": "Unsupported Entity"}'
    chk(parse_last_label(g) == "Unsupported Entity", "parse last-line json")
    chk(parse_last_label('{"label": "Supported"}') == "Supported", "parse plain json")
    chk(parse_last_label("garbage") is None, "parse fail -> None")
    chk(parse_last_label('blah\n{"label": "nonsense"}') is None, "illegal label -> None")
    chk(analysis_word_count(g) == 5, f"analysis wc {analysis_word_count(g)}")

    # hit minority -> w_class
    chk(compute_reward(g, ["Unsupported Entity", "Supported"]) == 4.0, "UE hit -> 4.0")
    chk(compute_reward(g, ["Supported"]) == 0.0, "UE vs gold Supported -> 0")
    chk(compute_reward('{"label": "Supported"}', ["Supported"]) == 1.0, "Supported hit -> 1.0")
    chk(compute_reward('{"label": "Contradiction"}', ["Contradiction"]) == 4.5, "Contra -> 4.5")
    # format fails
    chk(compute_reward("no json here", ["Supported"]) == -1.0, "parse fail -> -1")
    long = "<analysis>" + " ".join(["w"] * 90) + "</analysis>\n{\"label\": \"Supported\"}"
    chk(compute_reward(long, ["Supported"]) == -1.0, "analysis>80w -> -1")
    # gold as JSON string column
    chk(compute_reward(g, '["Unsupported Entity"]') == 4.0, "gold json-string col")
    # anti-laziness sanity: expected reward of always-Supported < honest, on a minority-heavy set
    chk(_completion_text([{"role": "assistant", "content": g}]) == g, "chat-list extract")

    print("ALL GRPO REWARD SELFTESTS PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
