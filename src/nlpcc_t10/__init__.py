"""NLPCC 2026 Task 10 Track 1 — claim-level faithfulness.

Pipeline: build_dataset -> (ms-swift LoRA SFT) -> infer -> aggregate -> eval_local.
See CLAUDE.md for the full design and constraints.
"""

__version__ = "0.1.0"

# 官方 5 类标签（字符串必须逐字一致；见 CLAUDE.md §2 / offline_eval/evaluate.py）。
TRACK1_LABELS = [
    "Supported",
    "Unsupported Causal Mechanistic",
    "Unsupported Entity",
    "Scope Overgeneralization",
    "Contradiction",
]

# 解析失败 / 越界标签时的安全兜底（最大化保 PEM，见 CLAUDE.md §4、§8）。
FALLBACK_LABEL = "Supported"
