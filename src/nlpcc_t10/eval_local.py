"""在 9:1 dev 上调官方评测器打分（薄封装）。

**永远用官方 offline_eval/evaluate.py 打分，不要自写指标**（CLAUDE.md §7）。
本模块只是便捷封装：定位官方评测器、跑、解析并打印 score / Macro-F1 / PEM / per-label，
便于在不同配置（重采样比例、阈值、checkpoint）间比 **score**。

设计契约：
  - --gold 默认 data/dev_gold.jsonl（build_dataset.py 生成，带 id + gold sentence_label）。
  - --pred 是 aggregate.py 的提交格式 {id, labels}。
  - 子进程调用 $DATA_ROOT/offline_eval/evaluate.py --track 1 --match id，
    读回 <pred_stem>_eval_result.json，打印关键指标（含 per-label F1，定位哪个少数类拖后腿）。
  - 可选 --log notes/experiments.csv：把 (配置 tag, score, macro_f1, pem) 追加，建实验台账。

CLI:
  uv run python -m nlpcc_t10.eval_local \
      --pred outputs/dev_submission.jsonl --gold data/dev_gold.jsonl \
      --data-root "$DATA_ROOT" [--tag oversample3x_thr0.6] [--log notes/experiments.csv]
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score Track 1 dev preds via the official evaluator.")
    p.add_argument("--pred", required=True)
    p.add_argument("--gold", default="data/dev_gold.jsonl")
    p.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    p.add_argument("--tag", default="", help="实验标签，记入台账")
    p.add_argument("--log", default="", help="CSV 台账路径，追加 (tag, score, macro_f1, pem)")
    return p


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    raise NotImplementedError(
        "实现待定（CLAUDE.md §9）。核心是子进程调用 offline_eval/evaluate.py 并解析报告。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
