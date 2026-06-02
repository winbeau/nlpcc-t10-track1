"""句子级原始预测 → 官方提交格式 {id, labels}。

设计契约（CLAUDE.md §2、§8）：

  - 读 infer.py 的句子级 raw（{id, sent_index, label, logprob?}），按 id 分组、按 sent_index 排序。
  - 输出每个 record 一行: {"id": <id>, "labels": [<每句 label>]}。
    * labels 数量必须 == 该 record 原始句子数（缺句 → 用 FALLBACK_LABEL 补，并 log 警告）。
    * 顺序严格按句子顺序；label 字符串必须 ∈ TRACK1_LABELS。
  - 可选段落级收口（--postprocess）: 用 logprob 阈值，对不确定句偏向 Supported，保 PEM（见 §4、§5）；
    阈值在 9:1 dev 上以官方 score 网格选定后再用于 testp1。
  - record 句子数从输入文件（dev split 或 testp1-track-1.jsonl）核对，保证 id 集合与句数对齐。

CLI:
  uv run python -m nlpcc_t10.aggregate \
      --pred outputs/testp1_raw.jsonl \
      --ref  $DATA_ROOT/data/testp1-track-1.jsonl \
      --out  outputs/testp1_submission.jsonl
      [--postprocess --supported-threshold 0.6]
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Aggregate sentence preds into Track 1 submission.")
    p.add_argument("--pred", required=True, help="infer.py 的句子级 raw jsonl")
    p.add_argument("--ref", required=True, help="对应输入 jsonl（取每 record 句子数与 id）")
    p.add_argument("--out", required=True)
    p.add_argument("--postprocess", action="store_true", help="开启段落级阈值收口（偏 Supported 保 PEM）")
    p.add_argument("--supported-threshold", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    raise NotImplementedError("实现待定（CLAUDE.md §9）。纯 Python，可在服务器或本地跑。")


if __name__ == "__main__":
    raise SystemExit(main())
