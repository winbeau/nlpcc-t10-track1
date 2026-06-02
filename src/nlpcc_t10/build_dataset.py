"""构造句子级训练/验证集（ms-swift 格式）。

设计契约（实现时遵循 CLAUDE.md §3–§5）：

输入: $DATA_ROOT/data/traindev-track-1.jsonl（3333 records / 17547 sentences）。

步骤:
  1) 按 **record** 9:1 切分 train/dev（**不可按句切**：同段句子会泄漏，且 dev 上 PEM 才有意义）。
     尽量按"该 record 是否含非 Supported 句"分层，保证 dev 的少数类覆盖。
     固定随机种子，切分结果落盘可复现。
  2) 句子级展开: 每个句子 → 1 条样本，prefix = (system + label 定义 + evidence_bundle[图+caption]
     + 完整 claim_text + "TARGET sentence: <该句>")，target = {"label": <单标签>}。
     - 多标签句: target 取 **少数类**（按全局频次最低者），单标签句取该 label（见 §3 评测规则）。
     - evidence 图片走 ms-swift 多模态 messages（<image> + 解析后的 images/<sha>.jpg 路径）。
  3) 不均衡处理（**仅对 train**）: 过采样 4 个少数类 + 下采样 Supported。
     - 倍率/采样率作为 CLI 参数，默认温和；**最终比例按 9:1 dev 的官方 score 网格选**（不是少数类 recall）。
     - dev 保持**原始分布**（不重采样），否则评测失真。
  4) 同时生成本地评测用的 dev_gold.jsonl（带 id、保留 gold sentence_label），供 offline_eval/evaluate.py。

输出（写到 --out 目录，默认 data/）:
  - train_sft.jsonl   # 句子级、已重采样，ms-swift messages 格式
  - dev_sft.jsonl     # 句子级、未重采样（推理/调参用）
  - dev_gold.jsonl    # record 级、带 id + gold sentence_label（官方评测器用）
  - split.json        # 记录 record id -> train/dev，便于复现与聚合对齐

CLI:
  uv run python -m nlpcc_t10.build_dataset \
      --data-root "$DATA_ROOT" --out data/ \
      --dev-ratio 0.1 --seed 42 \
      --minority-oversample 1.0 --supported-downsample 1.0   # 默认不改分布；调参时覆盖
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build sentence-level ms-swift dataset for Track 1.")
    p.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    p.add_argument("--out", default="data/")
    p.add_argument("--dev-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minority-oversample", type=float, default=1.0,
                   help="少数类过采样倍率（>1 增）；按 dev score 调")
    p.add_argument("--supported-downsample", type=float, default=1.0,
                   help="Supported 下采样保留率（<1 减）；按 dev score 调")
    return p


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    raise NotImplementedError(
        "实现待定（CLAUDE.md §9）。在远程服务器、图片解压后执行；"
        "切分按 record、dev 不重采样、官方 evaluate.py 打分。"
    )


if __name__ == "__main__":
    raise SystemExit(main())
