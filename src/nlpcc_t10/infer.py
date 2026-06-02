"""句子级批量推理（远程服务器）。

设计契约（CLAUDE.md §5）：

  - 输入 split: dev（用 split.json 取 dev records）或 testp1（$DATA_ROOT/data/testp1-track-1.jsonl）。
  - 对每个 record: 构造与训练一致的 prefix（system + label 定义 + evidence 图/caption + claim_text），
    再对 record 内每个句子追加 "TARGET sentence: <句>" 解码 1 个 label。
  - **KV-cache 前缀复用**: 同一 record 的 evidence/claim 前缀只前向一次，n 句复用（省 ~n 倍前缀算力）。
    用 ms-swift / vLLM 的 prefix-caching，或手动缓存 past_key_values。
  - 解析输出 JSON {"label": ...}; 非法/越界 → FALLBACK_LABEL（保 PEM，见 §8）。
  - 输出句子级原始预测 outputs/<split>_raw.jsonl，每行:
      {"id": <record_id>, "sent_index": i, "label": "...", "logprob"?: ...}
    （带 logprob/置信度便于段落级阈值收口，见 §5）。
  - aggregate.py 负责把句子级 → record 级 {id, labels}。

CLI:
  uv run python -m nlpcc_t10.infer --split testp1 --data-root "$DATA_ROOT" \
      --adapter outputs/qwen3vl8b_lora --out outputs/testp1_raw.jsonl
"""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentence-level inference for Track 1.")
    p.add_argument("--split", choices=("dev", "testp1"), required=True)
    p.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    p.add_argument("--adapter", default="outputs/qwen3vl8b_lora", help="LoRA adapter 目录")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    _ = build_arg_parser().parse_args(argv)
    raise NotImplementedError("实现待定（CLAUDE.md §9）。仅在服务器执行。")


if __name__ == "__main__":
    raise SystemExit(main())
