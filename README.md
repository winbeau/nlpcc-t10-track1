# nlpcc-t10-track1

NLPCC 2026 Shared Task 10 — **Track 1**（Claim-level Faithfulness to Experimental Results）参赛方案。

句子级判别每句声明对图/表证据的忠实度，5 类标签：`Supported` / `Unsupported Causal Mechanistic` /
`Unsupported Entity` / `Scope Overgeneralization` / `Contradiction`。

## 方案概览

- **模型**：Qwen3-VL-8B + LoRA（bf16），框架 **ms-swift**。
- **建模**：句子级——共享 evidence/claim 前缀 + TARGET 句子 → 1 个 label；推理时同段前缀 KV-cache 复用。
- **不均衡**：过采样 4 个少数类 + 下采样 Supported（按 dev **score** 调比例）。
- **验证**：按 record 9:1 切分；官方 `offline_eval/evaluate.py` 打分。

> 关键认知：官方 score = (Macro-F1 + PEM)/2 **由 PEM 主导**，trivial「全 Supported」≈43.2% 即超过所有 LLM 基线。
> 胜负手是保住干净段落的 PEM、只高置信召回少数类。详见 [`notes/analysis.md`](notes/analysis.md)。

## 约束

- 训练 / 推理 / 下权重 / 解压图片 **只在远程 GPU 服务器**（tmux + ssh）跑，本地不跑重活。
- 用 `uv` 管理；重依赖在 `[project.optional-dependencies].train`，仅服务器 `uv sync --extra train`。
- 官方数据见 sibling 仓库 `../NLPCC-2026-Task10-Science/`，**不复制、不提交、不转发**。

详细约定、流程与命令见 [`CLAUDE.md`](CLAUDE.md)。
