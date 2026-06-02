# Track 1 数据 / 指标分析（决定策略的依据）

> 数字来自 `../NLPCC-2026-Task10-Science/data/traindev-track-1.jsonl`（3333 records）与
> `testp1-track-1.jsonl`（586 records）。用官方 `offline_eval/evaluate.py` 复核打分逻辑。

## 1. 规模

| 切分 | records | sentences | 备注 |
|---|---:|---:|---|
| train-dev | 3333 | 17547 | 自带/可生成 id；含 gold `types` |
| Phase1 test | 586 | 5096 | 自带 id（`track1-p1-test-NNNNNN`），无 gold |

- 句子数/段落：1–22，集中在 2–8 句（4 句最多）。
- `claim_text`：均值 726 / 中位 671 字符。

## 2. 证据（为什么用 VL 模型）

- train-dev evidence item：**image 2017 + table 1979，没有任何 `text` 类型**。
- 每条 evidence 都带 caption（均 276 / 中位 208 字符），但**真实表格数字在图里**。
- 平均 1.2 个 evidence/record（最多 7）。
- → 仅靠 caption 文本不足以判 `Contradiction` / `Unsupported Entity`（要核对图表里的具体数值/实体），**多模态（Qwen3-VL）是合理选择**。

## 3. label 分布（含对原计划的纠错）

原计划写的 `Unsupported Entity(276)` 是**笔误**：276 是 *Unsupported Causal Mechanistic* 的 any-of 计数。真实：

| Label | primary `types[0]` | any-of `types`（GUIDELINES 表）|
|---|---:|---:|
| Supported | 16438 (93.7%) | 16438 |
| Scope Overgeneralization | 540 | 605 |
| Contradiction | 216 | 216 |
| Unsupported Entity | 213 | 639 |
| Unsupported Causal Mechanistic | 140 | 276 |

- 527 句多标签。评测：命中 gold `types` 任一即对；F1 用命中的 label，错则用 `types[0]`。
- **训练目标**：多标签句优先取**少数类**（利于 macro-F1），单标签句直接用该 label。

## 4. 指标动力学 —— 全场最关键

`score = (Sentence Macro-F1 + PEM) / 2`，5 类等权 macro-F1。

**trivial「全 Supported」基线：**
- Macro-F1 = **0.1935**（Supported 一类 F1≈0.967，其余 4 类 0；/5）。
- PEM = **0.6697**（**2232/3333 段全是 Supported**，全预测 Supported 即整段对）。
- **score ≈ 0.4316 = 43.2%**。

**官方报告的 LLM 基线（同一指标）：**

| Model | score | Macro-F1 | PEM |
|---|---:|---:|---:|
| Gemini 3.1 Pro | 10.35% | 19.17% | **1.54%** |
| GPT-5.4 | 16.35% | 26.04% | 6.66% |
| Qwen3.6-Plus | 18.90% | 25.33% | 12.46% |

→ **三个 LLM 全部低于 trivial 43.2%**。原因：它们乱开少数类，把 67% 本来干净的段落 PEM 打碎（Gemini PEM 从 0.67 掉到 0.015）。Macro-F1 只小涨（+0.06~0.07），但 PEM 暴跌，净亏。

**推论（写进 CLAUDE.md §4）：**
1. score 由 **PEM 主导**，是 **precision-critical**。
2. Supported 句上每个 false positive 都可能让整段 PEM 翻车，代价远大于多召回一个少数类的收益。
3. 胜负手：**保住 PEM（别打碎干净段）+ 只在高置信时召回少数类。**
4. 「过采样少数类」会抬 recall 但可能压 precision → 必须用 **9:1 dev 上的真实 score**（而非少数类 recall）来调重采样比例 / 阈值 / 弃权。

## 5. 难段落画像

- 含 ≥1 个非 Supported 句的「PEM-hard」段落：**1101/3333 = 33%**。
- 这 33% 才是 Macro-F1 与 PEM 的真正战场；另外 67% 只要别乱动就稳拿 PEM。

## 6. 复算脚本（可在服务器/本地纯 Python 重跑）

见 git 历史中本文件配套的一次性分析；核心：
```python
import json
recs=[json.loads(l) for l in open(f"{DATA_ROOT}/data/traindev-track-1.jsonl")]
all_sup=sum(all('Supported' in s['types'] for s in r['sentence_label']) for r in recs)
# PEM_trivial = all_sup/len(recs) = 0.6697
```
