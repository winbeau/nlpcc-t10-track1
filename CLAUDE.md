# CLAUDE.md — NLPCC 2026 Task 10 / Track 1 解题仓库

> 本文件每个 session 自动载入。改动时保持精简（只放跨 session 必须知道的事实与约束）。
> 全局 `~/.claude/CLAUDE.md` 仍生效；本项目级文件优先级更高。

## 0. 这是什么

NLPCC 2026 Shared Task 10 **Track 1**：*Claim-level Faithfulness to Experimental Results*（科学报告的可靠性 / 声明级忠实度）的参赛仓库。

- 官方题目仓库（**只读，不要改**）：`../NLPCC-2026-Task10-Science/`
  - 指南：`GUIDELINES.md`（EN）/ `GUIDELINES_ZH.md`（中）
  - 数据：`data/`（train-dev + Phase 1 test，图片在 Git-LFS）
  - 官方评测器：`offline_eval/evaluate.py`（**唯一权威打分**，必须用它）
  - 基线 prompting kit：`baseline_prompting/`
- 目标：在官方指标上**超过 trivial baseline（≈43.2%）**并尽量冲榜（见 §4）。
- Phase 1 提交平台：Codabench 16666。Phase 2 隐藏测试集 2026-06-11 放出，**所有结果 2026-06-20 截止**。

## 1. 硬约束（务必遵守）

1. **绝不在本地下载模型权重 / 跑 `uv sync` / 跑训练或推理 / 解压图片。** 一切重活在**远程 GPU 服务器**上做，通过 **tmux + ssh**（用 `tmux-ssh-remote` skill）。本地只做：读数据、写代码、设计、轻量纯 Python 分析。
2. **用 `uv` 管理项目**（`pyproject.toml`）。重依赖在 `[project.optional-dependencies].train`，**只在服务器** `uv sync --extra train`。
3. **数据使用政策**：官方数据仅限本赛事使用，**不得转发 / 镜像 / 改名再发布**。不要把 `data/` 或图片提交进 git。
4. 官方数据是 sibling 仓库，通过 `DATA_ROOT`（默认 `../NLPCC-2026-Task10-Science`）引用，**不复制**。派生产物（句子级数据、切分、预测、checkpoint）放本仓库 `data/`、`outputs/`，均已 gitignore。

## 2. 任务定义（Track 1）

每条样本：`claim_text`（一段 2–7 句的声明）+ `evidence_bundle`（图/表证据，见 §3）+ 已切好的句子列表。
**对每个句子输出 1 个 label**（5 类之一）：

| Label（**字符串必须逐字一致**） | 含义 |
|---|---|
| `Supported` | 句子被证据充分支持 |
| `Unsupported Causal Mechanistic` | 引入证据未支持的因果/机制解释 |
| `Unsupported Entity` | 提到证据未涉及的数据集/指标/模型变体/baseline 等实体 |
| `Scope Overgeneralization` | 把结论外推到证据支持范围之外 |
| `Contradiction` | 与证据直接矛盾 |

**提交格式**（JSONL，一条 record 一行）：
```json
{"id": "track1-p1-test-000008", "labels": ["Supported", "Contradiction", "Supported"]}
```
- `labels` 数量必须 == 该 record 句子数，**顺序与输入句子一致**。
- 测试集自带 `id`（如 `track1-p1-test-000008`），评测**按 id 匹配**，务必原样回填 id。

## 3. 数据关键事实

- 路径：`$DATA_ROOT/data/{traindev,testp1}-track-1.jsonl`；图片在 `images.zip` / `images-testp1.zip`（**LFS，需在服务器 `git lfs pull` 后解压到 `images/`**，使 `images/<sha>.jpg` 可解析）。
- train-dev：**3333 records / 17547 sentences**；Phase 1 test：**586 records / 5096 sentences**。
- **所有证据都是图/表**（train-dev 中 `image:2017`、`table:1979`，**没有任何 `text` 类型**；每条 evidence 都有 caption，平均 caption 276 字符）。→ 这是用 **Qwen3-VL（多模态）** 的核心理由：真正的表格数字在图里，caption 往往不够。
- 平均 1.2 个 evidence/record（最多 7）；`claim_text` 中位 671 字符。
- **真实 label 分布**（注意：纠正了原计划里 "Unsupported Entity 276" 的笔误）：

| Label | primary (`types[0]`) | any-of `types`（=GUIDELINES 表） |
|---|---:|---:|
| Supported | 16438 (93.7%) | 16438 |
| Scope Overgeneralization | 540 | 605 |
| Contradiction | 216 | 216 |
| Unsupported Entity | **213** | **639** |
| Unsupported Causal Mechanistic | **140** | 276 |

- **527 句是多标签**。评测规则：预测命中 gold `types` 中**任意一个**即算对；算 F1 时若命中用命中的那个作 reference，若错则用 `types[0]`。
  - → 多标签句构造训练目标时**优先选少数类**（更有利于 macro-F1 学习），单标签句直接用该 label。

## 4. 评测与打分策略（最重要）

`offline_eval/evaluate.py --track 1`：
- **Sentence Macro-F1**：5 类 F1 等权平均。
- **PEM (Paragraph Exact Match)**：整段所有句子全对才算 1。
- **score = (Macro-F1 + PEM) / 2**。

**核心洞察（决定整个策略）：**
- **trivial「全预测 Supported」基线 ≈ score 43.2%**（Macro-F1 **0.1935** + PEM **0.6697**）。
- 这个 trivial 基线**碾压官方报告的所有 LLM 基线**（Gemini 10.4% / GPT-5.4 16.4% / Qwen 18.9%）——因为 LLM 乱开少数类，把 **67% 本来全 Supported 的段落**打碎，PEM 崩到 1–12%。
- 结论：**score 由 PEM 主导，是 precision-critical 局面。** 在 Supported 句上每多 1 个 false positive，就可能让一整段从 PEM-correct 翻成 wrong，代价极大。
- **胜负手 = 在不打碎干净段落（保 PEM）的前提下，只在高置信时召回那 ~6.3% 少数类句子（提 Macro-F1）。**
- ⚠️ 因此「过采样少数类」（见 §5）必须配**精确率约束**：重采样比例、阈值/弃权要**按 9:1 dev 上的真实 score 调**，不能只看少数类 recall——否则 score 反而下降。

评测命令见 §7。

## 5. 方法（已定方案）

| 决策 | 选择 |
|---|---|
| 模型 | **Qwen3-VL-8B** + **LoRA**，bf16 |
| 训练框架 | **ms-swift**（ModelScope，对 Qwen3-VL 一等支持）|
| 建模粒度 | **句子级**：每个训练样本 = 共享 evidence/claim 前缀 + 1 个 TARGET 句子 → 输出 1 个 label |
| 数据扩展 | 3333 records → ~17547 句子级样本（推理时同一 record 的 evidence/claim 前缀做 **KV-cache 复用**，n 句只算 1 次前缀）|
| 类别不均衡 | **过采样 4 个少数类 + 下采样 Supported**（比例按 §4 的 dev score 调，防 PEM 崩）|
| 验证切分 | **按 record 9:1**（不可按句切，否则同段句子泄漏、且 PEM 无意义）；尽量按"是否含少数类"分层 |

**训练样本格式**（chat / ms-swift messages）：
- `system`：careful scientific claim verification system，输出严格 JSON。
- `user`：label 定义 + `evidence_bundle`（图片 + caption）+ 完整 `claim_text` + **TARGET sentence**（指明判这一句）。
- `assistant`（label / 训练目标）：`{"label": "Unsupported Entity"}`。

**推理后处理建议**（即使句子级训练，也在段落级做一次收口，呼应 §4）：
- 段落级一致性/阈值校正：对不确定句**偏向 Supported**（降低 false positive，保 PEM）。
- 阈值/弃权点在 9:1 dev 上以 **score** 为目标网格搜索。

## 6. 仓库结构

```
nlpcc-t10-track1/
├── CLAUDE.md                  # 本文件
├── pyproject.toml             # uv；重依赖在 [optional-dependencies].train
├── configs/
│   └── qwen3vl_lora_sft.yaml  # ms-swift 训练/推理配置模板
├── src/nlpcc_t10/
│   ├── build_dataset.py       # 9:1 切分 + 句子级展开 + 重采样 → ms-swift jsonl
│   ├── infer.py               # 批量推理（KV-cache 前缀复用）→ 句子级原始预测
│   ├── aggregate.py           # 句子级预测 → 提交格式 {id, labels}（保序、数量对齐）
│   └── eval_local.py          # 调官方 evaluate.py 在 9:1 dev 上打分
├── scripts/                   # 服务器上 tmux/ssh 跑的 .sh
│   ├── setup_env.sh           # uv sync --extra train + lfs pull + 解压图片（仅服务器）
│   ├── train.sh / infer.sh / eval.sh
├── notes/analysis.md          # 数据/指标分析与策略（含 §4 的全部数字）
├── data/   (gitignored)       # 派生数据：句子级 jsonl、9:1 split
└── outputs/(gitignored)       # 预测、checkpoint、eval 报告
```

## 7. 端到端流程（命令都在服务器上跑）

```bash
# 0) 一次性：环境 + 数据（服务器）
bash scripts/setup_env.sh        # uv sync --extra train; git lfs pull; unzip images

# 1) 构造句子级训练/验证集（9:1，重采样）
uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data/

# 2) LoRA 训练（ms-swift）
bash scripts/train.sh            # 读 configs/qwen3vl_lora_sft.yaml

# 3) 推理（dev 或 testp1）
bash scripts/infer.sh dev        # 或: bash scripts/infer.sh testp1

# 4) 聚合成提交格式
uv run python -m nlpcc_t10.aggregate --pred outputs/<raw>.jsonl --out outputs/submission.jsonl

# 5) 本地评测（仅 dev，有 gold 时）—— 用官方评测器
uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 \
  --gold data/dev_gold.jsonl --pred outputs/dev_submission.jsonl --match id
```
- **永远用官方 `offline_eval/evaluate.py` 打分**，不要自己另写指标。
- testp1 提交：跑 §3 的 `testp1-track-1.jsonl` → `aggregate` → 传 Codabench。

## 8. 约定与坑

- **label 字符串逐字一致**（含空格大小写），见 §2；非法 label 会被评测器直接报错。
- 提交 `labels` 数量必须 == 句子数，**顺序不能乱**；缺一条 record 评测会报 id 集合不一致。
- id：测试集自带，原样回填；本地 9:1 dev 用 `prepare_dev_eval.py` 风格的 `track1-NNNNNN` id 生成 gold。
- 模型解析失败/越界标签时，**fallback 到 `Supported`**（最安全、保 PEM）。
- 改 prompt / 重采样比例 / 阈值后，**必须在固定的 9:1 dev 上用官方评测器复测 score** 再决定。
- 实验记录写进 `notes/`（每次配置 → dev score / Macro-F1 / PEM），别只看单一指标。

## 9. 待办 / 开放项

- [ ] §5 句子级 prompt 模板定稿（label 定义措辞、是否给同段其他句作上下文）。
- [ ] 重采样比例（少数类倍率 / Supported 下采样率）网格——按 dev **score** 选。
- [ ] 推理段落级收口策略（阈值/弃权/偏 Supported）是否真涨 score。
- [ ] 表格图是否需要 OCR 辅助文本（vs 纯 VL）。
- [ ] LoRA 超参（rank/alpha/lr/epoch、是否冻结 vision encoder）。

## 10. 参考（官方文件路径）

- 指南：`../NLPCC-2026-Task10-Science/GUIDELINES.md`
- 评测器：`../NLPCC-2026-Task10-Science/offline_eval/evaluate.py`（Track1 逻辑见 `evaluate_track1`）
- 基线 prompt：`../NLPCC-2026-Task10-Science/baseline_prompting/prompts/track1_{system,user}.txt`
- dev 构造参考：`../NLPCC-2026-Task10-Science/baseline_prompting/prepare_dev_eval.py`
