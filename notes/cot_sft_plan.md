# 环节② CoT-SFT 计划（s01 配方 + CoT 目标，点火）

> 总概 `notes/cot_grpo_roadmap.md`；上游 ①`notes/cot_distill_plan.md`（教材）；下游 ③`notes/grpo_plan.md`。
> **核心原则（用户钦定）：基座配方 = s01，其他方法一致，只加 CoT。** s01 = per-sentence **softmin**(β5/λ0.5) + **过采样 grid**(os3.0/ds0.6) + MAX_PIXELS 401408 + 1 epoch + 2 卡 DDP + `use_logits_to_keep false`（复现命令见 `notes/submissions_log.md` s01 节）。
> 为什么必须 softmin 不能 plain-CE：log 已钉死「**过采样 regime 下 plain-CE 过度开火少数类 → PEM 崩**（s33 42.87 << s01 47.80，PEM −5.8），softmin 的全部价值 = 控过采样诱发的 FP」。CoT 不改变这个机制。

## 1. 训练目标格式（与 ①③ 对齐的唯一标准，三文档一致）

```
user:    (s01 原版 per-sentence prompt：label 定义 + evidence 图/caption + 完整 claim + TARGET sentence)
assistant（有教材的句子）:
  <analysis>claim cites CoRTX vs L2X on Table 6; provided table has no CoRTX entry</analysis>
  {"label": "Unsupported Entity"}
assistant（label_only 桶的句子）:
  {"label": "Unsupported Entity"}
```

- **末行 = 与 s01 完全相同的严格 JSON**：推理/聚合解析只需改成「取输出末行 JSON」，fallback 链路（Supported 兜底）不变。
- `<analysis>` ≤45 词英文，来自 `data/cot/units.jsonl` 的 rationale（builder 原样填入，不再加 `-> label` 尾巴）。
- 混合训练合法：教材覆盖不到的句子（CANNOT_JUSTIFY/核验淘汰）保持纯 label 目标，模型学会「能分析则分析、不能则直接给」。

## 2. 实验矩阵（变量唯一性红线：每行相对上一行只改一处）

| 实验 | 数据 | 目标 | 用途 | 状态 |
|---|---|---|---|---|
| **E0** | train' + os3.0/ds0.6 | label-only | **对照**（s31 是无过采样 softmin=34.21，不能当对照，必须新训） | 不依赖教材，可先行 |
| **E1** | 同 E0 | **+CoT**（§1 格式） | 主实验：CoT 增益归因 | 待教材 |
| **E2** | E1 + 密度匹配合成段做 claim 前缀（P1 合流） | +CoT | 攻「≤1 少数类/段」伪先验：per-sentence 形态下，密度匹配体现在**TARGET 句所处的 claim 上下文**含多个少数类句；合成段内每句教材已有（①§5：跨 record 拼接无需重调 gpt5） | 待 E1 过闸 |
| **E3 = Track B** | **full-traindev** + os3.0/ds0.6 | +CoT | **冲榜版**（G2 过闸后）：需补 dev' 501 records 教材（重跑①管线，独立文件 `units_dev.jsonl`，物理隔离）；该模型放弃 devbench 评测资格，直接 testp1/Phase-2 | 待 G2 |

## 3. 实现要点

1. **builder**：`build_dataset` 加 `--cot data/cot/units.jsonl`——展开/重采样逻辑全不动，仅在产出 assistant content 时按 (record_id, sent_idx) 查 rationale 拼 §1 格式。过采样复制的样本带同一条 rationale（合理：教材随句子过采样）。
2. **softmin 落点（唯一实现难点）**：softmin-PEM loss 必须只作用于**末行 label JSON 的 token 段**，`<analysis>` 段走纯 CE——否则 softmin 会把"推理写得最差的 token"当瓶颈，语义全错。定位方式：目标模板固定（`</analysis>\n{"label":`），从 target 末尾反向匹配 JSON 段起点即可；`train_softmin.py` 里 per-sample 记录 label-span 偏移。**先写 10 行单测**（构造样本验证 span 定位）再开训。
3. **长度**：target 加 ~60 token；`MAX_LENGTH` 在 s01 基础上 +128 即可，显存影响可忽略。
4. **推理**：`infer.py` 解析改「末行 JSON」（兼容无 analysis 的输出）；`--max-new-tokens` 提到 ~120。**KV 前缀复用不变**（analysis 只增加每句 decode ~60 token；testp1 5096 句 ≈ +300k decode token，可接受）。min_logprob 改为对**末行 label token 段**计算（union/门控下游依赖它）。
5. **训练资源**：与 s01 相同（2×H200 DDP，~4h/个）；E0/E1 可串行一晚出双结果。

## 4. 闸门 G2（densematch，同-regime 排序可信——s31/s32 已实证）

```bash
uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 \
  --gold data/devbench/dev_gold_densematch.jsonl --pred <E*_pred> --match id
```

| 结果 | 判定 | 动作 |
|---|---|---|
| E1 > E0 且少数类召回（逐类命中数）不降 | **CoT 增益成立** | → E2、E3(Track B)、③GRPO 起步用 E1/E2 |
| E1 ≈ E0（±0.5 内） | 教材未起效 | 抽检 E1 输出的 analysis 质量（是否退化成套话）→ 修教材/格式再试一次 |
| E1 < E0 | CoT 有害（softmin-span 实现 bug 优先排查） | CoT 线止损；③GRPO 退化为在 E0 上做 label-only 版 |

- E1/E2/E3 每个 winner **至多 1 发 testp1**（s{NN} 取号 + 追加 log + `validate_submission.py`）。
- 附加观察（不作闸门）：E1 的 `<analysis>` 是否真引用图中数值（抽 30 句人检）——这是 ③ reward 不看推理的前提下，推理质量的唯一体检。

## 5. 产物与归档（硬约束：权重→model HF，数据→dataset HF）

| 产物 | 本地 | 归档 |
|---|---|---|
| 构造的 SFT 数据 | `data/cot/sft/{E0,E1,E2,E3}_train.jsonl` | dataset HF `track1/processed/cot/sft/`（`hf_data.py push --what cot`） |
| adapter ckpt | `outputs/cot_sft/E*/...` | model HF `models/cotE1_s01cot/` 等（`hf_push.py` ADAPTERS 加条目） |
| 预测/提交 | `outputs/` + `submissions/testp1_s{NN}_...` | submissions 照旧 git-track + model HF |
| 复现命令 | 追加 `notes/submissions_log.md` | 同步 SUBMISSIONS.md（hf_push 自动） |

## 6. 风险

| 风险 | 对策 |
|---|---|
| softmin label-span 定位错 → 静默训坏 | 单测先行；E1 训前先 100 步冒烟看 label-span loss 曲线分离 |
| 教材覆盖率低（少数类 label_only 占比高）→ CoT 增益稀释 | ①§7 stats 出来先看覆盖率，<60% 少数类覆盖则 G2 预期下调 |
| analysis 退化成模板套话（过采样复制加剧） | 教材生成时同句多条 rationale 轮换填充（①每句采 2-3 条正好够） |
| 时间挤压（Phase-2 窗口） | E0 不等教材先训；E1 一晚；G2 后 E2/E3 与 ③ 并行 |
