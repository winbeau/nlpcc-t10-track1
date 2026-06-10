# 环节③ GRPO 后训练计划（在 CoT-SFT 上吹旺）

> 总概 `notes/cot_grpo_roadmap.md`；上游 ②`notes/cot_sft_plan.md`（G2 过闸的 E1/E2 checkpoint = 本环节起点；G2 不过则退化为 E0 上的 label-only GRPO，价值降低仍可一试）。
> 定位：SFT 拟合数据，GRPO 学方法——①直接优化**类平衡的不可微目标**；②强化「查实体/数字/范围」的推理轨迹而非条件反射，抗 Phase-2 跨域。**RL 只放大已有火苗**：起点模型必须偶尔能答对少数类（E1 的 pass@8 决定可训性，见 §3）。

## 1. 形态与框架

- **格式 = per-sentence**（与 E1 策略的训练格式严格一致；rollout = `<analysis>...</analysis>\n{"label":...}`）。段落级 PEM-bonus 需要 joint 格式，**本期不做**（s01 线是 per-sentence；joint+PEM reward 留作后续变体）。
- 框架：ms-swift `swift rlhf --rlhf_type grpo`，**colocate vLLM** + offload，2×H200。
- **工程坑（前期调研已核实，照抄规避）**：
  - Qwen3-VL + `vllm_enable_lora true` 触发 vLLM LoRA shrink bug（ms-swift #6670/#6506，未修复）→ **`--vllm_enable_lora false`（全权重同步，慢但可用）**。
  - `gradient_accumulation_steps ≥16` 有视觉 embedding shape mismatch（#6521）→ **grad_accum ≤8**。
  - `USE_HF=0` 必设（基座在 ModelScope ms_cache，否则卡死在 HF 下载）。
  - 多图 record 显存峰值高 → 沿用 MAX_PIXELS 401408；`--offload_optimizer true --offload_model true --sleep_level 1`。

## 2. Reward 设计（custom ORM plugin，防三种钻空）

gold 经数据集列传入（ms-swift reward fn 可读全部列）；对每条 rollout：

```
parse 末行 JSON →
  解析失败 / label 非法 / analysis>80词        → R = -1        (防格式钻空)
  label 命中 gold types 任一                   → R = w_class    (类平衡)
  label 错误                                   → R = 0
w_class：Supported=1.0；四少数类 = min(5.0, 训练集逆频)，初值 UCM 5.0 / Contra 4.5 / UE 4.0 / SO 3.5，
        以「全-Supported 策略的期望 reward < 诚实策略」为校准下界（防躺平，s05=18.85 的教训）
```

- **reward 只看 label 不看推理内容**（推理真假由 ①Stage C 在教材侧把关 + ②G2 人检兜底）——但 analysis 长度上限进 reward，防 rationale 膨胀骗熵。
- 多标签句命中 gold `types` 任一即得分（与官方评测一致）。
- 实现：`scripts/grpo_reward.py`（ORM 子类，`orms['track1'] = ...`），CLI `--external_plugins scripts/grpo_reward.py --reward_funcs track1`。

## 3. 数据与采样（RL 继承 prompt 分布——伪先验在这里修）

- **训练 prompt 集 = E1 的重采样训练集**（os3.0/ds0.6 的 train' 句子级样本，去掉 assistant 列）；E2 过闸则加密集合成段上下文的样本——**密度匹配的上下文必须进 RL 的 prompt 分布**，否则 GRPO 在稀疏上下文里强化出的还是旧先验。
- `--num_generations 8`，rollout temperature 1.0（起步）。
- **静态 DAPO 预筛（解零梯度）**：训前用 E1 对全部候选 prompt 各采 8 次，**只保留 pass@8 ∈ (0,1) 的样本**（全对=已会、全错=没火苗，advantage 均为 0 白算）。预筛同时输出报告：各少数类 pass@8>0 的句子占比 = 本环节可训上限的直接估计（若某类 <20%，对该类的期望要下调）。
- 规模：预筛后取 ~3-5k prompts，1-2 epoch；`per_device_batch 1-2 × grad_accum ≤8`。

## 4. 监控与闸门 G3

训练中每 ~200 step：
1. **reward 曲线**（升→正常；陡升→查钻空）；
2. **少数类开火率**：rollout 中少数类预测占比（塌向 0 = 躺平回潮 → 调高 w_class；爆涨 → 查 reward/降 w）；
3. **KL/熵**：熵塌缩（8 条 rollout 全同）→ 升 temperature 或加 entropy bonus；
4. **densematch 快评**：当前策略贪心解码 → 官方 evaluate.py。

**G3 闸门**：densematch 上 GRPO > E1（同-regime 排序可信）且少数类逐类命中不偏科 → 1 发 testp1（s{NN} 取号 + log）→ 进 Phase-2 主力池 + 作 union 异构成员重组 s15（GRPO 模型的错误模式与 CE/softmin 模型本质不同，正是 naive-union 天花板 50.26 需要的新召回来源）。不过闸 → 保 E1/Track B 成果，GRPO 经验记 log。

## 5. 产物与归档（硬约束：权重→model HF，数据→dataset HF）

| 产物 | 去处 |
|---|---|
| GRPO adapter（LoRA） | model HF `models/grpoE1_<tag>/`（`hf_push.py` ADAPTERS 加条目） |
| 预筛 pass@8 数据 + RL prompt 集 | dataset HF `track1/processed/cot/grpo/`（`hf_data.py push --what cot`） |
| reward plugin / 训练脚本 | 仓库 `scripts/`（git）+ model HF scripts/（hf_push 自动） |
| 提交 | `submissions/testp1_s{NN}_...` + log 一行 |

## 6. 风险与时间盒

| 风险 | 对策 |
|---|---|
| 全权重同步慢（vllm_enable_lora false）→ 吞吐低 | 接受；规模盒定在 3-5k prompts;若 >36h 不收敛即止损 |
| reward 钻空（格式游戏/analysis 膨胀） | §2 三防 + 人工抽 rollout 20 条/天 |
| 熵塌缩 / 训练发散 | §4 监控项 2/3;KL β 从 ms-swift 默认起，不行再调 |
| 挤占 Phase-2 保底推理 | 总概 §3 冲突规则：**保底推理 > GRPO**；GRPO 最晚 6-17 止损出结论 |
| E1 少数类 pass@8 过低（火苗不足） | §3 预筛报告先行——开训前就知道值不值得烧 |
