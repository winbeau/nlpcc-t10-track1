# CoT+GRPO 路线总概（P1.5：教材蒸馏 → CoT-SFT → GRPO）

> 2026-06-10 定稿。环节文档：**①造教材** `notes/cot_distill_plan.md` / **②CoT-SFT** `notes/cot_sft_plan.md` / **③GRPO** `notes/grpo_plan.md`。
> 一句话：**s01 配方不动，只加 CoT**（教材 = gpt-5.5 接地读图 × gold 惯例），SFT 点火后用 GRPO 直接优化类平衡指标——目标是造出 union 里没有的「新正确少数类召回」，单模破 47.80 或当 s15 的强异构成员。

## 0. 论证摘要（为什么是这条路，详见 2026-06-10 调研对话）

- **DPO 已论证放弃**：极短 JSON label 输出是 likelihood displacement 最坏场景，DPO 退化为对比 CE（=拟合数据）；实证 GRPO > DPO（arXiv 2512.22631）。
- **GRPO+CoT 的两个独特价值**：① 直接优化不可微的类平衡指标（SFT 的 CE 做不到）；② 强化的是「先查实体/数字/范围再判」的**推理动作**而非「题面→label」条件反射 → 抗 Phase-2 跨域（"SFT memorizes, RL generalizes", arXiv 2501.17161）。
- **RL 只放大已有能力不创造**（pass@k 上限，arXiv 2504.13837）⇒ **SFT 热启动必须先行**，且训练分布要让正确少数类 rollout 采得出来（过采样 + 密度匹配上下文）。
- **教材分工**（呼应 `notes/gpt5_probe_result.md`：gpt5 当分类器死、读数能力真）：**gold 给惯例，gpt-5.5 只给接地的证据读取**——不与已证伪结论冲突。

## 1. 三环节与闸门

```
①造教材(任意机,纯API)          ②CoT-SFT(H200)                  ③GRPO(H200×2)
train' 15008句 CoT units   →   s01配方+CoT目标(E0/E1/E2)   →   E1/E2 ckpt 起步,类平衡reward
   │G1 试点闸门                  │G2 CoT增益闸门(densematch)      │G3 RL增益闸门(densematch)
   │agreement≥55%               │E1>E0 且少数类召回不降           │>E1 且少数类开火率健康
   ▼                            ▼                                ▼
 17k 调用全量              Track B: full-traindev 冲榜版      Phase-2 主力/union 成员
```

- 每个闸门**不过就止损**：G1 不过改 prompt 重试点；G2 不过 CoT 线停、GRPO 退化为 label-only 版（价值降低仍可试）；G3 不过保 E1/Track B 成果。
- **评测纪律**：densematch 离线台只用于**同-regime 排序**（s31/s32/s30 两台同序已实证；不能跨 regime 外推、不给绝对分）。每个 winner 至多 1 发 testp1，命名 s{NN} 规范 + 追加 `notes/submissions_log.md`。**s01(47.80)/s15(50.26) 永久保底不动。**

## 2. 存储与服务器分工（硬约束）

| 资产 | 去处 | 工具 |
|---|---|---|
| **模型权重**（所有新 adapter） | **model HF** `winbeau/nlpcc2026-task10`（model 仓）`models/<名>/` | `scripts/hf_push.py`（ADAPTERS dict 加条目，服务器跑） |
| **数据集**（教材/构造的 SFT 集/devbench/raw） | **dataset HF** `winbeau/nlpcc2026-task10`（dataset 仓）`track1/{raw,processed/{devbench,cot}}` | `scripts/hf_data.py push\|pull`（任意机，零项目依赖） |
| 官方原始数据+图片 | 各机器从官方 git lfs 拉，images 不进 HF | `git lfs pull` + unzip |

服务器分工：**造教材 = 任意有 DATA_ROOT+图片的机器**（默认 H200，纯 API 不占卡）；**SFT/GRPO = H200**（GRPO 需 2 卡 colocate）；本地只做设计/轻分析/文档。机器间一切数据经 dataset HF 中转，互不直连。

## 3. 时间表（⚠️ Phase-2 窗口：6-11 放出，6-20 全部截止）

| 日期 | 事项 |
|---|---|
| 6-10 | 文档落盘（本批）；写 `cot_distill.py`；试点 50 records → G1 |
| 6-11 | **Phase-2 放出：当天最高优先 = 打通 Phase-2 推理提交管线，s01/s15 配方先交保底**；教材全量 Stage A 启动（夜间） |
| 6-12 | Stage B/C；E0 对照训练（可与教材并行：E0 不需要教材） |
| 6-13 | 教材交付（push HF）→ E1 训练 + G2 闸门 |
| 6-14~15 | E2（密度合流）/ Track B（full-traindev 冲榜版）训练 + 提交；GRPO 环境冒烟（reward plugin + vLLM bug 规避） |
| 6-15~17 | GRPO 正式训练 + G3；中途 densematch 监控 |
| 6-18~19 | 赢家 → Phase-2 推理 + 提交；union 重组实验（E1/GRPO 模型进 s15 成员池） |
| 6-20 | 截止日 buffer，只交不训 |

冲突规则：**GPU 冲突时 Phase-2 保底推理 > GRPO 训练**；RL 线任何一步延误不得挤占 6-18 之后的提交窗口。

## 4. 红线汇总（各环节文档展开）

1. 教材只产自 devbench image-disjoint **train'**（2832 records）；dev' 教材仅 Track B 冲榜时独立文件另产，物理隔离（见 ②§Track B）。
2. devbench 模型（E0/E1/E2/GRPO-on-E1）**绝不碰 dev'**，否则离线台报废。
3. **配方变量唯一性**：E1 相对 E0 只改训练目标（+CoT），其余（softmin/过采样/分辨率/epoch/卡数）一律 s01 原参——否则归因失效。
4. label 字符串逐字一致；解析失败 fallback Supported；提交前 `validate_submission.py`。
5. 商业 API 使用须报告 model/version/调用量（`data/cot/stats.md` 记录）。
