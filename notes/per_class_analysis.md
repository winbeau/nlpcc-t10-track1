# 逐类判断报告:难易、瓶颈、按 ROI 排序的未来方法(2026-06-14,8-agent workflow)

> 模型 = A1_plainCE greedy on densematch(image-disjoint train' 干净 8B,代表 8B 家族)。计分 = 官方 any-of credit。所有路由计数/纠缠数字在 train'(2832 rec)+ dev_gold_densematch(206)上**逐一实测**。

## 1) 逐类总表

| 类 | gold | P | R | F1 | 难度 | 主因 | 需新能力 | train' target 覆盖 |
|---|---:|---:|---:|---:|---|---|---|---:|
| Supported | 1058 | .98 | .99 | **.99** | 已解 | — | 否 | 1.00 |
| UCM | 40 | .95 | .97 | **.96** | **已解** | 全局最稀→min() 永远赢,0 丢失 | 否 | 236/236=**1.00** |
| UE | 110 | .88 | .68 | **.77** | 中 | 缺 cue 漏判 + figref 过触发(FP) | 多数否 | 448/529=.85(**净受益**) |
| SO | 105 | .93 | **.26** | **.40** | 难 | **训练目标路由(68.6% 被路由走)** | **否(可修)** | 157/500=**.31** |
| Contra | 26 | 1.0 | **.08** | **.14** | 最难 | **路由(52% 丢)+ 读图数字(纯-Contra 核心)** | **部分需** | 91/190=.48 |

**冒烟枪:target 覆盖率≈线性预测召回**(SO .31↔.26;UCM 1.00↔.97)。两大瓶颈 = SO/Contra 召回,均"高精度-低召回→默认吐 Supported"(SO→Sup 9/SO→UE 5;Contra→Sup 5/Contra→UE 2)。

## 2) 错因总图(按召回缺口归因)

| 类 | 训练目标路由 | 读图数字 | 惯例cue | 多标签 | 稀缺 |
|---|---:|---:|---:|---:|---:|
| SO | **~65%** | ~25% | ~10% | (并入路由) | 低 |
| Contra | **~55%** | **~30%**(5/26 纯数字✅) | — | ~5% | ~10% |
| UE(此为 FP/过触发) | ~55%(prior 膨胀) | ~15% | ~30%(figref 过触发) | — | — |

- **路由是最大共因**:既饿死 SO/Contra 正样本,又把 UE prior 喂胖→figref 过触发污染 UE 精度。
- **Contra 硬地板 = 读图数字**:densematch 仅 **5/26 纯-Contra**(其余 21 多标签可被路由修复救回);这 5 句需读表格数字(8B 弱项;GPT-5.5 读数 R.65 但 P.27 过报)。
- **SO 几乎不需读图**:错例全是强 scope cue(`across all panels`/`Across all tables`/`throughout Table`)= 语言惯例题,不是读图题。
- **UCM 全程 0 路由损失→已解;任何方案不能动 UCM 的 236 目标**(148 来自多标签,re-route 会让它回归)。

## 3) 未来方法(按 ROI)

> ⚠️ 对抗核实关键裁决:**multi-target 不是精度改进器,本质仍是召回加法器**(+343 SO/+99 Contra 训练副本),面对把"加 firing"清零三次(s30/s16/s41)的同一道 testp1 **PEM 精度墙**。它比纯推理期加法器(s41/union)**安全一档**(教模型一个从未训过的 cue=修训练产物,而非单纯 OR 更多 firing),但**不能按 densematch 外推卖"+4~6 macro"**。现实 testp1 = **持平~小正**,**必须 Codabench 门控**。

### A 类(改训练目标/数据,便宜,优先)
- **★A1 Multi-target(首推 linchpin)— 攻 SO+Contra**:多标签句不再 pick_rarest,对 types[] **每个 gold 各出一训练副本**。实测 SO 157→500、Contra 91→190、UE 448→529(**涨**)、UCM 236→236(不回归已解类)。预期(校准):SO R .26→**.40-.60**;Contra R .08→**.15-.30**(CI 极宽);densematch macro +3~5,**testp1 现实 持平~+3**。成本低(~20 行改 `pick_rarest_label`/`expand_record`/`expand_record_joint` L576)。**零 GPU 预检**:重建 train_sft 断言 SO~500/UE~529/UCM236。
  - ⚠️ **零和风险(kill-switch)**:62/105 SO-gold 与 UE 纠缠,纯加法 headroom 仅 43 句;部分 SO 增益只能从 UE 桶转 credit(pred UE→SO = +1SO −1UE,互斥)。**评测盯 UE-R:若从 .68 跌破 ~.55 = 零和搬运,回退。**
- **A2 SO 优先路由([SO,UE]→SO)**:A1 更省变体,唯一带真精度机制(移除 UE 副本→降 figref 过触发),但代价 UCM 236→~153、UE→~206 回归。**混合,不如 A1 纯加法。**
- **A3 路由修复后逐类过采样 ⚠️ 默认 KILL**:纯召回加法器=死亡模式(s09/s33-38 全崩,oversample+plainCE 砸 PEM)。仅 A1 过 Codabench 后单独小试(SO 2-3×/UE 1×),PEM 不跌才留。
- **A4 multi-target joint 作 union 成员 ⚠️ 低优先**:堆 union 成员已死(s18-29 ≤50.26)。

### B 类(需读图,贵/难,低优先)
- **B1 OCR/数字通道 — 仅攻 Contra 5/26 纯数字核心**:至多 Contra R +.1-.2/macro ~+1;FP 风险高(P.27),PEM 受限会砸,需严格高置信门。**最低优先**。预检:人工查那 5 张图,caption 无数字+OCR 不可靠则跳过。
- **B2 推理期 SO scope-cue prior 后处理 ⚠️ 默认不上**:cue 精度仅 .288=低精度抑制类邻域死亡模式。仅"top1=Supported 且含强 cue→翻 SO"严格门控 + 离线 PEM 不跌才作 A1 补充。

## 4) 本周该先做
1. **★A1 multi-target 端到端一发**:Step0 零 GPU 改路由+重建 train_sft 断言计数(不达标=edit 错,不进 GPU)→ Step1 train' 重训 densematch 评(SO-R 升 **且** UE-R 不崩)→ Step2 净正交一发 testp1 Codabench(唯一可信信号;densematch 小样本 CI 宽+ρ.59 高估加 firing 杠杆)。
2. (条件)A1 过后再单独小试 A3,PEM 不跌才留。**不要 A1+A3 一起上。**

## 诚实边界
- **可改(训练产物,无新能力)**:SO 召回崩塌主体(路由)、UE figref 过触发、Contra 21/26 多标签部分。
- **受能力限(需读图,贵)**:Contra 5/26 纯数字冲突——8B 读不出表格数字,multi-target 救不了,**Contra F1 硬地板**。
- **不能动**:UCM(236 目标 0 损失,re-route 会回归)。**s01(47.80)/s15(50.26) 永久保底。**

改点:`src/nlpcc_t10/build_dataset.py`(`pick_rarest_label` L194 / `expand_record_joint` L576)。
