# Track1 错误样本 & 多模判断冲突 — 根因分析（2026-06-16）

> 方法：本地拉全部预测+证据图（10 个差异化模型 testp1 raw + clean phaseC + leaky s01 + 1128 张证据图，62MB），
> 先做量化对账，再用 multi-agent **实际读图**对 77 例（52 dev-gold 错例 + 25 testp1 十模冲突例）逐例诊断根因，
> 争议例由 opus 对抗复核。产物在 `data/analysis/out/`（metrics.json / dev_errors.jsonl / testp1_conflicts.jsonl /
> diagnoses.json / SYNTHESIS.md / cases/*.json）。脚本：`scripts/error_conflict_analysis.py`、`curate_cases.py`、`wf_root_cause.js`。

## 0. 量化地基（官方 evaluate.py 口径，any-of 匹配）

**CLEAN phaseC（densematch，无泄漏，最诚实）** macro-F1=0.792 acc=0.968，逐类 P/R/F1：

| | SUP | UCM | UE | SO | CON |
|---|---|---|---|---|---|
| Recall | .994 | 1.00 | .852 | **.658** | **.222** |
| F1 | .985 | .975 | .867 | .771 | **.364** |

- 40 个错例（errors-only）：**MISSED_MINORITY 26（65%）** > WRONG_MINORITY 8 > FALSE_ALARM 6。
  - 漏召回方向：UE→SUP ×12、SO→SUP ×9、CON→SUP ×5。错判少数类方向：主要 SO→UE ×5。
- **泛化鸿沟而非容量**：leaky s01（记住了含 dev 的全 traindev）CON R=.857 / SO R=.844 —— 干净一掉到 .22/.66。
- **置信度无法救场**：52% 干净错例 min_logprob>−0.05（p>0.95），是"自信地错"。
- **SO↔UE 混淆大半是合法多标签**：gold 里 SO+UE 共现 348 次（第一大多标签对，全体多标签仅占 3.0%）。
- **testp1 十模冲突**（5096 句）：unanimous-SUP 4510；SUP-vs-单少数类 460（召回战场）；SUP-vs-多少数类 65；
  unanimous-minority 50；minority-only 11。分歧 label-pair：SUP↔UE 312 / SUP↔UCM 127 / SO↔SUP 78 / CON↔SUP 75 / SO↔UE 40。
- **union 召回收割**：单模 flag 少数类 4.8% → 10 模并集 11.5%（+139%），但 precision 无 gold 不可验。
- 单模少数类调用率：Qwen32B-joint 3.6% … Gemma-26B 7.5%（2 倍差 = 激进/保守谱）。

## 1. 读图诊断的 root_cause 分布（77 例）

| root_cause | dev(52) | testp1(25) | 合计 | 性质 |
|---|---:|---:|---:|---|
| ENTITY_PRESENCE | 15 | 8 | **23** | 文本/结构接地，可修 |
| SCOPE_QUANTIFIER | 12 | 4 | **16** | 量词推理，可修 |
| CAUSAL_CLAIM | 4 | 10 | **14** | 子句分解，可修 |
| GOLD_ERROR | 12 | 0 | **12** | 标签噪声，模型不背锅 |
| DEFAULT_TO_SUPPORTED | 4 | 1 | 5 | 惰性偏置，可修 |
| 其它(MULTI_LABEL/NUMERIC_OCR/AMBIGUOUS/…) | 5 | 2 | 7 | 混合 |

- assessment：model_wrong_recoverable 42 / **model_defensible 24** / gold_suspect 8 / ambiguous 2 / model_wrong_hard 1。
- 对抗复核 verdict：**confirm_contested 27 vs reject_contested 6**（争议例 82% 站"模型/争议方有理"）。
- figure_readability：numbers_legible 62 / partly 10 / no_relevant_content 5 —— **图基本看得清，瓶颈不在像素**。

**两分布形状不同（最重要结构事实）**：dev 错例 = ENTITY+GOLD_ERROR+SCOPE 三分天下（GOLD_ERROR 占 23%）；
testp1 冲突 = CAUSAL_CLAIM+ENTITY 主导且 **GOLD_ERROR=0**（十模分歧是真难句，不是标注噪声）。

## 2. 召回崩塌的真因：≈95% 是「该判却懒判」，不是「想判却看不见」

把 43 个"真模型错"（剔除 gold 噪声/UNCLEAR）按"是否真需 8B-VL 力不能及的视觉"拆桶：

| 需求类型 | 真错例 | 可修性 |
|---|---:|---|
| 文本/结构接地（句中图号·指标名 vs caption/表头） | 17 | 完全可修，**不需更强视觉** |
| 量词/范围推理（SO，把 "all" 拆逐条核验） | 11 | 可修 |
| 子句分解（CAUSAL，独立评每个子句） | 7 | 可修 |
| DEFAULT_TO_SUPPORTED 纯惰性 | 5 | 可修 |
| **真·像素读图（NUMERIC_OCR 两行数字方向比较）** | **1** | 8B-VL 受限 |

→ 43 真错里 **42 可修，仅 1 卡视觉天花板**。`image_numeric` modality 标签（32/77）有误导：逐例核查后，那些 ENTITY 错
其实只需对照表头字符串（句子写 Top-5、表里只有 Top-1），是文本接地不是 OCR。

## 3. 代表性实例（已读图核实）

- ENTITY：`clean:track1-001080:s1` 句称 "no effect on **Top-5** accuracy"，表 8a 只有 Top-1 列 → 模型被 Top-1 数字带跑判 SUP，正解 UE。
- ENTITY(testp1)：`t1:...-000334:s8` caption 只枚举子图 (a)(b)，句子引 "Fig.5**(c)**"（不存在）→ 9/10 投 UE 正确。
- SCOPE：`clean:track1-001096:s5` 把"baseline 12 层退化"外推成"所有架构上限 12 层"，而作者自己的模型撑到 45+ 层 → SO。
- CAUSAL(testp1)：`t1:...-000027:s0` 柱状图确显两剂量赢 placebo（前半对），尾句 "dose-independent **therapeutic mechanism**" 无据 → 10 模全判 UCM。
- GOLD_ERROR：`clean:track1-000064:s4` 三个 transformation 全在表里（UE 不成立）+ "may reflect a learned **mechanism governing**" → 正解 UCM，gold UE 是 UE/UCM 边界误标（对抗复核 confirm）。

## 4. gold 噪声 / 模型其实对了

dev 错例里 **≈29%（15/52）是标注噪声或本质歧义**（confirm_contested 12 + correct-实为-UNCLEAR 5，去重）。
三类确凿误标（应清洗训练集）：UCM 被误标 UE（`000064:s4`）；section-header 句误标 UE（`000797:s0` "Visualization of Edge and Attention Maps"，图里就有该列头）；caption 逐字 paraphrase 误标 UE（`000994:s1`）。
testp1 冲突 GOLD_ERROR=0 且 17/25 model_defensible → testp1 的可争议空间在"激进模型抓对、保守漏"，**可由更好召回收割**。

## 5. 可操作杠杆（按 ROI；全为纯加法召回，不碰抑制类后处理）

| # | 杠杆 | 命中桶 | 覆盖真错 | 收益 |
|---|---|---|---:|---|
| 1 | **实体缺席检测**（训练注入 negative：句中 token vs caption/表头核对；推理加"每个命名实体是否在证据中"） | ENTITY_PRESENCE | 23 | 高 |
| 2 | **复合句逐子句独立判 + UCM/因果标记词**（mechanism/governs/suggesting that→UCM 候选） | CAUSAL_CLAIM | 14 | 高 |
| 3 | **量词分解**（all/significantly/full range 拆逐条；先查实体在场排除 UE→强制走 SO） | SCOPE + SO↔UE 边界 | 16 | 中-高 |
| 4 | **去 SUP 先验**（清洗"每段≤1少数类"伪先验 + 密度匹配长段训练） | DEFAULT_TO_SUP | 5+全局 | 中（PHASE C 已过门 +2.68 densematch） |
| 5 | **清洗 gold 训练噪声**（UE↔UCM / header / caption-paraphrase 三类重标） | GOLD_ERROR | 训练侧 12 | 中 |
| 6 | OCR 数字通道 | NUMERIC_OCR | 1 | 低（仅 1/77 真需，ROI 低） |

**一句话**：召回崩塌 ≈95% 是惰性而非失明；最高 ROI = 把"实体缺席检测 + 复合句逐子句判 + 量词分解"做进训练目标
（杠杆 1-3 命中 53/77 例），与 testp1 RECALL-critical 和已过门的密度匹配方向一致；真·读图(OCR)瓶颈只占 1/77，不值得当主攻。

---
完整逐例诊断见 `data/analysis/out/diagnoses.json`；opus 综合原文见 `data/analysis/out/SYNTHESIS.md`。
