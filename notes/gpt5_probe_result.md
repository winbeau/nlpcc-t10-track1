# gpt-5.5 前沿 API 探索结果 —— 能力不是瓶颈,惯例才是(OCR/读数路证伪)

> 2026-06-04。回应"用 OCR/前沿模型读图破局"的假设。工具:`scripts/gpt5_infer.py`(gpt-5.5 via aiapis.help 代理,规则明确允许商业 API)。**结论:停掉 gpt5 分类路,聚焦训练(惯例+召回)。**

## 做了什么
gpt-5.5 零样本「读证据图 + CoT 推理 → 5 类」,逐 record(joint)。验证了:
- **文本/视觉都通**(需浏览器 User-Agent 绕 Cloudflare 1010;图路径 `<DATA_ROOT>/data/images/<sha>.jpg`)。
- **它真在读图里的数字**:raw 里能引 "clean 82.6 vs 85.6 / PGD 45.3 vs 45.4"、折线 "quality3 处 adaptive≈0.19 vs fixed≈0.26"、"横轴只到 45 不是 >60%"——**这正是我们 8B 读不出的能力**。

## densematch dev(有 gold)上的分(trivial 全-Supported≈17.24)
| prompt | score | 句准 | 备注 |
|---|---|---|---|
| 裸激进(少数类常见/别默认Supported) | 13.1 | 30% | <trivial,狂喷少数类 |
| 重标定(Supported 先验~85%) | 22.2 | 58% | 略过 trivial |
| + 6 条 few-shot(真 gold 样例) | **25.2** | 60% | **仍 ≪ 8B 单模 47.8 / union 50.26** |

**逐类(few-shot 后 vs gold)**:Supported 59% vs **85.7%**;UE **19% vs 2.9%(过开6.5×)**;UCM **13% vs 1.6%(过开8×)**;SO 4% vs 7.7%(欠开);Contra 5% vs 2.1%。**few-shot 几乎没撼动分布。**

## 🔑 durable 结论
1. **任务是「标注惯例约束」,不是「能力约束」**:gpt-5.5 读数远超我们 8B,得分却只有一半——**因为 8B 从 1.7 万样本微调学到了"什么算 UE/UCM/SO 的边界"(所以 47.8),gpt5 零样本没有(所以 25)**。
2. **「读不清图 / OCR / 读数」是伪瓶颈,别再投入**。真瓶颈 = **惯例校准 + 少数类召回**,这是**训练**给的。
3. **闭源前沿 API 当分类器 = 死路**:惯例靠 prompt/few-shot 补不上,只能微调,而闭源 API 不能微调。
4. **残值**:gpt-5.5 仅可"当读数器喂 8B"(additive 通道),但 EV 已降低(读数非主瓶颈),**优先级低于密度匹配**。

## → 下一步(回到 breakthrough_plan 主线)
**全力 P1 密度匹配训练**:修「≤1 少数类/段」伪先验、提召回,在 `data/devbench/train_sft.jsonl`(image-disjoint train')训 → `dev_gold_densematch.jsonl` 离线验(PHASE C 累积协议)+ 交 testp1。gpt5 探索**反向确认了这条主线**(能力够、缺的是训练出的惯例)。
保底 s01(47.80)/s15(50.26) 不动;gpt5 的 s30 **未交**(25 ≪ 保底)。
