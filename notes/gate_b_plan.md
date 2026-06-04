# GATE-B 离线评测台 — 实现计划(下一步主线)

> 2026-06-04。**这是当前第一优先任务**。背景见 [[eval_bench_design]](原理+对抗核实) / [[breakthrough_plan]](为什么它是破局前置) / [[data_augmentation_research]](已否决的方向)。

## 一句话目标
把"每个想法盲烧 1 发 Codabench"换成"离线先筛、过了再交",10× 迭代。

## 原理(三步,缺一不可)
1. **无泄漏精确切分**:按图哈希 union-find 切(component_aware),整组进 train' 或 dev',保证 dev' 的图模型没训过(消除 32% 图泄漏)。
2. **重塑成 testp1 形状**:只留长段(≥8句)+ 下采样使 ~84% 段含 ≥1 少数类(匹配 testp1 段长/少数类密度直方图)。因为 PEM 是段落级、行为在"长密"vs"短稀"上天差地别。
3. **用 23 锚点验证排序**:**不信绝对分(域偏移=不同论文,消不掉),只信排序**。台子可信 ⟺ 它给 23 个已知 testp1 真分的提交打的排序 ≈ testp1 排序(Spearman)。

## 关键实测信号(来自 evalbench-design workflow,可直接用)
- Spearman(MF1, testp1_score)=**+0.980**、Spearman(PEM, score)=**+0.945**、少数类总数=+0.837(union 家族内塌到 +0.150)。→ 台子只要复算 MF1+PEM 排序就近乎定死 score 排序。
- 免费 gold-free 信号:与 s15 句级一致度 Spearman=+0.872(可粗筛,但定不了 union 拐点)。
- 少数类投票双峰:5096 位中 4582 位 0 票、189 lone、87 全票。

## PHASE A — 建台数据(本地纯 Python,零 GPU/零配额,先做)
**A1**:`uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data/devbench --split-mode component_aware --val-ratio 0.15 --seed 42`
→ 产出 `data/devbench/{train_sft,dev_sft,dev_gold,split}.jsonl`。验收:零-sha 断言通过(build_dataset.py:317)、dev ~500 record、打印 dev 段长&少数类密度直方图。
**A2**:写 `scripts/reshape_devbench.py`(~50行,无 torch),从 dev_gold 选 record 产:
- `dev_gold_raw.jsonl`(无泄漏原形状,对照)
- `dev_gold_reshaped.jsonl`(长段≥8 + 下采样 long-easy 使 ~84% 含少数类;testp1 段长中位 8)
→ 验收:打印重塑后 vs testp1 目标(段长直方图、少数类密度)贴近。
**A3**:commit 脚本 + shape 报告(台数据本身 gitignored)。

## PHASE B — 验证可信(远端 GPU,A 之后)⚠️ 老 adapter 有泄漏,只当粗筛信号
- B1 盘点 adapter 存活(HF winbeau/nlpcc2026-task10 有 8 Qwen;远端 outputs/ 有 gemma s26/s27)。
- B2 对可复现 base(5 Qwen s01/08/09/12/13 + gemma26/31)在 dev_sft 上重跑 → dev raw(含 min_logprob)。
- B3 本地合成 union 锚点 dev 预测(ensemble_union 吃成员 dev 预测)。
- B4 每锚点 × 每 gold 变体跑官方 evaluate.py → (dev_MF1,dev_PEM,dev_score)。
- B5 写 `scripts/rankcorr_meta.py`(纯 numpy,scipy 不在,手写 Spearman):Spearman(dev_score,testp1_score)+分量+union 拐点(台能否排 s15 > s18-s29)。
- B6 判据:≥0.8 且排序对 → 可信选型台;0.6-0.8 → 只当"杀坏方向"粗筛器;<0.5 → 转纯累积协议。
- ⚠️ 老 adapter 在全量(含 dev')上训过 → 对它们 dev' 泄漏 → B 是粗略 sanity,不是干净验证。

## PHASE C — 锁累积协议(干净长期解)
写进 CLAUDE.md §9:**今后每候选 = 在 data/devbench 的 train' 训 + dev_gold_reshaped 评 + 同时交 testp1**,累积 (dev,testp1) 对,rank-corr 越攒越实、天然无泄漏。**第一个干净点 = 密度匹配模型(本就在 train' 训)→ 评测台与破局主攻合一**。

## 交付物
`data/devbench/`、`scripts/reshape_devbench.py`、`scripts/rankcorr_meta.py`、rank-corr 报告、CLAUDE.md §9 规程。

## 验收门(统计功效诚实声明)
~500 record 的台**分不清 50.0~50.3 的细微 union 取舍**(噪声内),但**能可靠杀大动作坏方向**(提分辨率/降采样/抑制后处理)→ 省一半盲交。顶尖取舍仍各交 1 发定。

---

## 状态 (2026-06-04):PHASE A 完成 ✅,PHASE B 待 GPU

**A1 ✅** `build_dataset --split-mode component_aware --val-ratio 0.15 --out data/devbench` → 2832 train / **501 dev** record,零-sha 断言通过。
**A2 ✅** `scripts/reshape_devbench.py`(纯 stdlib)产 3 个 gold 变体 + `notes/devbench_shape_report.md`:
- `dev_gold_raw`(501,对照,traindev 形状)/ `dev_gold_reshaped`(30,长≥8+密 84%,testp1 regime 探针)/ `dev_gold_densematch`(206,密度 84% 无长度地板,高功效)。
- **关键验证(trivial all-Supported 试金石,官方评测器)**:raw=**42.37**(PEM 65!=抑制陷阱,会奖励已证伪的压少数类);reshaped=**17.88** / densematch=**17.24** ≈ testp1 trivial **18.85**(s05)。→ **reshape 成功把台子从「奖励抑制」翻成 testp1 同款「召回-critical」regime**。
- ⚠️ **池子天花板坐实**:image-disjoint dev 只有 ~25 个 record 同时「长(≥8)+ 含少数类」→ 严格 reshaped 注定 ~30 record(低功效,只能定方向不能排 union 顶部)。**正是 PHASE C 密度匹配训练数据=干净长期解的理由。**
**B5 写在前(ready,未跑)** `scripts/rankcorr_meta.py`:手写 Spearman(selftest 过)+ 内嵌 24 个锚点 testp1 分(已对账 submissions_log.md 零误差)+ 端到端 plumbing 已用合成 dev 预测 smoke 过(rho 方向正确)。**跑它只差 PHASE B 的 dev 预测(需 GPU)。**

**PHASE B 起跑清单(下次上 GPU)**:在 `data/devbench/train_sft.jsonl`(=train', image-disjoint 无泄漏)重训/或重跑可复现 adapter → 对 `data/devbench/dev_sft.jsonl` 推理 → `aggregate` 出每锚点 `outputs/devbench_preds/<sNN>.jsonl` → 本地 `ensemble_union.py` 合成 union 锚点 → `rankcorr_meta.py --gold dev_gold_{reshaped,densematch,raw}.jsonl --pred-dir outputs/devbench_preds`。⚠️ 旧 adapter 在全量(含 dev')训过=对它们 dev' 泄漏,B 是粗 sanity;干净点 = 在 train' 重训的新模型(与 PHASE C/P1 合一)。
