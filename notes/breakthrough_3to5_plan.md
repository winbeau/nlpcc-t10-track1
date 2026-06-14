# Track1 终局突破计划：从 50.26 再冲 +3~5（2026-06-14，38-agent ultracode workflow）

> 综合：棋盘重述 + 12 个杠杆评分 + 对抗复核（4 个 fatal/serious 击杀）+ 本地数据复核（s01/s11/s15/s16 firing 计数已对账）。诚实、校准、可执行。
> 关联：[track2_transfer_assessment.md](track2_transfer_assessment.md)、submissions_log.md、breakthrough_plan.md、gate_b_phaseB_result.md、phaseC/。

## 1. 直接回答：+3~5 现实吗？

| 目标 | 概率 | 理由 |
|---|---|---|
| **P(+3 → 53.3)** | **~25-30%** | 需主攻做出 ≥+2 单模/重聚合突破 **再叠加** 1-2 个便宜 add-only 杠杆，且 Codabench 配额够验证 |
| **P(+5 → 55.3)** | **~6-8%** | 无单杠杆历史 >+2；+5 需两条独立轴同时各兑现上限且互不抵消 = 乘积，很小 |

**+5 很难，+3 可争取。现实期望 ~+1.5~2.5（→51.8~52.8），上探 +3。把 +5 当 stretch，不为它牺牲保底。**
历史铁律：单杠杆只 +0~2（grid→union 这唯一一次 +2.5 已吃掉）→ 必须靠"主攻支柱 + 2~3 个便宜 add-only 叠加"组合。

## 2. 为什么卡在 50.26（三重墙互锁）
1. **PEM 精度墙**：50.26 是 PEM-limited，`--min-votes 1` hard-OR 最不容错，一个错句 flip 整段 1→0。s21 比 s15 净加 +90 真少数类（472 vs 382）却掉到 47.77（PEM 39.59<43.34）。**加正确 firing 不是瓶颈，PEM 精度才是。**
2. **严重 OOD**：train 83% CS → testp1 ~87% bio/physics/chem，92% 原始图。读图在最需读图的 Contra/UE 两类退化最狠。
3. **是约定不是能力**：GPT-5.5 读图远好却零样本仅 25 分；8B 的 47.80 来自微调学到的标注约定校准 → **别把 OCR/前沿模型当主攻**。

**点数缺口（每类约值 10 分）**：UE 最大池但文本不可见(lift 1.18×)、Contra R~22% 全在图里、Scope R~63% 部分文本可检(2.67×,池~540)、UCM proxy F1~0.92 但最小类(140)。长段 PEM 几何漏分（均长 8.7，29.7% 段≥10 句）；最高价值 = 长 Supported 段里被埋的那一个正确少数类 flip PEM 0→1。**注意：不在"每段多个少数类"——testp1 长段 ~0% 有≥2 gold 少数类，原 PHASE C 多少数类论点瞄准空地。**

## 3. 路线分层（含对抗裁决）

### 层 A — 主攻支柱
- **A1 Per-sentence 温度 self-consistency（纯 add-only，删门控）★最高优先**：现有模型 K=8/T=0.7 采样，纯 add-only hard-OR 合并。离线 PHASE C E1 已验证 densematch +1.40 / **PEM +1.94**。预期 +1~3（诚实 +1.5）。成本 ~6-10 H200-h + ~100 行代码 + 1 发 Codabench。**裁决：L5 原版的 ≥2-of-8+prob-margin 门控被双重 serious 击杀（s16 ≥2-vote 同族 -2.97，且"修不存在的问题"）→ 保留机制删门控；严格限 per-sentence（JOINT 上 E1b 已失败）。** 注意 L5 把两个 E1 搞混（PHASE C E1=self-consistency KEEP；CoT E1=s39 FAILED），那条 caveat 是事实错误。
- **A2 PHASE C 密度修正（收窄去合成）**：原 +3~6 不可信，现实 +0~2 高方差。**合成注入 B1 已死（densematch -2.91 PEM）；"每段多少数类"论点已死。仅留非合成 reweighting/真段落选择瞄单个被埋少数类。**
- **A3 Logit-adjusted/Balanced-Softmax（roadmap A2，从未跑）— 最被低估的 on-root-cause 目标**：5-label-token span 加闭式 margin `δ_y=-τ·log(π_y)`，零行复制不记忆（解耦 s09 -4.0 的 FP 崩）。单模 +0~1.5，作 decorrelated 成员 +0~1，P(single>47.80)~14%。**裁决：(1) regime-b 落 s31/s32 平台(34-44)别单追；(2) regime-a 须小 τ{0.3,0.5}+非对称 clip 避 s33 FP 崩；(3) 别 hard-OR 评，作 soft per-sentence blend 融进 s15；(4) 离线 proxy 假——infer.py 不存 5-way logits，终判须 Codabench 一发。**

### 层 B — 便宜叠加（each +0~1，PEM-safe by construction）
- **B1 Track2 文本-only soft-blend（唯一真正新轴）**：DeBERTa-v3 sentence-only，误差与死掉的多模态 union 去相关。`P=P_main+W·P_text`，W 在 Codabench ADD-mode 扫。预期 +0~2。**硬约束：(a) 必须 soft-blend 绝不 hard-OR；(b) 文本只看见 UCM+Scope，对 Contra/UE 证明性失明，而 UE 是最高 firing 类(155/382)→ 天花板碰不到 UE。** 融合脚本尚不存在。
- **B2 Locked-paragraph marker injection — 原版 fatal 击杀**：错-flip 是对-flip 的 ~12×（"firing 进 locked 段只能 0→1"是假的，~16% testp1 段真 all-Supported）。**仅留残存路：当多标签 OR-hit 第二标签，只加在已 firing 句（零 PEM 风险，薄 MacroF1 收益）；上 Codabench 前手工审计 strict-fire locked 段，>10% 真干净即 KILL。**
- **B3 Same-family model-soup — 仅 Phase-2 OOD hedge**：s11=union(s01+s08+s09)=48.22<50.26 已知上界；soup 是平均 → 不可能超 union 它们自己，作 s15 成员近零去相关。**别花 Codabench。**

### 层 C — 高风险（图读取口袋，都没破解）
- **C1 within-para contrastive margin — contrastive 半边 fatal**（loss.py `_per_sentence_ce` teacher-forced gold string，sibling 上 P(minority) 不可读；"复用 grouped logits"机械错误）。focal 半边可建但薄，优先级低于 A1。
- **C2 图接地路线**：VCD/CRG（L3）serious——GPT-5.5 探针证伪"图读取是瓶颈"，只跑 logit-gap probe(~1-2 H200-h) 决定；evidence-recrop（L4）serious=higher-res 换皮（s02 -3.3）不做；constrained-decode（L8）抑制半边=死的 ≥2-vote，若做严格 add-only 且禁 Contra/UE；paragraph-corrector（L12）serious=无 honest 训练数据/clean OOF=死的 s30(43.85)，若做 REMOVE-only。

### 明确已死（别碰）
更多 union 成员 / 抑制·门控·≥2-vote / 提分辨率·recrop hard-OR / 降过采样·逐类过采样·2-epoch / 段落联合作单模 / CoT-SFT / 外部数据增强 / OCR·前沿模型作主攻 / L2 原版·L4·contrastive margin（fatal）。

## 4. 执行序列（2-3 周，先零/低 GPU gate → GPU → 最后稀缺 Codabench）
- **第 1 周（零/低 GPU gate，零 Codabench）**：①A1 cheap gate（infer.py 加 `--n`/`top_logprobs`，s01 K=8/T=0.7 densematch，A/B add-only vs 门控，GO iff add-only 重现 +1.4 且新 firing >1.5× 在 base-WRONG 段）；②B1 Step-0 gate（DeBERTa sentence-only，densematch per-class any-of F1，PASS iff UCM/Scope F1>>0 且 firing 落 base-WRONG 段）；③B2 手工审计。过 1 项即 Codabench 一发验 A1。**累计 → ~51.5-52.5。**
- **第 2 周**：④A3 Stage-A（regime-b clean → 过则 regime-a 小 τ+非对称 clip）；⑤建 B1 融合脚本，W 扫一发；⑥A2 非合成 reweighting（若 A1/A3 未达 +2）。**A3 作 soft 成员 + B1 兑现 → ~52.5-53.5（P(+3) 主路径）。**
- **第 3 周**：⑦层叠 add-only 杠杆（s15 + A1 + A3-shifted + B1），每加一层验 PEM 不破 43.34；⑧C2 logit-gap probe 决定是否上 VCD。**全兑现上限(小概率) → ~54-55；现实 ~52-53。**
- **Codabench 预算 ~5-7 发**；densematch 只做 same-regime 相对排序，绝不跨 regime 外推。

## 5. 风险 + 兜底
头号风险：**A1 densematch 重现 +1.4 但 testp1 不迁移**（densematch 只在 clean train' regime 验过；s30 前例 densematch #1 却 testp1 PEM 崩 39.08）→ 必花一发 Codabench 确认。其他：A3 regime-a 重入 s33 FP 崩；B1 只摸 UCM+Scope 对 UE 失明；便宜叠加互相抵消饱和在 +1.5。
**兜底铁律：s01=47.80 / s15=50.26 永久保底，新尝试 ADD-mode only 绝不覆盖。** 3 周后无 >50.26 干净点 → 保 s15 作 Phase-1 终交，model-soup 当 Phase-2 OOD hedge。Phase-2：OOD 杠杆（文本 blend/soup）P1 权重高、P2 权重低，别为 testp1 过拟合伤 testp2(CS in-domain)。

## 6. 本周就做（最高 priority × 最便宜 gate，零 Codabench）
- **① A1 纯 add-only self-consistency cheap gate**：infer.py 加 `--n`+`top_logprobs`（当前硬编码 temp=0 / top_logprobs=1）；s01 K=8/T=0.7 densematch，A/B add-only vs 门控。唯一离线已验证 PEM 涨(+1.94)、结构上不可能退化成 s16、可层叠。
- **② B1 文本通道 Step-0 gate**：DeBERTa-v3 sentence-only 训 devbench/train_sft，densematch per-class any-of F1。唯一在"正确的轴"上（证据无关语言指纹）、PEM-safe 可迁移；融合脚本本周建。

**关键文件**：`src/nlpcc_t10/infer.py`(A1 加 `--n`/`top_logprobs`，当前 temp=0)、`scripts/ensemble_union.py`(`--min-votes 1` 复用)、`src/nlpcc_t10/swift_softmin/loss.py`(A3 复用 `restrict=`；C1 在此证不可建)、`data/devbench/`(densematch 唯一干净 bench)、B1 融合脚本(待建)。
