# 离线选型台(RANK-BENCH)设计 + 两条硬约束

> 2026-06-04,multi-agent workflow(9 agent,Mine 阶段在 23 锚点上实算 + 2 轮对抗核实)。回应"建可信 testp1-shaped 无泄漏评测台 10× 迭代"。

## Mine 阶段实测信号(金子,可复用)

在 23 个已评分提交上算:
- **MF1 是最强 score 预测器**:Spearman(MF1, score)=**+0.980**;Spearman(PEM, score)=**+0.945**;Spearman(少数类总数, score)=+0.837。→ 台子只要能复算 MF1+PEM 的排序,就几乎定死 testp1 score 排序。
- **union 家族内,少数类数↔score 彻底崩**:Spearman 仅 **+0.150**(s23 少数类最多却比 s15 低)。**PEM 才是 union 拐点的判别器**。
- **s23 比 s15 多的 116 个少数类 = 100% gemma 单独贡献的 lone-vote**,把 PEM 砸掉 −1.36(量化坐实)。
- 投票分布双峰:5096 位中 4582 位 0 票、189 位 lone(1票)、87 位全票 → 天然的置信分层。
- **免费 gold-free 信号**:与 s15 句级一致度,Spearman(agree_with_s15, score)=**+0.872**(意外强,可粗筛;但对 s23/s26 失灵,定不了 union 拐点)。

## 推荐台 = RANK-BENCH
image-disjoint(component_aware_split,零 sha 泄漏)dev,产 3 种 gold 形态(raw / testp1-reshaped 长段密少数类 / para-resamp),**用 23 锚点的 Spearman(dev_score, testp1_score) 挑哪种形态可信**——不看绝对分(域偏移让绝对分无意义,GATE-A 当初的 81≠50 是 absolute red herring,从没测排序)。否决伪标台(非循环 ρ≤0.25;循环 ρ≈0.9 但判反 s15>s23 拐点、且不如免费 NULL)。

## ⚠️ 两条对抗核实加的硬约束(都 weakened,不是否决)

1. **统计功效不足以定 union 拐点**:~400-500 dev record 上,顶部 union 成员的差距(s15 50.26 vs s23 49.83 = 0.43)**窄于台子的置信区间** → 台子**无法可靠分辨 union 顶部取舍**(恰是我们最想要的)。**但能可靠杀大动作坏方向**(提分辨率/降过采样/抑制后处理,都是多分效应)。→ 台子定位 = **杀坏方向(省一半盲交)**,不是定 union 冠军。

2. **旧 adapter 在新切分上有泄漏**:23 锚点的模型是在**全量 traindev**(含新切出的 dev')上训的 → 对它们 dev' 不是 held-out → **重跑旧 adapter 评分 = 泄漏**。要干净验证必须**在 train'(=traindev−dev')上重训**(GATE-A 式纪律),或走累积协议。

## 结论:台子值得建,但务实定位
- **能做**:离线杀掉已知掉分的大动作(省盲交预算 ~一半)、用 agree-with-s15 + MF1/PEM 复算做粗筛。
- **不能做**:可靠分辨 50.0~50.3 区间的 union 顶部取舍(仍需各烧 1 发 Codabench)。
- **最干净的形态 = 累积协议**:今后每个候选【在 train' image-disjoint 上训 + 在 reshaped dev' 上评 + 同时交 testp1】,累积 (dev_score, testp1_score) 对 → rank-corr 自然建立,不依赖旧 adapter 是否还在、无泄漏。

## 立即可做(本地纯 Python,零 GPU,零配额)
- **STEP 0**:`build_dataset --split-mode component_aware --val-ratio 0.15 --seed 42 --out data/devbench`(务必显式 component_aware,默认 random=泄漏)。核对零-sha 断言 + dev ~500 record。
- **STEP 1**:写 `scripts/reshape_devbench.py`(~50 行,不碰 torch):产 raw / reshaped(选 record 匹配 testp1 段长直方图+~84% 含少数类)/ para-resamp 三个 gold 变体。
- 之后(1 次 GPU 过夜)按累积协议重训/重跑 base adapter 拿 dev 预测 → `rankcorr_meta.py`(纯 numpy,手写 Spearman)挑可信变体。

## 与其它结论的衔接
- 这台子是 [[breakthrough_plan]] 里 P1 密度匹配训练的**验证器**:密度匹配单模能否在长段多 catch 真少数类、且 PEM 不跌,正好用 reshaped dev' 离线看(且密度匹配本就要在 image-disjoint train' 上训 → 天然无泄漏,绕开约束2)。
- 数据增强([[data_augmentation_research]])已否决,省下的力气投这里。
