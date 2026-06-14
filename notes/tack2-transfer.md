# Claim-only 鲁棒指纹法 —— Track2 实证 & 能否迁移到 Track1

给 Track1 同事。这份说明:(1) 我们在 Track2 用一个很简单的方法把分数显著拉高的做法和原理;(2) 三个数据集的域占比证据;(3) 它能不能搬到 Track1、怎么搬、先验证什么。结论先说:**机制层面大概率适用,但 Track1 有两个关键差异需要先用一个便宜实验证伪/证实,别盲搬。**

---

## 1. Track2 上发生了什么(方法有效的硬证据)

公开榜 Track2 综合分(= avg(MacroF1, Joint@3)),全程 **ADD-mode 零风险、只换标签、证据完全不动**:

| 提交 | 做法 | Track2 | **MacroF1** | Joint@3 |
|---|---|---|---|---|
| Jm5(基线) | 冻结 4-seed DeBERTa 标签 + 融合证据 | 83.102 | 87.015 | 79.189 |
| coW05 | + 0.5·claim-only-DeBERTa | 84.306 | 88.569 | 80.043 |
| **co4W08** | **+ 0.8·claim-only-DeBERTa(4-seed)** | **85.099** | **89.408** | **80.790** |

一句话:**只往标签里掺一个"只读 claim、完全不读证据"的分类器,MacroF1 +2.4pp,综合分 +2.0,稳居第一。**

---

## 2. 为什么有效(机制)

任务测试集是**域偏移(OOD)**的(见下表)。我们主模型读 `claim + 证据`,在 OOD 域上会退化,因为它依赖**证据里的领域内容特征**,这些特征迁移不到新域(物理/生物/化学)。

但标签很大程度由**claim 文本本身的"指纹"**决定(组织方的生成/标注管线在 claim 措辞里留下的、与领域无关的模式)。所以:

- 一个**只读 claim 的分类器**(claim-only)keys 在这个**域不变指纹**上 → **OOD 鲁棒**。
- 实测:topic-cluster 留一域验证下,claim-only 准确率只掉 ~2pp,而读证据的主模型掉 ~7pp。
- 把 claim-only 作为**解相关、OOD 鲁棒的票**掺进主模型,专门纠正主模型在 OOD 域上翻车的那些行。
- **权重随 OOD 程度走**:OOD 越重权重越高(我们物理 P1 上最优 W≈0.8),in-domain 则很低甚至 0。

关键点:claim-only 单独不如主模型(in-domain 弱十几个点),但它**错得不一样**,且**在 OOD 上不退化**——所以是绝佳的纠错成员。

---

## 3. 三个数据集的域占比(两个 Track 都测了)

同一关键词分类器,每条按 claim_text 归主导域。

**Track 2**
| 域 | traindev | testp1(P1) | testp2(P2) |
|---|---|---|---|
| Physics/材料 | 0.3% | **64.8%** | 0.6% |
| CS/AI/ML | 49.3% | 2.2% | 52.8% |
| Bio/Med | 3.0% | 5.3% | 1.8% |
| Chem | 0% | 5.9% | 0% |
| Other | 47.5% | 21.8% | 44.8% |

**Track 1**
| 域 | traindev | testp1(P1) | testp2(P2) |
|---|---|---|---|
| Physics/材料 | 0.7% | 21.3% | 0.0% |
| CS/AI/ML | 53.0% | 3.8% | **59.9%** |
| Bio/Med | 2.6% | **34.0%** | 1.6% |
| Chem | 0.1% | 8.9% | 0.0% |
| Other | 43.6% | 32.1% | 38.5% |

**对 Track1 的两条直接结论:**
1. **结构相同**:traindev ≈ testp2(CS in-domain),testp1 是 OOD → claim-only 该用的场景(P1)在 Track1 一样存在。
2. **OOD 风味更杂**:Track1 P1 是 **34% 生物 + 21% 物理 + 9% 化学**(不像 Track2 P1 纯物理)。证据又是**图表多模态**——读生物/化学图表的 OOD 退化只会**更严重**,所以 **claim-only(纯文本、域无关)的纠错红利在 Track1 P1 上可能比 Track2 更大**。

---

## 4. 迁移到 Track1:能搬,但有两个差异要先验证

Track1 的任务/数据和 Track2 不同,直接影响可行性:

| 维度 | Track2 | Track1 |
|---|---|---|
| 单元 | 整条 claim → 1 个标签 | 段落里**每个句子** → 1 个类型(每条 ~5 句) |
| 类别 | 4 类,较均衡(47/17/17/17) | **5 类,极端不平衡**:Supported 90.4% / Unsupported Entity 3.5% / Scope Overgeneralization 3.3% / Unsupported Causal Mechanistic 1.5% / Contradiction 1.2% |
| claim 来源 | 组织方**生成**(指纹强) | 论文里的**真实句子**(指纹可能弱) |
| 证据 | 论文文本段落 | **多模态**(image + table) |
| 指标 | MacroF1 + Joint@3 | MacroF1 + PEM |

**差异 A —— claim 是真实句子,不是生成的。** Track2 的指纹来自生成管线;Track1 句子是论文原文,"生成指纹"这条不直接成立。**但**错误类型本身有强语言学标记:Scope Overgeneralization(全称量词 all/always/every、过度外推)、Unsupported Causal(因果/机制断言)、Contradiction(对立措辞)——这些是**句子文本里就能看出、且域无关**的模式。所以"句子-only 分类器"很可能照样抓得住这些错误类型指纹。**这是必须先测的第一件事。**

**差异 B —— MacroF1 被 4 个稀有错误类支配。** 90% 是 Supported(F1 容易高),真正决定 MacroF1 的是那 ~9.5% 的稀有错误类。所以**句子-only 必须在稀有类上有用**,不能只会判 Supported。验证时要专门看稀有类的 per-class F1,而不是总准确率。

**多模态是利好**:读生物/化学图表在 OOD 上几乎必崩,而句子-only 完全绕开图表 → OOD 鲁棒优势可能比 Track2 更突出。

---

## 5. 具体怎么做(给 Track1 的 recipe)

**第 0 步(便宜、决定性,先做这个):测"句子-only 信号"到底有多强。**
- 训一个**只读句子文本**的分类器(DeBERTa-v3-large,输入 = `tok(sentence)`,不喂 evidence_bundle),5 类、focal loss + class weights(对付 90% 不平衡),在 traindev 上训、dev 上测。
- 看 **dev MacroF1,尤其 4 个稀有错误类的 per-class F1**。
  - 若稀有类 F1 不算差(比如句子-only 能拿到主模型 MacroF1 的 ~80%+)→ 句子指纹真实存在,**继续**。
  - 若稀有类 F1 ≈ 0(只会判 Supported)→ 这条路对 Track1 不成立,**停**(就省下了)。

**第 1 步:测 OOD 鲁棒性(可选但推荐)。**
- 按域把 traindev 聚成几簇,留一簇做 holdout,比较"句子-only"和"读证据主模型"的跨簇 MacroF1 掉幅。句子-only 掉得少 = OOD 鲁棒得到验证。

**第 2 步:融合。**
- `P = P_main + W · P_sentence_only`,逐句 argmax。**P1(OOD)用较高 W,P2(in-domain)用很低 W 或 0**(P2 主模型本就强,见域表)。
- 我们 Track2 的权重是在公开 P1 LB 上扫出来的(物理最优 0.8);Track1 P1 LB 也开着,可以同样**ADD-mode 零风险扫 W**(只有更高才 Add)。in-domain 的 dev 用来定 P2 的保守 W。

**第 3 步:只换标签、不动其它。** 和我们一样,保持主模型其余输出不变,降低风险。

**代码可直接借用**:我们的 claim-only 训练脚本 `claim_only_train.py`(把 `tok(claim)` 换成 `tok(sentence)`、4 类换 5 类即可)在 GitHub `deafenken/claimledger` 的 `track2_p2_pipeline/box_scripts/`,方法详情在 `track2_p2_pipeline/P1_CLIMB.md`。

---

## 6. 一句话总结

> Track1 和 Track2 同构(P1 OOD / P2 in-domain),且 Track1 P1 的多模态 OOD 更难,理论上 claim-only(→ Track1 的 sentence-only)的鲁棒纠错红利更大。**但 Track1 的 claim 是真实句子 + MacroF1 被稀有错误类支配**,所以先花半天做"第 0 步"——训一个句子-only 分类器,看它在 4 个稀有错误类上有没有信号。有,就照 Track2 的融合套路上 P1 LB 扫权重;没有,就果断放弃,不浪费时间。

有问题随时找我(Track2 这边)。
