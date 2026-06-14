# Track2 sentence-only 迁移评估 —— 能否 +5 分？（2026-06-14，11-agent workflow）

> 对照 `notes/tack2-transfer.md`（Track2 同事建议：掺一个「只读句子文本、不读证据」的分类器做去相关 OOD 纠错成员）。
> 本评估全部基于本地纯-Python 实测（traindev/testp1 jsonl + devbench 代理实验），无 GPU。

## 结论（一句话）

**大概率不到 5 分。P(≥+5) ≈ 0.08；现实涨幅 +0~+2（须过 Step-0 且用 soft-blend）。但 Step-0 验证便宜且决定性，值得做——定位成「便宜探针 / +1~3 的赌注」，不是「+5 的把握」。**

## 关键实测证据

### 1. text-only 只看得见 4 个少数类里的 2 类（signal probe，traindev 17547 句）
| 类 | marker lift | text-only F1（估/代理） | 判定 |
|---|---|---|---|
| **Unsupported Causal Mechanistic** | **5.08×** | **0.80–0.92** | 强（~70% 句面含 mechanism/thereby/induces）|
| **Scope Overgeneralization** | **2.67×** | **0.46–0.70** | 中（all/every/across-all，功能词、域不变）|
| Contradiction | 1.10× | 0.00–0.20 | **几乎不可见**（标记词在 Supported 同频）|
| Unsupported Entity | 1.18× | 0.08–0.20 | **几乎不可见**（实体「缺席」必须读证据）|

代理 Step-0（TF-IDF 1-2gram + class-balanced LogReg，训 `devbench/train_sft`、评 `densematch`，0 record-id 泄漏）实测：UCM F1≈0.92、Scope F1≈0.46、Contra F1=0.00。**不塌缩到全-Supported**。
→ UCM/Scope 是**功能词指纹、域不变**，可见；Contra/Entity 本质是「句子 vs 证据」关系，**单看文本不可判**。

### 2. testp1 确实严重 OOD（机制迁移前提，成立）
| 域 | traindev | testp1 |
|---|---|---|
| CS/AI/ML | 82.96% | 8.02% |
| Bio/Med | 1.3% | 33.45% |
| Physics | 2.0% | 35.84% |
| Chem | 1.0% | 17.58% |
训练 83% 在 CS，testp1 ~87% 在 bio/physics/chem。这正是 text-only 卖点（MM 读证据在 OOD 最弱处，功能词指纹最强）。也是双刃：少数类监督 83% 来自 CS 域。

### 3. 是真正的新轴，但「新轴 ≠ 破 50.26」
- **为什么新**：之前死掉的 union 成员（q4b s18 / InternVL s19-21 / Gemma s23/s28/s29 / 31B s27）**全是多模态、条件于同一证据通道**，误差相关、签名一律「MF1↑/PEM↓」。text-only 键于**证据无关的语言指纹**，误差去相关 + OOD 稳健——第一次在「正确的轴」上区别于死亡堆。
- **为什么仍可能破不了**：50.26 是 **PEM 精度墙**不是召回墙。`ensemble_union.py --min-votes 1` 是**硬 OR**（lines 66-77：任一成员开少数类即采纳、无权重无弃权），精度最不宽容——一个误开少数类把整段 PEM 1→0。text-only 改了相关性，没改 hard-OR 的残酷不对称。

### 4. 对抗复核：LBC2「text-only firing 高精度」被双双判 refuted
该高精度**从未被测量**（正是 Step-0 要测的量）；in-distribution 代理 F1 被域内容记忆**夸大**，OOD 下更差；text-only **对 Contra/Entity 可证盲**，而 **Entity 是 s15 firing 计数最高类之一（155/382）**——最能撬动 Macro-F1 的类恰恰看不见。净增益要求 firing 不成比例落在 base-WRONG 段（PEM 0→1），未验证。

## 决定性下一步：Step-0（一次性、不烧 Codabench 配额）
- **模型/输入**：DeBERTa-v3，输入 = `tok(sentence)` only（不喂证据/claim 前缀），focal loss + class weights。
- **训** `data/devbench/train_sft.jsonl`（image-disjoint train'）→ **评** `data/devbench/dev_gold_densematch.jsonl`（206 rec / 1235 句，0 泄漏），官方 any-of credit per-class F1。
- **门**：
  1. 首要门 = UCM、Scope per-class F1 >> 0（代理已给 UCM≈0.92/Scope≈0.46，DeBERTa 应≥此）。塌缩到 ~0 → 停。
  2. 真正的门 = **minority PRECISION 限定在「5-Qwen base 预测 Supported」的位置**（blend 唯一改标签处）。一个 FP 杀一整段 PEM → 新 firing 必须 mostly-correct。
  3. 去相关验证：正确 catch 不成比例落在 base-WRONG 段（PEM 0→1）。
- **集成（务必）**：**不要**丢进 `--min-votes 1` 硬 OR（那正是杀死 s18-s29 的配置）。移植 Track2 **soft-blend**：`P = P_main + W·P_text`，逐句 argmax，W 在 LB 上 ADD-mode 扫（只升才提交）。代码库**还没有**这个机制（需 Qwen + text-only 的 per-class 概率 + 新 soft-fusion 脚本）。
- ⚠️ densematch ρ 仅 +0.59（leaky 下界，分不出 50.0–50.3）→ Step-0 过关也无法离线证明 >50.26，最终判定要花**恰好一发** ADD-mode Codabench。s01=47.80 / s15=50.26 永久保底。

## 期望管理
Track2 自己只 +2.0（精度宽容的 soft 加权平均、4 类近均衡、Joint@3 宽松）。Track1 更难（5 类极端失衡、PEM 全段全对最脆、text-only 只贡献 2/4 类、Qwen base 过自信 min-logprob~−0.003 → soft-blend 易退化成 hard-OR）。但 Track1 自己 union 做出过 +2.46（s01→s15），证明 PEM 增益可达。**最可能结局：Step-0 过关 + soft-blend → +0~+2；丢进 hard-union → 净负。**
