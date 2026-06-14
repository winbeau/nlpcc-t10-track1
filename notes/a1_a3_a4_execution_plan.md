# NLPCC-2026 Task10 / Track1 — A1 → A3 → A4(GRPO) 统一执行计划

> 本文档是 **唯一可执行作战计划**，串联三条加法杠杆：A1（逐句温度自洽）→ A3（logit-adjusted margin）→ A4（GRPO）。
> 所有训练/推理在 **H200 GPU 服务器**（tmux `nlpcc-t10-track1-h200`，`USE_HF=0` 必设）。本地只读数据、写代码。
> 铁律见 §1.3，永不违反。**离线 densematch 永远只判方向/粗 lift，绝不判 50.0-50.3 区间的绝对落点（Spearman +0.59，分辨力不足）——所有 sub-point 的绝对分判定一律落到 Codabench。**

---

## 1. 目标与现实预期

### 1.1 基线与目标

| 锚点 | testp1 score | PEM | MF1 | 配方 |
|---|---:|---:|---:|---|
| **s01**（单模永久保底） | **47.80** | 42.83 | 52.77 | full-traindev, softmin grid b5_l0.5, os3.0/ds0.6, max_pixels 401408, 1ep, lr1e-4, lora r16/a32 |
| **s15**（union 永久保底，当前最优） | **50.26** | 43.34 | 57.17 | 5 个差异化成员 `ensemble_union.py --min-votes 1` hard-OR |
| s11 | 48.22 | 41.64 | 54.80 | union(s01+s08+s09) |
| s14 | 49.99 | **56.80** | **43.17** | union(s01+s08+s09+s13)；⚠️ union 段 PEM>MF1（与 s01 列序相反） |

- **目标**：从 50.26 朝 **+3~5** 推进，现实落点 **~52-53**。
- **概率（独立估计，历史单杠杆只给 +0~2）**：P(+3, 即 >53.3)~25-30%；P(+5, 即 >55.3)~6-8%。
- **单杠杆 expected gain**：A1 +1~2（PHASE-C 离线 **plain-CE per-sentence** 验证 +1.40 score / +1.94 PEM，见 §3.0）；A3 P(single>47.80)~14%；A4 P(single>47.80)~12%、P(有用 union 成员)~15-20%、combined P(win)~20%。
- **组合路径累计预期（核心策略）**：单杠杆会饱和（CLAUDE.md 明确「single levers only +0-2 → must COMBINE」）。最高 EV 路径 = **把三个去相关杠杆叠进 s15 union**：
  - A1 嵌进 s15（每成员自洽再 OR）→ 目标 50.5-52.0；
  - A3 作为 **soft-blend** 第 6 成员 → 目标 50.0-52.5；
  - A4 GRPO ckpt 作 hard-OR 第 6 成员（若 L0/G3 过）→ 目标 >50.26。
  - 三者去相关召回若部分不重叠 → 现实合并落点 **52-53**；若高度重叠 → 撞 naive-union 天花板（s15=50.26 即此天花板）。每步必查 raw 重叠（新正确 firing 是否相互独立）再决定叠不叠。

### 1.2 离线评测台与泄漏铁律（决定整套门控的可信度）

- **唯一干净离线台 = densematch**：`data/devbench/dev_gold_densematch.jsonl`（**206 rec**，已核对；与 train' image-disjoint）。trivial 全-Supported ≈ 17.2（对照 testp1 trivial ≈ 19）→ 召回-critical 性质一致。
- densematch ↔ testp1 **Spearman 仅 +0.59**，**无法分辨 50.0-50.3 区间**，且 CS in-domain → **不能测 OOD-robustness**。
- 🔴 **泄漏陷阱**：s01/s08/s09/s15 在 **full-traindev**（含 densematch 记录）上训 → densematch 对它们 **泄漏**（虚高，例：s30 densematch 83.90 但 testp1 43.85；s10 dev 98 但 testp1 41.7）。
- ⇒ **门控规则（贯穿 A1/A3/A4，三条）**：
  1. **门必须跑在 train'-only adapter 上**（A1=plain-CE per-sentence train'；A3=devbench A3 train' regime-a；A4=见 §5 的 L0/G3 泄漏处理），测 **相对增益**（lift vs greedy / vs s31 floor）；lift 是 regime-internal、可迁移。
  2. **门-adapter 配方必须 == 生产-adapter 配方所对应的可迁移量**：A1 的 +1.4 证据来自 plain-CE per-sentence，**门必须用同 lineage 的 plain-CE train' adapter**（绝不用 softmin cotE0/s40，见 §3.0 FATAL 修正）；然后把验证过的旋钮 APPLY 到 full-traindev 生产模型，**并在生产模型自身上再测 within-model 的 greedy→union 相对 lift**（§3.5b），双重确认迁移。
  3. **绝不**把 full-traindev adapter 的 densematch 绝对值当 testp1 预测器读。
- 其它 gold：`dev_gold_raw.jsonl`（501 rec 全 dev raw）；`dev_gold_reshaped.jsonl` = **0 字节（空），禁用**。

### 1.3 铁律（inviolate）

1. **s01=47.80 / s15=50.26 永久保底**，任何东西都不覆盖它们；所有新尝试 **ADD-mode only**——只在「击败当前最优」时才提交。floor 文件名 `testp1_s01_m_q8b-grid.*` / `testp1_s15_u_q5.*` 永不复用、永不重写。
2. 若平台按最新提交计分，实验后必须把 s01(47.80)/s15(50.26) 顶回保底位。
3. fallback：解析失败/越界标签 → `Supported`（保 PEM）。
4. **抑制少数类的后处理在 testp1 单调掉分，禁用**；`--min-votes 1`（纯并集）> `--min-votes 2`（s15 50.26 vs s16 47.29）。
5. **每个产生新 Codabench 上传的配置变化都必须占一个新 s-号 + 一行 `notes/submissions_log.md`**（序号/类型/描述/score/MF1/PEM/少数类数）+ 复现命令 + 为什么试它。**禁止在已编号提交上原地改 decode/(T,K)/成员组合而不升号**（见 §2 末「ledger 纪律」）。
6. 命名：`testp1_s{NN}_{T}_{desc}.{zip,jsonl}`，`T`∈{m,u,pp}，desc 内 `-`、`_` 只分三段；zip 内顶层文件固定 `track1_pred.jsonl`。
7. **决策树里所有 50.0-50.3 的细分一律是 Codabench-only 决策**（离线台分辨不了）；离线只给二元粗判：lift 在 score+PEM **双metric** 上清晰为正且 firing-集中 ratio≥1.5 → 烧枪，否则 KILL。

---

## 2. 提交编号台账（s41 起，三阶段统一序列）

> 最新已用 = **s40**。下面把 A1/A3/A4 的编号 finalize 成一条连贯序列（A1: s41-s43；A3: s44-s46；A4: s47-s48；**stacked / 改-(T,K) 重跑预留 s49+**）。**条件提交**——只在 submit_condition 满足时才占用 Codabench 配额。

| s号 | 类型 | 文件名 | 这是什么 | 何时交（submit_condition） | 预期分 |
|---|---|---|---|---|---|
| **s41** | m | `testp1_s41_m_a1sc-s01` | A1：s01 单模自洽（T0.7/K8 + greedy 成员，add-only union） | A1 GATE GO 或 SOFT-GO 后；ADD-mode，s01=47.80 兜底；validate 通过 | 48.5-49.5（乐观 50.0） |
| **s42** | u | `testp1_s42_u_a1sc-s15` | A1：自洽嵌进 s15（5 成员 ×(greedy+K8) 大并集） | **仅当 s41 在 Codabench 真>47.80**（确认 densematch→testp1 迁移）且离线 gate ratio≥1.5 | 50.5-52.0（乐观 52.5） |
| **s43** | m | `testp1_s43_m_a1sc-k16-s01` | A1：s01 K=16 高采样变体（应急） | 仅当 s41 落 48.0-49.5 **且** 离线 K16 严格>K8（score+PEM 双赢）；否则跳过 | ≈s41 ±0.3 |
| **s44** | m | `testp1_s44_m_a3-tau05` | A3：tau0.5 asym-clip 单模（full-traindev regime-a） | A3 Stage-A GO（Contra+UE any-of F1 升 + PEM 不降，相对 s31 同-regime）后；ADD-mode | 47-49（P>47.80~14%） |
| **s45** | u | `testp1_s45_u_s15-a3soft` | A3：s15 + A3 **soft-blend** 第 6 成员（非 hard-OR） | s44 raw 有 s15 漏掉的新正确少数类 **且** soft-blend 离线不崩 PEM；A3 最高 EV 一枪 | 50.0-52.5（P>50.26~25-30%） |
| **s46** | m | `testp1_s46_m_a3-tau03` | A3：tau0.3 fallback（仅 tau0.5 过冲时） | 仅当 Stage-A 中 tau0.3>tau0.5 **且** tau0.3 过 GO；否则此号不用 | 47-49 |
| **s47** | m | `testp1_s47_m_grpoE0full` | A4：GRPO run#1 ckpt（E0-full 暖启 + Dr.GRPO 逐段归一），greedy testp1 单模 | **全部 G3 子门通过** 后才烧一枪 | 47-50（modal <47.80，P>47.80~12%） |
| **s48** | u | `testp1_s48_u_s15-grpoE0full` | A4：s15 + GRPO ckpt 作第 6 个 hard-OR 异构成员 | 仅当 s47 G3 过 **且** 离线 union 检查 >50.26；ADD-mode | 目标 >50.26（P 有用成员~15-20%） |
| **s49+** | u/m | （动态命名） | **预留**：A1×A3 stacked union；或任何改了 (T*,K*)/成员的 numbered 模型重跑 | 触发时即取下一空号 + 追加 log 一行 | — |

**Codabench 配额预算**：A1 期望 1-2 枪（s41 必，s42 条件，s43 罕见）；A3 期望 2 枪（s44+s45，s46 仅 tau 分离时）；A4 期望 0-2 枪（modal=0，L0/G3 多在烧枪前 KILL）。**总期望 3-5 枪**。

**ledger 纪律（铁律 §1.3.5 落实）**：§6 决策树里出现的「stacked 变体」「把 (T*,K*) 设为后续 default 后重跑某 numbered 模型」——**任何这类会产生新 Codabench 上传的变化都必须取 s49+ 新号 + 新 log 行**，绝不覆盖 s44-s48 的既有行。把 (T*,K*) 设为 A3/A4 的 decode default 只影响**尚未跑的**新模型；已编号模型不回改。

---

## 3. A1 详细执行 — 逐句温度自洽（add-only）

### 3.0 🔴 FATAL 修正：门必须跑在 plain-CE per-sentence train' adapter（不是 softmin cotE0/s40）

- **证据来源（已核对 `notes/phaseC/CAMPAIGN_SUMMARY.md:12-18`）**：A1 自洽 +1.4 lift 出自 **A1 plain-CE (per-sentence) train' adapter**：densematch greedy 81.13 → E1 self-consist(3×) 82.53（score +1.40 / PEM 79.24→80.10 +0.86…—以 score+PEM 双正为门）。
- **陷阱**：仓库有**两个都叫 "E1" 的东西**——PHASE-C E1=自洽（KEEP），CoT E1=s39（FAILED, -5.31）。原草稿误把门跑在 `cotE0_s01label`（=s40 的 **softmin-grid label-only** train' adapter，densematch 0.84，testp1 43.85）。softmin 锐化逐 token 分布 → 采样多样性更低 → union lift 可能显著缩水。**在 softmin adapter 上测 lift ≠ 复现已验证的 +1.4，会用一个未验证量冒充已验证量。**
- **修正（强制）**：A1 门-adapter = **PHASE-C plain-CE per-sentence train' adapter**（A1/U3 lineage 的 A1 成员）。
  - 若该 ckpt 仍在服务器：直接 resolve（见 §3.3 `A1CE_CKPT`）。
  - **若已删**：必须先用 **plain-CE per-sentence** 配方在 `data/devbench/train_sft.jsonl` 上重训一遍再做门——**绝不**用 softmin adapter 替代。
- **同 lineage 引用纪律**：门只引用 **同配方** 的离线证据（plain-CE +1.4）。把旋钮 apply 到 s01（softmin full-traindev）时，**额外**在 s01 自身上测 within-model 相对 lift（§3.5b）以补迁移确认——不再单靠 plain-CE 数字外推到 softmin 生产模型。

### 3.1 现状（已核对源码，~95% 已实现，A1 不需改 infer.py / ensemble_union.py）

- `infer.py` 默认 greedy（`RequestConfig` temperature=0.0，与现状逐字节一致）。
- `--temperature`（:724，默认0.0）、`--seed`（:729）、`--n`（:733）、`--top_logprobs`、`--no-logprob`（:760）**全已存在**。
- ✅ **`--n` 已能在单进程内产 K 个 sample 文件**（infer.py:815-863：`n>1` 时逐 choice 写 `<out>.s0..s{n-1}`），比「跑 K 次」省 GPU；**但 `--n` 需 temperature>0**（:820-823 硬报错）。
- seed 仅在 `temperature>0 且 --seed 给定` 时进引擎（infer.py:813-814）。
- `ensemble_union.py --min-votes 1`（默认，:47-78）= **精确 add-only hard-OR**：一个位置只要任一 sample 开非-Supported 即采纳；全 Supported 才 Supported；分歧→最多票，平票→更稀类（PRIORITY）。**直接复用**，**绝不用 `--min-votes 2`**（=s16 家族，-2.97）。

**两条 A1 路径（seed 语义不可混用，§7.2 风险表锁定）**：
- **PATH A（默认）**：K 次独立 run，**每次必传一个相异 `--seed`**；漏传 → K 个样本同种子坍缩成一个 → union 静默 no-op。产物文件名清晰、与 union 语义对齐。
- **PATH B（省 GPU 可选）**：单 run `--n K --temperature>0`，**不需要也不应靠 `--seed` 制造多样性**（n 个样本来自同一采样器流，seed 只作用于这一次调用）。**用 B 就别期望 --seed 增多样**。
- 本计划默认 PATH A；不在两路之间交叉引用 seed 语义。

### 3.2 代码改动

| 文件:函数 | 改动 |
|---|---|
| `src/nlpcc_t10/infer.py` | **零改动**。PATH A 传 `--temperature 0.7 --seed K`；PATH B 传 `--temperature 0.7 --n K`。**不要**新加 num_samples 逻辑。 |
| `scripts/ensemble_union.py` | **零改动**。复用 `--min-votes 1`。 |
| `scripts/a1_selfconsist.sh`（**唯一新增产物，~80 行**） | 参数化包装：env `ADAPTER, SPLIT(dev\|testp1), REF, T(0.7), K(8), SEEDS(1..K), GPU, OUTDIR`。PATH A：循环 seed `infer(--temperature $T --seed $s --out raw_$s)`→`aggregate(--pred raw_$s --ref $REF --out sub_$s)`；再 `ensemble_union --min-votes 1 sub_1..K → final`；有 gold 则跑 `evaluate.py`。可选 `GREEDY_SUB` 把 K=1 greedy 作 union 成员（保证对 greedy 纯加法）。 |

### 3.3 服务器命令（CWD=repo root）

```bash
# ===== 环境 =====
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache; export USE_HF=0; export TOKENIZERS_PARALLELISM=false
DENSE=data/devbench/dev_gold_densematch.jsonl

# ===== STEP -1：解析并断言所有 ckpt 存在（outputs/ 本地为空，路径全在服务器、带 v0-* 时间戳）=====
# 🔴 A1 门 adapter = PHASE-C plain-CE per-sentence train'（NOT softmin cotE0/s40）。若不存在则先重训（见 §3.0）。
A1CE_CKPT=$(ls -d outputs/phaseC/A1_plainCE/ckpt/v*/checkpoint-* 2>/dev/null | sort | tail -1)
S01_CKPT=$(ls -d outputs/grid_b5_l0.5/v*/checkpoint-2000 2>/dev/null | sort | tail -1)   # full-traindev 生产 adapter
for v in A1CE_CKPT S01_CKPT; do
  p=$(eval echo \$$v); if [ -z "$p" ] || [ ! -d "$p" ]; then echo "FATAL: $v unresolved/missing ($p)"; fi; echo "$v=$p"
done
# 若 A1CE_CKPT 缺：plain-CE per-sentence 重训 devbench train' 后再门（绝不用 softmin 替代）：
#   USE_HF=0 TAG=phaseC_A1_plainCE LOSS=ce TAU=0.0 ... DATA=data/devbench/train_sft.jsonl bash scripts/train_softmin.sh
mkdir -p outputs/a1 submissions/a1

# split.json 绕过：infer.py 用 --records-file $DENSE 直接读 206-rec densematch 整-record 文件（A3/L0 同），不碰/不依赖全局 data/split.json，无 dangling 态
# (DEV_SFT removed: infer.py --records-file $DENSE drives inference from the 206-rec densematch record file directly)

# ===== STEP 0：plain-CE greedy 基线（lift 参照，与 +1.4 证据同 lineage）=====
CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $A1CE_CKPT --engine pt --data-root "$DATA_ROOT" --temperature 0.0 --out outputs/a1/ce_greedy_raw.jsonl
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a1/ce_greedy_raw.jsonl --ref $DENSE --out submissions/a1/ce_greedy.jsonl
uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold $DENSE --pred submissions/a1/ce_greedy.jsonl --match id   # GREEDY baseline (expect ~81.1)

# ===== STEP 1：leakage-aware GATE = 在 plain-CE train' adapter 上扫 (T,K)，测 lift vs greedy =====
# PATH A，一次产 16 seeds，离线再子采样 K∈{4,8,16}；T∈{0.5,0.7,1.0}；TWO seed-blocks 估噪声（见 §3.4）
for T in 0.5 0.7 1.0; do for s in $(seq 1 16); do
  CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $A1CE_CKPT --engine pt --data-root "$DATA_ROOT" --temperature $T --seed $s --out outputs/a1/ce_t${T}_seed${s}_raw.jsonl \
  && PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a1/ce_t${T}_seed${s}_raw.jsonl --ref $DENSE --out submissions/a1/ce_t${T}_seed${s}.jsonl
done; done
# 每 (T,K) 取并集并评分；K=8 用两个不相交 seed-block(1-8 与 9-16)各算一次 → lift 的两点估计
for T in 0.5 0.7 1.0; do for K in 4 8 16; do
  uv run python scripts/ensemble_union.py --min-votes 1 --out submissions/a1/ce_sc_t${T}_k${K}.jsonl $(for s in $(seq 1 $K); do echo submissions/a1/ce_t${T}_seed${s}.jsonl; done)
  echo "=== T=$T K=$K (block1) ==="; uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold $DENSE --pred submissions/a1/ce_sc_t${T}_k${K}.jsonl --match id
done
  # 第二 block（seeds 9-16）只算 K=8，给 lift 误差棒
  uv run python scripts/ensemble_union.py --min-votes 1 --out submissions/a1/ce_sc_t${T}_k8_blk2.jsonl $(for s in $(seq 9 16); do echo submissions/a1/ce_t${T}_seed${s}.jsonl; done)
  echo "=== T=$T K=8 (block2) ==="; uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold $DENSE --pred submissions/a1/ce_sc_t${T}_k8_blk2.jsonl --match id
done

# ===== STEP 1b：firing-concentration 诊断（新增的少数类是否落在 greedy-WRONG 段）=====
uv run python - <<'PY'
import json
g={r['id']:r['labels'] for r in map(json.loads,open('submissions/a1/ce_greedy.jsonl'))}
u={r['id']:r['labels'] for r in map(json.loads,open('submissions/a1/ce_sc_t0.7_k8.jsonl'))}
gold={r['id']:r['labels'] for r in map(json.loads,open('data/devbench/dev_gold_densematch.jsonl'))}
aw=ar=0
for i in gold:
  gp,up,gl=g[i],u[i],gold[i]; pw=any(a!=b for a,b in zip(gp,gl))
  for a,b in zip(gp,up):
    if a=='Supported' and b!='Supported':
      if pw: aw+=1
      else: ar+=1
print('union-added firings on greedy-WRONG paras:',aw,' on already-right:',ar,' ratio:',round(aw/max(1,ar),2))
PY
```

### 3.4 leakage-aware densematch GATE（必跑在 plain-CE per-sentence train' adapter，绝不在 softmin / s01）

- **Bench**：densematch（206 rec）。必须 `aggregate --ref $DENSE`；用 `--records-file $DENSE`（**不** cp 改全局 `data/split.json`，避免 §7.2 dangling-state 坑）。
- **测量**：self-consistency LIFT = (add-only union score − greedy score)，扫 T∈{0.5,0.7,1.0} × K∈{4,8,16}，**两个不相交 seed-block 各算 K=8 给误差棒**（单点 +1.4 无方差，必须 bound 噪声）。
- **GO 条件（粗二元，铁律 §1.3.7）——全满足才 GO**：
  1. add-only union 的 score lift **和** PEM lift 在两个 seed-block 上 **都清晰为正**（目标 ~+1.4/+1.9，PHASE-C plain-CE 已验；以「双 block 双 metric 同号且幅度明显」为准，不抠 ±0.3）；
  2. 新正确 firing **集中在 greedy-WRONG 段**：诊断 ratio `add_on_wrong/add_on_right ≥ 1.5`；
  3. 所选 (T,K) 处 **PEM 不回退**（分歧位 FP 是 s30/s37 失败模式）。
- **GO** → 用该 (T*,K*) 走 Codabench s41/s42。**默认 GO 点：T*=0.7, K*=8**（PHASE-C 验证点）；K=16 仅在 score+PEM **双严格**超 K=8 时用，否则 K=8 省一半 GPU。
- **SOFT-GO**（lift 微正但 ratio 1.0~1.5 或一个 block 偏弱）→ 只烧 **s41 一枪**（便宜、ADD-mode 兜底），暂不交 s42。
- **KILL**（best (T,K) lift 不清晰为正 或 PEM 回退 或 ratio<1.0）→ A1 是 no-op/有害，**不提交**，转 A3。A1 的跑-K-次基建留作 §4 soft-blend 成员素材。
- ⚠️ **所有 50.0-50.3 级别的绝对落点判定一律 Codabench**（densematch 分辨不了）；离线 GATE 只决定「烧不烧 s41」这个粗二元。

### 3.5 生产提交命令（GATE GO 后）

```bash
# ===== s41：s01 自洽 union（含 greedy 成员保证纯加法）=====
for s in $(seq 1 8); do
  CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split testp1 --model Qwen/Qwen3-VL-8B-Instruct --adapter $S01_CKPT --engine pt --data-root "$DATA_ROOT" --temperature 0.7 --seed $s --out outputs/a1/s01_testp1_t07_seed${s}_raw.jsonl \
  && PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a1/s01_testp1_t07_seed${s}_raw.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/a1/s01_testp1_t07_seed${s}.jsonl
done
CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split testp1 --model Qwen/Qwen3-VL-8B-Instruct --adapter $S01_CKPT --engine pt --data-root "$DATA_ROOT" --temperature 0.0 --out outputs/a1/s01_testp1_greedy_raw.jsonl \
&& PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a1/s01_testp1_greedy_raw.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/a1/s01_testp1_greedy.jsonl
uv run python scripts/ensemble_union.py --min-votes 1 --out submissions/testp1_s41_m_a1sc-s01.jsonl submissions/a1/s01_testp1_greedy.jsonl $(for s in $(seq 1 8); do echo submissions/a1/s01_testp1_t07_seed${s}.jsonl; done)
uv run python scripts/make_submission_zip.py --sub submissions/testp1_s41_m_a1sc-s01.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/testp1_s41_m_a1sc-s01.zip && uv run python scripts/validate_submission.py submissions/testp1_s41_m_a1sc-s01.zip

# ===== s42（仅 s41 真>47.80 后）：A1 嵌进 s15。5 成员各 K=8 testp1 采样，再一把大并集 =====
uv run python scripts/ensemble_union.py --min-votes 1 --out submissions/testp1_s42_u_a1sc-s15.jsonl <5 成员 greedy subs> <5 成员 ×8 采样 subs>
uv run python scripts/make_submission_zip.py --sub submissions/testp1_s42_u_a1sc-s15.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/testp1_s42_u_a1sc-s15.zip && uv run python scripts/validate_submission.py submissions/testp1_s42_u_a1sc-s15.zip
```

### 3.5b 🔴 SERIOUS 修正：生产模型自身的 within-model 迁移确认（softmin 熵风险）

- **问题**：lift 在 plain-CE train' adapter 上验，却 apply 到 **softmin full-traindev s01**。softmin + 近记忆（E0-full loss ~0.005）的 s01 在同温度下采样熵远低于 plain-CE train' 模型 → 同 T=0.7/K=8 在 s01 上产生的分歧位少得多 → 真实 lift 可能远小于门预测。
- **修正（强制，烧 s41 前做，便宜）**：把 K=8/T=0.7 自洽**直接跑在 s01 自身**的 densematch 上，**只读 within-s01 的 greedy→union 相对 DELTA**（s01 densematch 绝对值泄漏不读，但同模型内的 greedy→union 差是 regime-internal 且合法，正是 §1.2 原则用在生产模型上）。
  ```bash
  # s01 自身 within-model lift（densematch，只读 DELTA 不读绝对值）
  for s in $(seq 1 8); do CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $S01_CKPT --engine pt --data-root "$DATA_ROOT" --temperature 0.7 --seed $s --out outputs/a1/s01_dev_t07_seed${s}_raw.jsonl && PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a1/s01_dev_t07_seed${s}_raw.jsonl --ref $DENSE --out submissions/a1/s01_dev_t07_seed${s}.jsonl; done
  CUDA_VISIBLE_DEVICES=3 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $S01_CKPT --engine pt --data-root "$DATA_ROOT" --temperature 0.0 --out outputs/a1/s01_dev_greedy_raw.jsonl && PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a1/s01_dev_greedy_raw.jsonl --ref $DENSE --out submissions/a1/s01_dev_greedy.jsonl
  uv run python scripts/ensemble_union.py --min-votes 1 --out submissions/a1/s01_dev_sc_k8.jsonl submissions/a1/s01_dev_t07_seed{1,2,3,4,5,6,7,8}.jsonl
  echo "s01 greedy:"; uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold $DENSE --pred submissions/a1/s01_dev_greedy.jsonl --match id
  echo "s01 self-consist k8:"; uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold $DENSE --pred submissions/a1/s01_dev_sc_k8.jsonl --match id
  ```
- **熵 sanity-check**：若 s01 自洽的逐位置分歧率 < 门 adapter 的 ~一半（或 within-s01 DELTA 明显弱于 plain-CE 门），**先把 s01 的 T 调高**（如 0.9/1.0）匹配多样性，再选 K，再烧 s41。within-s01 DELTA 清晰为正才烧 s41。

### 3.6 A1 分数应对（见 §6 主决策树更全）

- s41 ≥ 50.3：A1 单杠杆已越过 union → 立即交 s42，并把 (T*,K*) 设为 A3/A4 的**新模型** decode default（不回改已编号模型）。
- s41 48.0-50.3：迁移确认 → 交 s42（主路）；若 K16>K8 排 s43。
- s41 47.0-48.0：近 no-op（+0.59 迁移 gap 咬人）→ **不交 s42**，转 A3。
- s41 <47.0：FP 崩 PEM（s30/s37 模式）→ 丢弃（s01 兜底），降 K/T 或 KILL A1 转 A3。

---

## 4. A3 详细执行 — logit-adjusted / balanced-softmax margin

### 4.1 思路与 regime（必须 regime-a：s01 softmin+oversample 之上，非 regime-b plain-CE）

训练时在 **label-span 位置** 对 gold 类 logit 加 per-class 偏置 `+clamp(tau*log(pi_y), min=CLIP, max=0)`；推理（无偏置）下稀类相对被抬 → 更多少数类召回（已验证的召回方向，与被证伪的「抑制」后处理相反）。`pi_y` = **真实未过采样、train-only PRIMARY-label 先验**（`compute_global_label_freq` over `train_idx`，**PRE-resample**——用 post-oversample 计数会双重纠正、复现 s09 -4.0 FP-collapse）。

### 4.2 代码改动

| 文件:函数 | 改动 |
|---|---|
| `loss.py:_per_sentence_ce`（:164，单一 chokepoint，被 floor :334 与 bottleneck :351 同时调用） | 加可选参 `class_bias=None`（[V] tensor）；构 label-span mask（复用 :192-194 `valid` 逻辑），`torch.where(mask[...,None], shift_logits+class_bias, shift_logits)`。**只加在 5 个 label token 位**，不碰 JSON 标点/CoT token。 |
| `loss.py:SoftMinPEMLoss._get_class_bias_vec()`（新，旁 `_get_brace_ids` :308） | 读 `SOFTMIN_TAU`（默认0.0→返回 None→**逐字节 no-op**，仿 SOFTMIN_WARMUP_FRAC）；否则从 `SOFTMIN_PRIOR_JSON` 载 5 先验，tokenize 5 个 TRACK1_LABELS，找各自 **FIRST DISTINGUISHING token id**（UCM/UE 共享前导 'Unsupported' → 必须按首个相异 index，否则偏置碰撞），建 [V] 向量值 `clamp(tau*log(pi_y), min=SOFTMIN_TAU_CLIP(-2.0), max=0.0)`；RANK0 打印一次。 |
| `loss.py:make_softmin_loss_cls.__call__`（:332） | `class_bias=self._get_class_bias_vec()` 一次，threaded 进 floor(:334) 与 bottleneck(:351)；tau≤0 时 None → 旧路径逐字节不变。 |
| `build_dataset.py:main`（:828-832 已算 label_freq over train_idx :829） | 同时写 `data/<out>/label_prior.json = {label: freq/sum(5)}`（同 train_idx，PRE-resample，train-only）；train'(devbench) 与 full-traindev 各得匹配先验。 |
| `retrain_fullres.sh`（:25/:31/:41） | 加默认 `TAU=0.0 / TAU_CLIP=-2.0 / PRIOR_JSON`；torchrun 行 export `SOFTMIN_TAU/SOFTMIN_TAU_CLIP/SOFTMIN_PRIOR_JSON`；🔴 把硬编码 `--use_logits_to_keep true`（:41）改成 `${USE_LK:-true}`，A3 run 传 **`USE_LK=false`**（full-vocab logits 索引偏置 + softmin 因果位移都要求 false，见 loss.py:64）。 |
| `scripts/ensemble_soft_blend.py`（**新**，ensemble_union 只 hard-OR） | 逐句 soft 融合 A3 成员进 s15：A3 开少数类且 min_logprob 超 per-class 置信 floor 且无更高置信成员矛盾时采纳；add-only + soft 守卫（**非 ≥2-票** = s16 -2.97）；保 `--min-votes 1` 并集为地板。 |

### 4.3 服务器命令（H200，USE_HF=0）

```bash
# === STAGE A：train'-only 干净 gate（devbench，与 densematch image-disjoint）===
# A.0 重建 devbench 以产 label_prior.json（train'-only PRE-resample；保持 s01 配方 os3.0/ds0.6 = regime-a）
USE_HF=0 PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --split-mode component_aware --val-ratio 0.15 --out data/devbench --minority-oversample 3.0 --supported-downsample 0.6
cat data/devbench/label_prior.json   # 验证：Supported~0.93 主导，稀类极小

# A.1 train' A3，tau=0.5，asym clip，use_logits_to_keep=false（trains data/devbench/train_sft.jsonl）
USE_HF=0 TAG=a3t05_trainp TAU=0.5 TAU_CLIP=-2.0 USE_LK=false PRIOR_JSON=data/devbench/label_prior.json DATA_ROOT="$DATA_ROOT" GPUS=0,1 BETA=5 LAMBDA=0.5 EPOCHS=1 bash scripts/retrain_fullres.sh
# ⚠️ 必须在 run log 里 grep 确认 use_logits_to_keep=false（否则 full-vocab 偏置索引错位）

# A.2 推理 train' A3 adapter 在 densematch DEV（非 testp1）→ aggregate（用 --records-file $DENSE，不改全局 split.json）
USE_HF=0 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $(ls -d outputs/p0_a3t05_trainp/v*/checkpoint-* | sort | tail -1) --engine pt --data-root "$DATA_ROOT" --out outputs/a3t05_dev_raw.jsonl
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/a3t05_dev_raw.jsonl --ref data/devbench/dev_gold_densematch.jsonl --out outputs/a3t05_dev_submission.jsonl

# A.3 GATE eval（相对 s31 train' floor，同 regime-a）
uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold data/devbench/dev_gold_densematch.jsonl --pred outputs/a3t05_dev_submission.jsonl --match id

# === STAGE B（仅 Stage-A GO）：regime-a full-traindev 生产 ===
USE_HF=0 PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data/ --minority-oversample 3.0 --supported-downsample 0.6   # 产 data/label_prior.json
USE_HF=0 TAG=a3t05_full TAU=0.5 TAU_CLIP=-2.0 USE_LK=false PRIOR_JSON=data/label_prior.json DATA_ROOT="$DATA_ROOT" GPUS=0,1 BETA=5 LAMBDA=0.5 EPOCHS=1 bash scripts/retrain_fullres.sh
# 脚本尾产 submissions/testp1_a3t05_full_submission.zip（= s44 单模）→ 重命名 testp1_s44_m_a3-tau05.*

# B.2 soft-blend 进 s15（非 hard-OR）→ s45
PYTHONPATH=src uv run python scripts/ensemble_soft_blend.py --base-members <s01 s08 s09 s13 s12 raw jsonls> --a3-member outputs/testp1_a3t05_full_raw.jsonl --out outputs/s15_plus_a3_soft.jsonl
PYTHONPATH=src uv run python -m nlpcc_t10.aggregate --pred outputs/s15_plus_a3_soft.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/testp1_s45_u_s15-a3soft.jsonl
uv run python scripts/make_submission_zip.py --sub submissions/testp1_s45_u_s15-a3soft.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/testp1_s45_u_s15-a3soft.zip && uv run python scripts/validate_submission.py submissions/testp1_s45_u_s15-a3soft.zip
```

### 4.4 Stage-A 门 + proxy 不可离线说明（同-regime 相对，非绝对读）

- **干净性**：A3 adapter 只在 devbench train' 训（从不见 206 densematch 记录）→ densematch **非泄漏**。
- **GO 比较是「A3-train'-regime-a vs s31-train'-regime-a」**：s31 = PHASE-C A0 softmin on devbench train' **WITH oversampling**（os3.0/ds0.6，已核对 = regime-a，**不是** log 里笼统写的「无过采样」）。两者同过采样、同 train' split → 合法的同-regime 相对地板。
- **GO 阈值（相对、非绝对常数）**：(Contra F1 + UE F1) any-of 平均 **升 vs s31** **且** PEM **不降 vs s31** **且** A3-train' 总 score **不低于** s31-train' 总 score（同 regime-a 比较，而非读 0.7897 这个绝对数当 testp1 level）。绑定信号 = **相对排序 + Contra+UE any-of F1 上升**，绝对数永不当 testp1 预测。
- **KILL（<5 GPU-h）**：Contra+UE any-of F1 不升 / PEM 降 / 总 score 低于 s31 → margin 不帮少数类召回 → KILL，不跑 Stage-B、不烧枪。先 tau=0.5；若失败试 tau=0.3 一次（过冲比欠冲便宜）。
- 🔴 **OFFLINE-PROXY 不可能**：infer.py 只存 min/sum_logprob over label span（:419-429，greedy，top_logprobs=1），**从不存 5-way class logits** → A3 **无法**对已存模型 post-hoc A/B，**每次评估都要 fresh train**。最终 testp1 判定 = **一枪 Codabench**（s44），是 A3 方向的 binding verdict。

### 4.5 A3 融合铁律

🔴 **绝不 hard-OR 一个 A3 成员进 s15**：s21 即使 +90 正确 hard-OR firing 仍掉到 47.77（PEM 39.59）——一句错翻一整段。A3 必须经 **soft-blend** 进。`ensemble_union.py` 只 hard-OR，不能复用于此。

---

## 5. A4 详细执行 — GRPO（CONDITIONAL-GO）

> P(single>47.80)~12%，P(有用 union 成员)~15-20%，combined P(win)~20%。机器（train_grpo.sh / grpo_reward.py / build_grpo_data.py / build_density_matched.py）已接好且 reward any-hit 逻辑正确，**但已核对：四个前置 P0-P3 当前全未满足**（`outputs/` 仅 .gitkeep；`data/cot` 只有 pilot/；无 train_b1_joint.jsonl；build_grpo_data 发 1-元素 gold；train_grpo.sh 硬接 CoT）。

### 5.0 ckpt 路径解析（与 §3 同纪律，先解析再断言）

```bash
# A4 所有 ckpt 用 ls|sort|tail 解析 + 断言存在，绝不硬编码字面路径（outputs/ 本地为空）
E0FULL_CKPT=$(ls -d outputs/E0full/v*/checkpoint-* 2>/dev/null | sort | tail -1)
[ -z "$E0FULL_CKPT" ] && echo "E0FULL missing — run P1 SFT first (§5.2)"
```

### 5.1 L0 dense pass@8 前置（**最先跑，near-0 即 STOP**，决定 ~80% GPU 是否值得花）

- **为什么 dominant**：s01 在 testp1 仅 3.8% 段开 ≥2 少数类；traindev 仅 8/3333 记录(0.46%) 有 ≥2 少数类句。若 E0-full 几乎从不在 dense 段采到正确的 **第二个** 少数类 → 每个 density-augment prompt 全是零优势 rollout，GRPO 在目标行为上是 no-op（**RL 只放大、不创造**）。
- 🔴 **SERIOUS 修正 — L0 也对 E0-full 泄漏**：densematch 的 206 记录是 full-traindev 的子集，而 E0-full 在 full-traindev 上训 → **E0-full 见过这 206 条**。在记忆过的记录上测 pass@8 会 **高估** warm-start 在**未见过**的 dense 段采到正确第二少数类的真实能力。false-GO 会绿灯 ~15-27 GPU-h 的 GRPO，其全部前提建立在泄漏数据上。
- **修正（强制，二选一）**：
  - **(首选) 在 image-disjoint held-out 上测 dense pass@8**：从 **devbench train'** 用 `build_density_matched.py` 造一个 dense 探针集（E0-full 训练用的是 full-traindev，devbench train' 与 densematch image-disjoint——但注意 E0-full 见过 full-traindev 全集，故须用一个 E0-full **未训练**子集；若无法干净切出，则走下一条）；
  - **(退而求其次) 在 densematch 上测，但显式当作 LEAKY UPPER BOUND**，GO 门槛抬高到 **远超 20%**（见下），并在 §7.2 风险表登记 L0 泄漏（镜像 G3 的处理）。
- **跑什么**：8 rollouts/prompt @ temp1.0（seeds 1-8，infer.py `--seed/--temperature`，零新代码），覆盖 SPARSE（E0-full 逐句 prompt）+ DENSE（从**已验证** train_b1_joint.jsonl 展开的逐句行）；pass@8 = 8 rollout 命中 **FULL gold types[]**（P2-修后多标签）的比例。出 {UCM,Contra,UE,SO}×{sparse,dense} 表。
- **GO/KILL**：
  - 干净 held-out 路：GO = 每个少数类 dense pass@8>0 率 **≥20%**。
  - LEAKY densematch 路：GO 门槛抬到 **dense pass@8 明显 >20%**（如 ≥35%，给记忆虚高留 margin）；near-threshold 不 GO。
  - 某稀类 dense<门 但 sparse≥门 → GO 但不抬其 w_class。某类两边都 0% → 不可训，跳过。
  - **KILL = ≥3/4 少数类 dense<门** → warm-start 无火种，RL 无放大对象 → A4 no-op → 记负、保 s01/s15、STOP。

```bash
# L0 pass@8（8 seeds，sparse + dense；分别 score vs FULL gold types[]）
for s in $(seq 1 8); do
  CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=401408 PYTHONPATH=src USE_HF=0 uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $E0FULL_CKPT --engine pt --data-root "$DATA_ROOT" --temperature 1.0 --seed $s --max-new-tokens 32 --out outputs/pass8_sparse_seed${s}.jsonl
done
# dense 集 = 从 verified train_b1_joint.jsonl 展开的逐句行，同 8-seed 循环；pass@8 命中 FULL gold types[]
```

### 5.2 P0-P3 前置修复（L0 GO 后才推进）

- **P0**（密度匹配数据，当前缺）：
  ```bash
  uv run python scripts/build_density_matched.py --data-root "$DATA_ROOT" --split data/devbench/split.json --out data/devbench/train_b1_joint.jsonl --min-minorities 2 --target-len 12 --max-len 16
  # K-prereq：若打印 n_synth==0（same-SHA 共证据饥饿，build_density_matched.py:99-110）→ STOP，A4 死
  ```
  🔴 **P0+P2 耦合**：`make_sample_joint`（build_dataset.py:556-566）只存 `{labels:[picked_single]}`，**无 per-sentence full types[]** → `--density-augment` 无法从 train_b1_joint.jsonl 恢复多标签 gold。**先**在 make_sample_joint 持久化 `gold_types_per_sent` sidecar。
- **P1**（E0-full 暖启，当前无）：full-traindev **label-only CE**（s01 是 softmin-loss 非干净 label-only；s40 是 train'-only=禁止跨 regime）。**新建 `scripts/train_e0full.sh`**：包 train_softmin.py，s01 配方逐项一致（grid b5/l0.5, os3.0/ds0.6, full data/train_sft.jsonl, 1ep, max_pixels 401408, **use_logits_to_keep=false**, max_length 10240）但目标 plain label-only `{label:L}`（loss_type CE 非 softmin_pem）；TAG=E0full；**绝不覆盖 s01**。
  ```bash
  uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data/E0full --minority-oversample 3.0 --supported-downsample 0.6
  TAG=E0full MAX_PIXELS=401408 GPUS=0,1 bash scripts/train_e0full.sh
  # 注：s01 不能直接当 E0-full 代理（softmin loss≠clean label-only CE）；必须真训 E0-full。
  ```
- **P2**（多标签 gold，真 bug）：`build_grpo_data.py:57` 现发 `gold_types=[label]`（单元素）→ grpo_reward 的 any-hit 被饿死，527 多标签句无法 credit 备选有效标签。修：按 (record_id + sentence index) 回 join `$DATA_ROOT/data/traindev-track-1.jsonl`，挂 `gold_types = 该句 full types[]`（rarest-pick 只作 SFT 目标，非 RL credit set），last_line_label 仅 fallback。`compute_reward/_as_gold_list`（:104-114/:83-101）已支持 list，**逻辑不改**；扩 `_selftest`（~:176-188）加多标签用例锁行为。**run#1 不加 R_fp**（伪装抑制，已 reject）。
- **P3**（label-only prompt，非 CoT）：`build_grpo_data.py` 默认 `--in=data/cot/e1/train_sft.jsonl`（也不存在；data/cot 只 pilot/）→ 改成 E0-full 集 `data/E0full/train_sft.jsonl`；train_grpo.sh 硬接 `<analysis>` CoT rollout（max_completion_length 160）= 重入被证伪的 CoT regime（E1=-5.31）→ 改 label-only `{label:L}`。

### 5.3 GRPO 启动（L0 GO + 干净 L1 SMOKE 后）

🔴 **SERIOUS 修正 — de-CoT 必须可验证、非无守卫手改**：shipped `train_grpo.sh` 硬编码 CoT 死胡同（~L2 `<analysis>` rollout、L21 ADAPTER 默认 E1=CoT 模型、L57 max_completion_length=160）。若任何步骤跑了**未改**的 train_grpo.sh（哪怕传了 ADAPTER=E0-full 但没剥 `<analysis>` 格式），A4 静默重入已证伪 -5.31 的 CoT regime。

```bash
# === 改 train_grpo.sh：ADAPTER=E0-full；--max_completion_length 64；--temperature 1.1；
#   --scale_rewards false（Dr.GRPO mean-only）；--beta 0（去 KL，散了再加 0.001）；label-only rollout；
#   per-paragraph advantage grouping（load-bearing；ms-swift 4.2.3 不支持则退 env w-band: UCM2.5/CONTRA2.25/UE2.0/SO1.75）；
#   保留 --vllm_enable_lora false / grad_accum<=8 / USE_HF=0 ===

# 🔴 de-CoT 断言（L1 SMOKE 前必跑；任一失败 = 立即 KILL，不是静默放行）
grep -q '<analysis>' scripts/train_grpo.sh && echo "FATAL: CoT rollout still in launcher" || echo "OK no <analysis>"
grep -E 'max_completion_length' scripts/train_grpo.sh   # 必须 <=64
grep -E 'ADAPTER' scripts/train_grpo.sh                 # 必须指向 E0-full，非 E1

uv run python scripts/grpo_reward.py --selftest
uv run python scripts/build_grpo_data.py --in data/E0full/train_sft.jsonl --out data/cot/grpo/prompts.jsonl --density-augment data/devbench/train_b1_joint.jsonl
ADAPTER=$E0FULL_CKPT SMOKE=1 GPUS=0,1 bash scripts/train_grpo.sh   # L1 SMOKE：8 步/64 prompt
# 🔴 SMOKE 后 dump 1 个 rollout，断言 completion 是 plain {"label":L} 且零 analysis token；否则 KILL
ADAPTER=$E0FULL_CKPT GPUS=0,1 PROMPTS=data/cot/grpo/prompts.jsonl NUM_GENERATIONS=8 GRAD_ACCUM=8 EPOCHS=1 TAG=grpoE0full bash scripts/train_grpo.sh   # L2 run#1
```

**reward**：class-weighted ORM（Sup1.0/UCM5.0/Contra4.5/UE4.0/SO3.5）any-hit；R_fp 移除；PEM 压力靠 **per-paragraph advantage normalization**（Dr.GRPO mean-only, scale_rewards false, KL beta=0, 8 gen, temp1.1, lr1e-6, 1ep, max_completion_length 64, max_pixels 401408, lora r16/a32）。

### 5.4 G3 门 + 两条赢路 + STOP

```bash
# G3：GRPO ckpt greedy 在 densematch（仅排序/健康，非 testp1 预测器）
CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=401408 PYTHONPATH=src USE_HF=0 uv run python -m nlpcc_t10.infer --split dev --records-file $DENSE --model Qwen/Qwen3-VL-8B-Instruct --adapter $(ls -d outputs/grpoE0full/v*/checkpoint-* | sort | tail -1) --engine pt --data-root "$DATA_ROOT" --max-new-tokens 32 --out outputs/dev_grpoE0full_raw.jsonl
uv run python -m nlpcc_t10.aggregate --pred outputs/dev_grpoE0full_raw.jsonl --ref data/devbench/dev_gold_densematch.jsonl --out outputs/dev_grpoE0full_sub.jsonl && uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 --gold data/devbench/dev_gold_densematch.jsonl --pred outputs/dev_grpoE0full_sub.jsonl --match id
```

- 🔴 densematch 对 A4 **泄漏**（full-traindev 暖启含 densematch 记录）且已对 s30 误报（83.90→testp1 43.85）→ **G3 仅方向/健康，不是 testp1 预测器**，testp1 一枪是唯一真信号。
- 🔴 **SERIOUS 修正 — G3 子门 #2 改为「ranking + health-only」，不挂不存在的 bench**：原子门 #2「no-leakage testp1-shaped held-out ≥ E0-full」**依赖一个仓库里不存在的台子**（densematch 对 full-traindev 泄漏；reshaped 0 字节；无独立 image-disjoint-from-both 切片）。不可满足的门会让「全子门过才烧枪」要么永不烧、要么被悄悄豁免回 densematch-only（=误报 s30 的那条授权）。
  - **决议（二选一，执行前定死）**：
    - **(A) 真造干净 held-out**：从 full-traindev 切一个与 **E0-full 训练集 AND densematch 都 image-disjoint** 的 testp1-shaped 切片——代价是 E0-full 必须只训严格子集（牺牲部分数据 = 真 regime cost，要权衡）；造好后子门 #2 才成立。
    - **(B) 显式 DROP 子门 #2**，G3 重写为「densematch ranking + health-only」，**承认 testp1 一枪是唯一真信号**（= rl_breakthrough_research 实际结论）。本计划**默认走 (B)**，并把「modal 0 枪」的预算理解为「靠 L0 + health 门 + per-class 健康抽查把绝大多数坏 run 在烧枪前 KILL」，而非靠一个不存在的离线门。
- **G3 子门（默认 (B)：全过才烧枪 s47）**：(1) densematch greedy **排序上**明显 > E0-full（ranking-only，不读绝对值）；(2) 各少数类召回均衡，无类（尤其 SO）塌到 ~0；(3) 少数类预测数在精度安全带（s15 ~382/5096 的按记录缩放类比）；(4) 20-rollout 人工抽查干净（格式、无 analysis、无明显 FP-翻段）。
- **两条赢路**：s47（单模，modal<47.80）；**s48（s15 + GRPO 作第 6 hard-OR 异构成员，更可能赢）**——GRPO 的新 dense-minority 召回正是 union 成员造不出的结构。
- **STOP/KILL 链**：K-prereq(n_synth~0 或无干净 E0-full)→L0 前 STOP；K0(≥3/4 dense<门)→GRPO 前 KILL；L1 SMOKE de-CoT 断言失败→立即 KILL（非静默）；L1 SMOKE 两次失败→退 w-band，再不行弃 A4；K1(reward 升但 FP/format 升)/K2(熵塌→加 beta0.001)/K3(少数类爆+densematch PEM 降)→kill-switch 不提交；K4(G3 ranking/health 失败)→不烧枪；K5(逼近 2026-06-17 deadline，Phase-2 GPU 优先)→弃 A4，保 s01/s15。

---

## 6. ★ 主决策树（每次提交后按 score band 反应）

> 通用前提：所有 band 都 **保 s01=47.80 / s15=50.26**；平台若按最新计分，实验后顶回保底。「当前最优」初始 = 50.26。**所有 50.0-50.3 细分都是 Codabench-返回数判定，不是离线判定（铁律 §1.3.7）。任何 stacked / 改-(T,K) 重跑取 s49+ 新号（铁律 §1.3.5）。**

### 6.1 A1 — s41（s01 自洽单模）回来后

```
s41 ≥ 50.3  ──► A1 单杠杆已越 union！立即交 s42（A1×s15，预期再跳一步）。
              把 (T*,K*) 设为 A3/A4 **新模型** 的 decode 默认（不回改已编号模型）。→ 进 6.2。
s41 ∈ 48.0–50.3 ──► 迁移确认（预期/好）。交 s42（主路 >50.26）。
              若 gate 中 K16>K8（score+PEM 双严格），排 s43。→ 进 6.2。
s41 ∈ 47.0–48.0 ──► 近 no-op（+0.59 迁移 gap）。不交 s42（单模都不抬，union 更不抬）。
              s41 记负，预算转 A3（§6.3）。
s41 < 47.0  ──► FP 崩 PEM（s30/s37 模式）。丢弃（s01 兜底）。降 K/T(T=0.5) 或分歧位锚 greedy 重试（取 s49 新号）；
              仍<s01 则 KILL A1，转 A3。绝不烧 s42/s43。
GATE KILL（离线，未烧枪）──► 0 枪。记死 lift/ratio，直接转 A3（§6.3）。A1 跑-K-次基建留作 soft-blend 成员素材。
within-s01 DELTA 不正（§3.5b）──► 即便 plain-CE 门 GO，s01 自身无 lift → 提 s01 的 T 重测；仍无则不烧 s41，转 A3。
```

### 6.2 A1 — s42（A1×s15 union）回来后

```
s42 > 当前最优（>50.26 或 >s41 若 s41 已破 50.26）──► 新最优。更新「当前最优」。
              继续：把 A3 soft 成员 + A1 自洽成员一起叠（§6.3 的 stacked 变体，取 s49 新号 + log 行）。
s42 ∈ 50.0–50.26 ──► 撞 naive-union 天花板 / 成员相关。不再加同质成员。转 A3 找去相关成员。
s42 < 50.0  ──► 嵌套 union 引入 FP。回退到 s15=50.26。转 A3。
```

### 6.3 A3 — Stage-A 门 → s44（单模） → s45（soft-blend）

```
Stage-A KILL（Contra+UE any-of 不升 / PEM 降 / 总 score 低于 s31 同-regime）──► <5 GPU-h，0 枪停 A3。
              GPU 转 A1（若还有）/A4 前置。记负。
Stage-A GO，s44 ∈ 47.8–50.2 ──► A3 单模不破 union 但是有效去相关成员。直奔 s45 soft-blend（A3 的 EV 所在）。
s44 < 47.80 ──► test 时 margin 过冲（tau 高/clip 松→长段 FP-collapse）。
              若 Stage-A 中 tau0.3 borderline → 落 s46(tau0.3)。tau0.3 也<47.80 → A3 单模死，
              只剩 s45 soft-blend 能救（先离线查 raw 是否有 s15 漏掉的新正确少数类再烧枪）。
s45 > 50.26 ──► 新最优。顶回 s01 到 floor 位。叠 A1 自洽成员（试 stacked union，取 s49 新号）。进 A4（§6.4）。
s45 ∈ 50.0–50.26 ──► A3 成员与 s15 相关 / FP 翻段。查最相关成员、丢掉重 blend（取 s49 新号）；仍无升则 A3 用尽，
              不再烧枪，转 A4。
```

### 6.4 A4 — L0/G3 门 → s47（单模） → s48（union）

```
L0 KILL（≥3/4 dense<门）或 K-prereq（n_synth~0 / 无干净 E0-full）──► 0 枪，未训。记负，保 s01/s15。
              （此时若 A1/A3 已给出 >50.26，直接收尾；否则 Phase-2 prep。）
L1 SMOKE de-CoT 断言失败 ──► 立即 KILL（非静默放行）。修 launcher 重跑或弃 A4。
G3 失败（ranking/health 任一）──► K4：不烧枪。GRPO 模型记负（densematch 误报先例）。保 s01/s15。
G3 过，s47 < 47.80 但 s48 > 50.26 ──► 更可能的赢路实现！s47 记为异构成员，s48 升为新最优，s15=50.26 永久 floor。交 s48。
G3 过，s47 > 47.80 且/或 s48 > 50.26 ──► 全赢。两者记录。s01/s15 永久 floor（平台按最新则顶回）。
              文档化 per-para-norm + density-augment 配方为 breakthrough。
s48 ≤ 50.26 ──► GRPO 成员相关/翻段。回退 s15。A4 用尽。
```

### 6.5 全局收口

```
最终「当前最优」= max(s15=50.26, s42, s45, s48, s49+...)。
若任一 >53.3（+3）──► 达成主目标，确保 floor 提交在位，进 Phase-2 防过拟合。
若全部 ≤50.26 ──► 三杠杆均饱和/迁移失败。s15=50.26 保底为最终单/集提交。诚实记负，留 Phase-2。
```

---

## 7. Codabench 配额规划 + 风险与兜底

### 7.1 Codabench 配额规划

| 阶段 | 必交 | 条件交 | 期望枪数 |
|---|---|---|---|
| A1 | s41（GATE GO/SOFT-GO + within-s01 DELTA 正后） | s42（s41 真>47.80）、s43（K16>K8 罕见） | 1-2 |
| A3 | — | s44（Stage-A GO）、s45（s44 有新召回）、s46（tau 分离） | 0-3（典型 2） |
| A4 | — | s47（G3 全过）、s48（s47 过+union 离线升） | 0-2（modal 0） |
| stacked | — | s49+（A1×A3 / 改-config 重跑） | 0-1 |
| **合计** | | | **期望 3-5 枪** |

配额纪律：每枪前必 `validate_submission.py` 通过；ADD-mode——只在击败当前最优时交；KILL/门失败一律不烧枪（A3 离线 proxy 假、A4 densematch 泄漏，省下的枪靠门控守住）。

### 7.2 风险与兜底

| 风险 | 影响 | 兜底 |
|---|---|---|
| **A1 门跑错 adapter lineage（已修 FATAL）** | +1.4 证据来自 plain-CE per-sentence；原草稿误用 softmin cotE0/s40 → 测的是未验证量（softmin 锐化→采样多样性低→lift 缩水） | 门硬绑 plain-CE per-sentence train' adapter；缺则 plain-CE 重训，绝不用 softmin 替代（§3.0） |
| **迁移风险（headline）** | densematch→testp1 Spearman 仅 +0.59，不分辨 50.0-50.3，CS in-domain 测不了 OOD；s30 densematch#1 却 testp1 PEM 崩 | 离线 lift 只作粗二元 GO 信号，**非分数预测器**；所有 sub-point 绝对判定落 Codabench；s41 一枪 mandatory 后才信 s42 |
| **softmin 生产模型熵低、lift 不迁移（已修 SERIOUS）** | plain-CE 门的 lift apply 到 softmin 近记忆 s01，s01 同温度分歧位少→真实 lift 缩水 | §3.5b 在 s01 自身测 within-model 相对 DELTA；分歧率 <门一半则提 s01 的 T 匹配多样性再选 K |
| **离线门阈值落在不可分辨区（已修 SERIOUS）** | 50.0/50.26/50.3 细分超过 +0.59 台子分辨力；单点 +1.4 无方差 | 铁律 §1.3.7：50.0-50.3 一律 Codabench-only；离线只粗二元（双 block × 双 metric 同号 + ratio≥1.5） |
| **FP/PEM 风险** | --min-votes 1 / soft-blend / margin 过冲 在分歧位引 FP，一句翻一段（s21=47.77, s30/s37） | 含 greedy 成员（对 greedy 纯加法）+ gate PEM-non-regression + firing ratio≥1.5；A3 走 soft 非 hard-OR；tau 小+asym clip |
| **泄漏误用** | 在 full-traindev adapter 上跑门 → densematch 虚高、GO 无效 | 门硬绑 train'-only adapter（A1=plain-CE train', A3=devbench A3 regime-a, A4=见下）；full-traindev 模型只做生产采样 + within-model 相对 DELTA，不读其 densematch 绝对值 |
| **A4 L0 也对 E0-full 泄漏（已修 SERIOUS）** | densematch⊂full-traindev，E0-full 见过这 206 条→pass@8 高估真实 dense 采样能力→false-GO ~80% GPU | L0 优先在 image-disjoint held-out 测；否则当 LEAKY UPPER BOUND，GO 门槛抬到 ≥35%，登记 L0 泄漏 |
| **A4 G3 子门 #2 挂不存在的台子（已修 SERIOUS）** | 「no-leakage testp1-shaped held-out」在仓库不存在→门不可满足→悄悄豁免回 densematch-only(=误报 s30) | 默认决议 (B)：DROP 子门 #2，G3 = densematch ranking + health-only，承认 testp1 一枪是唯一真信号 |
| **seed 坍缩（A1 静默 no-op）** | PATH A 漏 --seed → K 个同样本 | PATH A 永远传相异 --seed；PATH B（--n>1，需 temp>0）不靠 seed 多样、不与 PATH A 混用 seed 语义（§3.1） |
| **id-set mismatch / 全局 split.json 污染** | --split dev 覆盖 501，densematch 是 206 子集；cp 改全局 data/split.json 留 dangling 态 | 全程用 `--records-file $DENSE`（不改全局态）+ `aggregate --ref dev_gold_densematch.jsonl` |
| **JOINT 陷阱（A1）** | s01 是 per-sentence softmin，传 --joint 会走 joint 模型路 | A1 严格 per-sentence，不传 --joint（JOINT 自洽 E1b 已失败） |
| **A3 proxy 假 / use_logits 陷阱** | infer 不存 5-way logits → 无法 post-hoc A/B；retrain_fullres:41 硬编码 use_logits=true | A3 每次 fresh train；run log 必 grep use_logits_to_keep=false（loss.py:64） |
| **A4 dense pass@8~0（dominant）** | RL 只放大不创造；E0-full 几乎不采到 dense 段第二少数类 → 整路 no-op | L0 先跑、便宜 KILL；信 near-0 结果不合理化 |
| **A4 四前置全未满足 + P0/P2 耦合** | train_b1_joint 缺、outputs 空、gold 单元素、prompt 源是不存在的 CoT 集；make_sample_joint 无 full types[] | 按 §5.2 顺序修；先 make_sample_joint sidecar 再 --density-augment；n_synth==0 即停 |
| **A4 静默重入 CoT（已修 SERIOUS）** | train_grpo.sh 硬编码 `<analysis>`/E1-default/maxlen160；未改即跑 = 重入 -5.31 CoT regime | L1 SMOKE 前 grep 断言无 `<analysis>` + maxlen≤64 + ADAPTER=E0-full；SMOKE 后 dump rollout 验 plain `{label}`；任一失败=立即 KILL |
| **硬编码 ckpt 路径（已修 SERIOUS）** | outputs/ 本地仅 .gitkeep；S01/E0/E0FULL 路径带 v0-* 时间戳，字面写死会指向缺失 adapter | 全部用 `ls -d .../v*/checkpoint-* | sort | tail -1` 解析 + STEP -1 断言存在再动 GPU |
| **饱和 / 天花板** | 三杠杆都加召回可能重叠不叠加，撞 s15=50.26 naive-union 天花板 | 必须 COMBINE 去相关杠杆；每步查 raw 重叠（新正确 firing 是否相互独立）再决定叠不叠 |
| **deadline 2026-06-17 + Phase-2 GPU 优先** | A4 全链 ~15-27 GPU-h（2 GPU），紧 | K5 硬 deadline；L0 立即起、返 GO 才推进；逼近则弃 A4，s01/s15 为 standing 提交 |
| **ms-swift gotchas** | USE_HF=0 不设→挂 HF 下载；--vllm_enable_lora false（#6670/#6506）；grad_accum≤8（#6521） | 全部 baked-in，勿删 |
| **ledger 漂移（铁律 §1.3.5）** | stacked / 改-(T,K) 重跑若覆盖既有 s号 → 复现表失真 | 任何新 Codabench 上传取 s49+ 新号 + 新 log 行；已编号模型不回改 decode/成员 |

### 7.3 成本汇总

- **A1**：GATE（plain-CE train', 206 rec）greedy1 + 3T×16seed=49 dev-infer ≈ 6-9 H200-h（K∈{4,8} 离线子采样免费）；within-s01 DELTA(§3.5b) K8+greedy ≈ 1-2 H200-h；生产 s01 testp1 9 run ≈ 2-3 H200-h；s42 5 成员×8 ≈ 8-12 H200-h。总 ~17-26 H200-h。
- **A3**：Stage-A ~3-5 GPU-h（+3h 若 tau0.3 重试）；KILL ≤5 GPU-h 0 枪；Stage-B ~3-4 GPU-h；soft-blend CPU 分钟级。GO 路 ~7-9 GPU-h + 2-3 枪。
- **A4**：L0 ~3-5h（gate ~80% 后续；若造干净 held-out 另加切分成本）；P1 E0-full SFT ~3h；P0 分钟级 CPU；L1 SMOKE ~40min；L2 ~8-18h。总 ~15-27 GPU-h（2 GPU）；modal 0 枪（L0/G3 KILL）。

---

**执行顺序**：解析+断言 ckpt（STEP -1）→ A1 GATE 在 **plain-CE per-sentence train' adapter**（最便宜、+1.4 同 lineage）→ §3.5b within-s01 DELTA 确认 → s41 →（按 §6.1 决定 s42）→ A3 Stage-A（同-regime 相对 vs s31）→ s44/s45 → A4 L0 pass@8（leak-aware；near-0 即 STOP）→ de-CoT 断言 → G3(ranking+health) → s47/s48。每步严守 ADD-mode、train'-only 门控、Codabench-only 绝对判定、ledger 升号纪律，s01=47.80 / s15=50.26 永不可破。
