# 环节① CoT 教材蒸馏计划（造教材）

> 总概 `notes/cot_grpo_roadmap.md`；下游 ②`notes/cot_sft_plan.md`（s01 配方+CoT）/ ③`notes/grpo_plan.md`。
> 2026-06-10。目标：用 gpt-5.5（aiapis.help 代理，无限额度，已验证 text+vision 通）为 **devbench train'（2832 records / 15008 句）** 的每个句子蒸馏一条「接地的推理链 + gold label」教材，供后续 CoT-SFT（点火）→ GRPO（学方法）使用。
> 复用已打通链路 `scripts/gpt5_infer.py`（代理调用 / UA 绕 Cloudflare / b64 图片 / 多线程 / 断点续跑 / raw 落盘）。
> 规则合规：指南明确允许商业 API，须报告 model/version/cost（无限额度也要记录调用量与模型版本字符串）。

## 0. 与 gpt5_probe 结论的关系（为什么这不是重走死路）

`notes/gpt5_probe_result.md`（2026-06-04）已证伪「gpt5 当**分类器**」：零样本 25.2 ≪ 8B 微调 47.8，因为它缺**标注惯例**；但同时证实了它**真能读出图里的数字**（"clean 82.6 vs 85.6"、"横轴只到 45"——8B 读不出的能力）。

本计划是 probe 留下的「残值」路线的实现，分工与 probe 的失败模式正好互补：

| | 谁提供 | 说明 |
|---|---|---|
| **标注惯例（label 边界）** | **gold**（拒绝采样筛 / gold 条件化生成） | probe 证明惯例 prompt 补不上 → 不让 gpt5 判，只让 gold 判 |
| **证据接地（读数/读实体）** | **gpt-5.5** | probe 证明这是它的真实强项，也是 8B 的真实短板 |

⇒ 教材 = gpt5 的「读图能力」× gold 的「惯例」。gpt5 分类准确率低（~60% 句准）只影响 Stage A 的**产率**，不污染教材质量（错的被 gold 筛掉）。

## 1. 数据范围与红线

- **输入 = train' 全量**：`data/devbench/split.json` 中 value=='train' 的 2832 条 record（id = `track1-{行号:06d}` 对 `$DATA_ROOT/data/traindev-track-1.jsonl` 的行索引）。
- train' 实测构成：15008 句；primary 少数类 932（SO 445 / Contra 190 / UE 177 / UCM 120）；any-of 少数类 ≈1455；多标签句 437；段长 mean 5.3 / max 22 / ≥6 句的段 1119 个。
- **多标签句的 target_label = types 中全局最稀有的那个**（与 `build_dataset.pick_rarest_label` 一致，利好 macro-F1）。
- 🔴 **红线 1：本计划绝不送 dev'（501 条）**。脚本入口处 assert 所有输入 id ∈ train'；产物落盘前再 assert 一次。违反 = devbench 评测台报废。（唯一例外：②的 **Track B 冲榜版**过 G2 闸后，由 ② 显式触发对 dev' 的独立补跑——产物落 `units_dev.jsonl` 单独文件、单独 push，与 train' 教材物理隔离，且 devbench 模型永不读它。）
- 🔴 **红线 2：few-shot 样例必须来自 train'**。`gpt5_infer.py` 的 `FEWSHOT_BLOCK` 6 条样例当时是按"真 gold"随手挑的，**复用前必须核对出处**；不确定就从 train' 重挑 6 条（覆盖 5 类 + 1 条 Supported 易混句）。
- 数据政策：官方数据仅限本赛事使用；代理是第三方服务，**prompt 里只含本赛事允许使用的数据**，产物（推理链）属派生数据，落 `data/cot/`（gitignored），不外发。

## 1.5 数据链路（HF 统一数据仓 + track1/ 规范）

```
NLPCC 官网/官方 git（唯一原始数据源，图片走 git lfs）
   │  每台机器各自 clone ../NLPCC-2026-Task10-Science（原始数据不经 HF）
   ▼
处理机（任意服务器；本计划默认 H200，因需图片）
   │  中间产物 push
   ▼
HF winbeau/nlpcc2026-task10（private）── track1/data/...
   │  pull
   ▼
算力服务器（训练）
```

- **HF dataset 仓 = 唯一数据中转仓**（`winbeau/nlpcc2026-task10`，repo_type=**dataset**，与同名 model 仓并存互不影响；private）。**所有上传一律进 `track1/` 前缀**，已建结构（2026-06-10 首推完成）：

```
track1/
├── raw/                  # 官方 track-1 jsonl + manifest（traindev/testp1；✅已传 10MB）
└── processed/
    ├── devbench/         # split.json + dev_gold*.jsonl + train/dev_sft.jsonl（✅已传 49MB）
    └── cot/              # 本计划全部产物（见 §5，产出后传）
```

- **images 不进 HF**（图片各机器从官方 git lfs 拉；本地 zips 是 LFS 指针，脚本自动识别跳过）；仓库保持 **private**、仅限本赛事使用，raw 上传是赛内跨机中转、不是再发布。
- 同步工具：**`scripts/hf_data.py`（已建并验证）**，任何机器零项目依赖可跑：
  ```bash
  uv run --no-project --with huggingface_hub python scripts/hf_data.py push --what raw devbench cot [--dry-run]
  uv run --no-project --with huggingface_hub python scripts/hf_data.py pull --what devbench cot
  ```
  token 在 `~/.cache/huggingface/token` 或 env `HF_TOKEN`。
- **每个 Stage 完成即 `push --what cot`**：断点续跑的进度在 HF 始终有副本，处理机可随时换。

## 2. 三阶段流水线

教材单元（产物 schema，`data/cot/units.jsonl`，一句一行）：

```json
{"record_id": "track1-000123", "sent_idx": 2, "sentence": "...",
 "gold_types": ["Unsupported Entity", "Supported"], "target_label": "Unsupported Entity",
 "rationale": "claim cites CoRTX vs L2X on Table 6; provided table has no CoRTX entry",
 "source": "blind|gold_cond", "verified": true, "verifier_note": "",
 "model": "gpt-5.5", "ts": "...", "n_tries": 1}
```

rationale 约束：**英文 ≤45 词**、必须引用证据中**具体的数值/实体/轴范围**、**纯推理文本不带 label 尾巴**（Stage A 解析时剥掉行尾 `-> <label>`；label 单独存 `target_label`）。45 词上限对齐 ②§1 的 per-sentence 目标格式（`<analysis>` 段 ~60 token/句，decode 开销可控）。**同句多条合格 rationale 全部保留**（units.jsonl 允许同 (record_id, sent_idx) 多行）——②的过采样复制样本轮换填充，防 analysis 模板化。

### Stage A — 盲判联合采集（per-record，k=2 采样）

- 直接复用 `gpt5_infer.py` 的 prompt 骨架（重标定 + few-shot 版，probe 实测最好），它**本来就逐句输出** `i) claim: ...; evidence: ...; verdict: <label>` + `FINAL:` 行——即每句一条天然推理链。
- 每 record 采 **k=2**（无限额度；temperature 默认即可，两次独立调用）。
- **gold 筛（拒绝采样）**：verdict==target_label（或 ∈ gold_types）的句子，其 reasoning 行收为教材；两条都对取更"接地"的一条（引用了具体数字/实体者优先，简单规则：含数字 > 不含）。
- 预期产率（按 probe 句准 ~60%）：Supported 句大部分收齐；少数类句大部分**收不到**（gpt5 惯例错位）→ 留给 Stage B。
- 调用量：2832 × 2 ≈ **5.7k 次**。

### Stage B — gold 条件化补漏（per-sentence，用户定逐句送）

- 对 Stage A 没收到教材的句子（预估 ~5-6k 句，含绝大多数少数类）逐句调用：prompt 给出证据图 + claim 全文 + 目标句 + **直接告知 gold label**，要求"写出把证据与该 label 连起来的推理，必须指向证据中具体的数值/实体/范围"。
- **CANNOT_JUSTIFY 逃生口（防瞎编的关键）**：明确指示"若在证据中找不到支撑该 label 的具体依据，输出 CANNOT_JUSTIFY 而不是编造"。输出 CANNOT_JUSTIFY 的句子进 `label_only` 桶（第二步训练时这些句子退回纯 label 目标，不配推理链）——**宁缺毋滥**。
- 失败重试 1 次（换措辞 re-prompt），再失败也进 `label_only`。
- 调用量：~6k × 1.3 ≈ **8k 次**。

### Stage C — 审稿核验（接地质检，防"圆而不真"）

- 对象：**全部少数类教材（A+B 产出，~1.4k 条）+ Supported 教材随机 10% 抽检（~1.3k 条）**。
- 第二遍独立调用（带图）："下面这段推理引用了哪些数值/实体？逐一核对是否真的出现在证据图表中，输出 PASS/FAIL + 理由"。FAIL → 少数类教材回炉 Stage B 重写一次，再 FAIL 进 `label_only`；Supported 抽检 FAIL 率 >10% 则扩大抽检到全量。
- 这是防御 GRPO 阶段最阴险的风险：**reward 只看 label 不看推理真假，瞎编的推理也能领奖**——所以瞎编必须在教材入库前杀掉。
- 调用量：~3k 次（+回炉 ~0.5k）。

**总调用量 ≈ 17k 次**。worker=6、~20s/次 → 单阶段全速 ~16h；A/B/C 串行依赖，但 A 跑完即可边筛边启 B。在 **H200 box** 跑（图片已解压、`gpt5_infer.py` 即为此设计）。

## 3. 试点闸门（先烧 50 条，再烧 17k）

正式跑前在 train' 随机 50 records（强制包含 ≥30 个少数类句）走完 A→B→C 全流程：

| 检查项 | 通过标准 | 不过怎么办 |
|---|---|---|
| Stage A 句级 agreement | ≥55%（probe dev 上 60%，应接近） | 检查 prompt/图路径/解析，别带病全量跑 |
| Stage B CANNOT_JUSTIFY 率 | <15% | >15% → 改 B prompt（如给出该类的 2 条 train' few-shot）再试 |
| Stage C 少数类 PASS 率 | ≥80% | 低 → rationale 约束加严（强制引用格式）|
| 人工抽 20 条（必含 10 少数类） | 引用的数字/实体肉眼可在图中确认；无编造；≤45 词 | 按失败模式改 prompt |
| 吞吐 | 实测 calls/min → 修正全量耗时估计 | 调 workers / 分批 |

试点产物落 `data/cot/pilot/`，结论追加到本文档 §7。

## 4. 实现（新脚本 `scripts/cot_distill.py`）

- 单脚本三模式：`--stage blind|gold|verify`；公共部分（`call_api` / `b64_image` / 断点续跑 / raw 落盘 / 线程池）直接 import 或拷自 `gpt5_infer.py`（不动原脚本——它还是 testp1 推理工具）。
- 入参：`--split-json data/devbench/split.json --data-root $DATA_ROOT --out data/cot/ --workers N --limit/--records-sample`（试点用）。
- env 同链路：`GPT5_KEY`（必须）、`GPT5_BASE`（默认 aiapis.help/v1）、`GPT5_MODEL`（默认 gpt-5.5）、`GPT5_REASONING_EFFORT`（可选）。
- 解析健壮性沿用 `parse_labels` 思路；Stage A 逐句行用 `^\s*(\d+)\)` 切分对齐句号，**对不齐的 record 整条丢弃重采**（教材容不得错位）。
- 全程双落盘：`*.raw.jsonl`（完整模型输出，审计用）+ `units.jsonl`（结构化教材）。
- 统计脚本 `--stats`：逐类教材覆盖率 / label_only 占比 / 平均 rationale 词数 / Stage C PASS 率 → `data/cot/stats.md`。
- 调用量计数 → 报告义务（model/version/调用次数）。

## 5. 产物与下游接口（第二步 CoT-SFT 怎么吃）

```
data/cot/                  (gitignored；⇄ HF track1/processed/cot/ 双向同步)
├── pilot/                 # 试点全套
├── blind.raw.jsonl        # Stage A 原始输出
├── gold_cond.raw.jsonl    # Stage B 原始输出
├── verify.raw.jsonl       # Stage C 原始输出
├── units.jsonl            # ★ 教材：句级 (record_id, sent_idx) → rationale+label
└── stats.md               # 覆盖率/质量统计
```

- 本地 `data/cot/` ⇄ HF `track1/processed/cot/`；`data/devbench/` ⇄ `track1/processed/devbench/`（已传）。
- 算力服务器训练前只需 `hf_data.py pull --what devbench cot`，不依赖处理机。

- **下游目标格式以 ②§1 为唯一标准（per-sentence，对齐 s01 配方）**：②的 builder（`build_dataset --cot data/cot/units.jsonl`）按 (record_id, sent_idx) 查 rationale，组装 `<analysis>{rationale}</analysis>\n{"label": "<target_label>"}`；`label_only` 桶的句子保持 s01 原版纯 JSON 目标。**末行 JSON 与 s01 完全相同**，推理解析只需改「取末行」。
- **与 P1 密度匹配合流（②E2）**：教材是句级单元，跨 record 拼接合成密集段时每个句子自带 rationale，per-sentence 展开后 TARGET 句的 claim 上下文变密、目标照常拼装——不需要为合成段重新调 gpt5。
- 教材同时定义 ③GRPO 的 rollout 格式（同 ②§1，reward 解析末行 JSON）。joint 段落组装（`i) ... -> label` + FINAL 行）留作后续 joint+PEM-bonus 变体，本期不用。

## 6. 风险表

| 风险 | 暴露点 | 对策 |
|---|---|---|
| gold 条件化诱导瞎编（圆而不真） | Stage B | CANNOT_JUSTIFY 逃生口 + Stage C 全量少数类核验 + label_only 兜底 |
| 教材只教会"风格"不教会"边界" | SFT 后 | 边界仍由 label 监督 + （第二步）densematch 段落几何给；rationale 只负责接地，预期不损害惯例 |
| 逐句行与句号错位 | Stage A 解析 | 对不齐整条丢弃重采，绝不 best-effort 对齐 |
| 代理不稳 / Cloudflare / 限流 | 全程 | 已有 retries+UA+resume；试点实测吞吐；分批跑 |
| 少数类教材覆盖不足（CANNOT_JUSTIFY 偏多） | UCM/SO 这类"边界模糊"类 | 该类 few-shot 注入 B prompt；最后兜底 label_only（与现状等价，不更差） |
| few-shot 样例混入 dev' | 红线 2 | 跑前核对/重挑，写进脚本 assert |
| 无限额度 ≠ 无限速率 | 全量 17k 调用 | 试点测 calls/min；必要时夜间分批 |

## 7. 试点结果（待填）

- [ ] Stage A agreement = ?
- [ ] Stage B CANNOT_JUSTIFY 率 = ?
- [ ] Stage C PASS 率 = ?
- [ ] 人工抽检结论 = ?
- [ ] 实测吞吐 = ? calls/min → 全量预计 ? h
- [ ] GO / NO-GO：

## 8. 执行顺序

0. ~~写 `scripts/hf_data.py`、建 `track1/` 结构、push raw+devbench~~（✅ 2026-06-10 完成，见 §1.5）
1. 核对/重挑 few-shot 6 例（train' 内）→ 写 `cot_distill.py`（半天）
2. 试点 50 records 全流程 + 人工抽检 → 填 §7 闸门 → push `track1/data/cot/pilot/`（半天）
3. GO → Stage A 全量（夜间）→ push → 筛 → Stage B（夜间）→ push → Stage C → `stats.md` → push
4. 教材交付环节②（算力服务器 `hf_data.py pull --what cot` → builder 对接、密度合流、Track B）——见 `notes/cot_sft_plan.md`
```
与 P1 的关系：本步与 P1 数据工作完全并行，不抢 GPU（纯 API + 本地 CPU）；P1 的 densematch SFT 是 GRPO 的前置，本步教材是 CoT-SFT 的前置，两者在第二步合流。
```
