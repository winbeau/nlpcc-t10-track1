# CLAUDE.md — NLPCC 2026 Task 10 / Track 1 解题仓库

> 本文件每个 session 自动载入。改动时保持精简（只放跨 session 必须知道的事实与约束）。
> 全局 `~/.claude/CLAUDE.md` 仍生效；本项目级文件优先级更高。

## 0. 这是什么

NLPCC 2026 Shared Task 10 **Track 1**：*Claim-level Faithfulness to Experimental Results*（科学报告的可靠性 / 声明级忠实度）的参赛仓库。

- 官方题目仓库（**只读，不要改**）：`../NLPCC-2026-Task10-Science/`
  - 指南：`GUIDELINES.md`（EN）/ `GUIDELINES_ZH.md`（中）
  - 数据：`data/`（train-dev + Phase 1 test，图片在 Git-LFS）
  - 官方评测器：`offline_eval/evaluate.py`（**唯一权威打分**，必须用它）
  - 基线 prompting kit：`baseline_prompting/`
- 目标：冲榜。**当前最优 testp1 = s15 ensemble 50.26**（最佳单模型 grid s01=47.80）；testp1 的 trivial≈19（**不是** 43.2，见 §4）。
- Phase 1 提交平台：Codabench 16666。Phase 2 隐藏测试集 2026-06-11 放出，**所有结果 2026-06-20 截止**。

## 1. 硬约束（务必遵守）

1. **绝不在本地下载模型权重 / 跑 `uv sync` / 跑训练或推理 / 解压图片。** 一切重活在**远程 GPU 服务器**上做，通过 **tmux + ssh**（用 `tmux-ssh-remote` skill）。本地只做：读数据、写代码、设计、轻量纯 Python 分析。
2. **用 `uv` 管理项目**（`pyproject.toml`）。重依赖在 `[project.optional-dependencies].train`，**只在服务器** `uv sync --extra train`。
3. **数据使用政策**：官方数据仅限本赛事使用，**不得转发 / 镜像 / 改名再发布**。不要把 `data/` 或图片提交进 git。
4. 官方数据是 sibling 仓库，通过 `DATA_ROOT`（默认 `../NLPCC-2026-Task10-Science`）引用，**不复制**。派生产物（句子级数据、切分、预测、checkpoint）放本仓库 `data/`、`outputs/`，均已 gitignore。

## 2. 任务定义（Track 1）

每条样本：`claim_text`（一段 2–7 句的声明）+ `evidence_bundle`（图/表证据，见 §3）+ 已切好的句子列表。
**对每个句子输出 1 个 label**（5 类之一）：

| Label（**字符串必须逐字一致**） | 含义 |
|---|---|
| `Supported` | 句子被证据充分支持 |
| `Unsupported Causal Mechanistic` | 引入证据未支持的因果/机制解释 |
| `Unsupported Entity` | 提到证据未涉及的数据集/指标/模型变体/baseline 等实体 |
| `Scope Overgeneralization` | 把结论外推到证据支持范围之外 |
| `Contradiction` | 与证据直接矛盾 |

**提交格式**（JSONL，一条 record 一行）：
```json
{"id": "track1-p1-test-000008", "labels": ["Supported", "Contradiction", "Supported"]}
```
- `labels` 数量必须 == 该 record 句子数，**顺序与输入句子一致**。
- 测试集自带 `id`（如 `track1-p1-test-000008`），评测**按 id 匹配**，务必原样回填 id。

## 3. 数据关键事实

- 路径：`$DATA_ROOT/data/{traindev,testp1}-track-1.jsonl`；图片在 `images.zip` / `images-testp1.zip`（**LFS，需在服务器 `git lfs pull` 后解压到 `images/`**，使 `images/<sha>.jpg` 可解析）。
- train-dev：**3333 records / 17547 sentences**；Phase 1 test：**586 records / 5096 sentences**。
- **所有证据都是图/表**（train-dev 中 `image:2017`、`table:1979`，**没有任何 `text` 类型**；每条 evidence 都有 caption，平均 caption 276 字符）。→ 这是用 **Qwen3-VL（多模态）** 的核心理由：真正的表格数字在图里，caption 往往不够。
- 平均 1.2 个 evidence/record（最多 7）；`claim_text` 中位 671 字符。
- **真实 label 分布**（注意：纠正了原计划里 "Unsupported Entity 276" 的笔误）：

| Label | primary (`types[0]`) | any-of `types`（=GUIDELINES 表） |
|---|---:|---:|
| Supported | 16438 (93.7%) | 16438 |
| Scope Overgeneralization | 540 | 605 |
| Contradiction | 216 | 216 |
| Unsupported Entity | **213** | **639** |
| Unsupported Causal Mechanistic | **140** | 276 |

- ⚠️ **traindev jsonl 按行序前 1/3 集中全部少数类**（decile 0-2 各 ~17.5% 少数类句、decile 3 收尾 6%、decile 4-9 为 0）——任何按行序抽样/截断/顺序处理的脚本都会踩中这个偏置（2026-06-10 实测）。
- **527 句是多标签**。评测规则：预测命中 gold `types` 中**任意一个**即算对；算 F1 时若命中用命中的那个作 reference，若错则用 `types[0]`。
  - → 多标签句构造训练目标时**优先选少数类**（更有利于 macro-F1 学习），单标签句直接用该 label。

## 4. 评测与打分策略（最重要）

`offline_eval/evaluate.py --track 1`：
- **Sentence Macro-F1**：5 类 F1 等权平均。
- **PEM (Paragraph Exact Match)**：整段所有句子全对才算 1。
- **score = (Macro-F1 + PEM) / 2**。

**核心洞察（已由 16+ 次 testp1 提交证伪/证实，详见 `notes/submissions_log.md`）：**
- ⚠️ **原计划的「precision-critical / 抑制少数类保 PEM」框架已被推翻。** 那是基于 train-dev 的 trivial（67% 全-Supported 短段 → PEM 0.67 → score 43.2）。**testp1 完全不同**：段落长（均值 8.7、最长 31 句）、少数类密集（~84% 段含 ≥1 gold 少数类），**全-Supported trivial ≈ 19 分**（实测 s05=18.85）。
- ⇒ **testp1 是 RECALL-critical**：拿 PEM 必须 catch 到长段里的 gold 少数类句。**抑制少数类（veto / 置信阈值 / 降过采样 / ≥2 票一致）在 testp1 上单调掉分。** PEM 也是召回受限，不是精度受限。
- **胜负手 = 最大化「正确的」少数类召回。** 已验证**唯一**超 grid 的杠杆 = **多样模型的并集 ensemble**（`scripts/ensemble_union.py --min-votes 1`；成员越多样，Macro-F1 和 PEM **同时**涨）。
- **当前最优：s15 = 50.26**（grid + 多样 8B/32B 的 5 模型并集）；最佳单模型 grid s01 = 47.80。
- **已证伪（别再走）**：提分辨率（−3.3）、降过采样（1.5×）、逐类过采样、2 epoch、段落联合格式（单模型 ≤grid，32B 比 8B 好 +1.1 但仍 <grid）、所有抑制类后处理。

评测命令见 §7。

## 5. 方法（已定方案）

| 决策 | 选择 |
|---|---|
| 模型 | **Qwen3-VL-8B** + **LoRA**，bf16 |
| 训练框架 | **ms-swift**（ModelScope，对 Qwen3-VL 一等支持）|
| 建模粒度 | **句子级**：每个训练样本 = 共享 evidence/claim 前缀 + 1 个 TARGET 句子 → 输出 1 个 label |
| 数据扩展 | 3333 records → ~17547 句子级样本（推理时同一 record 的 evidence/claim 前缀做 **KV-cache 复用**，n 句只算 1 次前缀）|
| 类别不均衡 | **过采样 4 个少数类 + 下采样 Supported**（比例按 §4 的 dev score 调，防 PEM 崩）|
| 验证切分 | **按 record 9:1**（不可按句切，否则同段句子泄漏、且 PEM 无意义）；尽量按"是否含少数类"分层 |

**训练样本格式**（chat / ms-swift messages）：
- `system`：careful scientific claim verification system，输出严格 JSON。
- `user`：label 定义 + `evidence_bundle`（图片 + caption）+ 完整 `claim_text` + **TARGET sentence**（指明判这一句）。
- `assistant`（label / 训练目标）：`{"label": "Unsupported Entity"}`。

**推理 / 集成（已验证的赢家路径，呼应 §4）**：
- ⚠️ **不要做「偏向 Supported / 阈值弃权」的抑制收口——已证伪会掉分。**
- **赢家 = 多样模型并集 ensemble**：训多个差异化模型（种子 / 基座 / 格式 / 数据切分），`scripts/ensemble_union.py --min-votes 1` 取并集（任一模型开少数类即采纳）。`--min-votes 2`（≥2 票）更差。
- **段落联合格式**（`build_dataset --joint` + `scripts/train_joint.sh`，纯 CE、one-forward/record）已打通：8B≈24G、**32B≈70G** 显存，是上 32B 的唯一前置（per-sentence 32B 因每句重复图前缀会 OOM ~200G）。单模型 ≤grid，但作 ensemble 的**异构成员**有用。
- `use_logits_to_keep`：softmin 必须 false（因果位移对齐）；联合纯 CE 可 true。提分辨率（max_pixels↑）已证有害，保持 401408。

## 6. 仓库结构

```
nlpcc-t10-track1/
├── CLAUDE.md / pyproject.toml
├── src/nlpcc_t10/
│   ├── build_dataset.py   # 9:1 切分 + 展开 + 重采样;支持 --joint(段落联合) / --minority-oversample-per-class(逐类)
│   ├── infer.py           # 推理 → 句子级 raw(含 min_logprob);支持 --joint(一记录一请求,解析 labels 数组)
│   ├── aggregate.py       # 句子级 → {id,labels}(保序/数量对齐/Supported 兜底)
│   ├── eval_local.py      # 调官方 evaluate.py 打分
│   └── swift_softmin/     # ms-swift 4.2.3 softmin-PEM loss + paragraph sampler 集成
├── scripts/
│   ├── train_softmin.py + retrain_fullres.sh   # per-sentence softmin LoRA(+infer+zip);TAG/EPOCHS/MAX_PIXELS env
│   ├── train_joint.py + train_joint.sh         # 段落联合 plain-CE(+infer --joint+zip);MODEL_ID 可换 32B
│   ├── ensemble_union.py        # ★赢家:多模型并集(--min-votes K)
│   ├── threshold_variants.py / lone_minority_veto.py   # 后处理(已证伪,留档)
│   ├── make_submission_zip.py / validate_submission.py # 扁平 zip(顶层 track1_pred.jsonl)+ 校验
│   └── grid_search.py / smoke_full_res.sh / overnight_recall_queue.sh
├── submissions/  # testp1_s{NN}_{tag}.{zip,jsonl}(已 git-track;命名规范见 §8)
├── notes/submissions_log.md   # ★权威:得分总表 + 逐项复现 + 命名规范
├── notes/overnight_plan.md / analysis.md
├── data/   (gitignored)  # 派生:train_sft.jsonl / split.json / dev_gold.jsonl
└── outputs/(gitignored)  # LoRA adapter checkpoint / raw 预测
```
- **模型/提交归档已上传 HF**:`winbeau/nlpcc2026-task10`(private;**17 个 adapter**(s01/02/07/08/09/10/12/13/26 softmin + s33-s36 plain-CE + phaseC_A0/A1/B0/B1 devbench)+ **submissions s01–s38** + scripts + `SUBMISSIONS.md`(=notes/submissions_log.md,逐提交复现表)+ README)。**增量重传:`uv run python scripts/hf_push.py`(在服务器跑;只传 adapter_model.safetensors+configs,不传 optimizer;`--dry-run` 预览)。** 加新模型先把 ckpt 路径加进 `hf_push.py` 的 `ADAPTERS` dict。
- **HF 数据链路(2026-06-10 起,已建)**:同名 **dataset 仓** `winbeau/nlpcc2026-task10`(private,与 model 仓并存)作跨服务器**数据中转仓**——处理可在任意机器,中间产物 push,算力服务器 pull 后训练。**一律进 `track1/` 前缀**:`track1/raw/`(官方 track-1 jsonl,已传)+ `track1/processed/{devbench,cot}/`(devbench 已传;本地 `data/devbench|cot/` 一一对应)。同步:`uv run --no-project --with huggingface_hub python scripts/hf_data.py push|pull --what raw devbench cot`(token 在 `~/.cache/huggingface/token`)。🔴 images 不进 HF(各机器从官方 git lfs 拉;脚本自动跳过 LFS 指针)。详见 `notes/cot_distill_plan.md` §1.5。

## 7. 端到端流程（命令都在服务器上跑）

```bash
# 0) 一次性：环境 + 数据（服务器）
bash scripts/setup_env.sh        # uv sync --extra train; git lfs pull; unzip images

# 1) 构造句子级训练/验证集（9:1，重采样）
uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data/

# 2) LoRA 训练（ms-swift）
bash scripts/train.sh            # 读 configs/qwen3vl_lora_sft.yaml

# 3) 推理（dev 或 testp1）
bash scripts/infer.sh dev        # 或: bash scripts/infer.sh testp1

# 4) 聚合成提交格式
uv run python -m nlpcc_t10.aggregate --pred outputs/<raw>.jsonl --out outputs/submission.jsonl

# 5) 本地评测（仅 dev，有 gold 时）—— 用官方评测器
uv run python "$DATA_ROOT/offline_eval/evaluate.py" --track 1 \
  --gold data/dev_gold.jsonl --pred outputs/dev_submission.jsonl --match id
```
- **永远用官方 `offline_eval/evaluate.py` 打分**，不要自己另写指标。
- testp1 提交：跑 §3 的 `testp1-track-1.jsonl` → `aggregate` → 传 Codabench。

### 7.1 Windows 桌面通知（notify-win，关键节点主动弹）

本机长任务多在远程 GPU 后台跑（训练/推理/poll），完成往往隔很久。**关键节点主动给用户弹一条 Windows 通知**，不要默默等：
- **何时弹**：远程长任务（训练 / testp1 推理 / 8-sample 自洽）跑完、关键实验结果落地（如某 sNN 出分、对照结论确定）、或任务失败需立即注意。**短轮询/中间进度不弹**（避免刷屏）。
- **怎么弹**：`notify-win -t '<短标题>' -m '<结果/关键数字一两句>'`（成功省 `-s` 用默认；失败加 `-s Alarm2`；参数用单引号；中文/emoji 安全）。退出码 0=投递成功，255=Windows 不可达。
- 详见用户级 skill `~/.claude/skills/notify-win/SKILL.md`。例：`notify-win -t 'Track1 出分' -m 's41=47.76≈s01,A1 自洽 testp1 NULL'`。

## 8. 约定与坑

- **label 字符串逐字一致**（含空格大小写），见 §2；非法 label 会被评测器直接报错。
- 提交 `labels` 数量必须 == 句子数，**顺序不能乱**；缺一条 record 评测会报 id 集合不一致。
- id：测试集自带，原样回填；本地 9:1 dev 用 `prepare_dev_eval.py` 风格的 `track1-NNNNNN` id 生成 gold。
- 模型解析失败/越界标签时，**fallback 到 `Supported`**（最安全、保 PEM）。
- ⚠️ **不要信 9:1 dev（=88）选型**：32% 记录跨切分共享图片（泄漏）+ 分布失配（短/少数类稀 vs testp1 长/密）→ 排不了模型、也没预警到 fullres 掉分。**真信号 = Codabench 排行榜**（上限 100）或无泄漏的 testp1-shaped 子集。
- **提交命名规范**：`testp1_s{NN}_{T}_{desc}.{zip,jsonl}`，`s{NN}` = 按创建顺序的稳定序号，`{T}` = 类型(`m`=单模 / `pp`=后处理 / `u`=union)，`{desc}` = 简短描述(desc 内用 `-`，`_` 只分隔三段)。新提交取下一个号 + 定类型 → 重命名 → 追加 `notes/submissions_log.md` 统一总表一行(序号/类型/描述/三元组得分/逐类计数)+ 复现 + 为什么试它。
- 数据构造命令**务必记**（含 `--supported-downsample`）+ **产物 md5/行数**。⚠️ **s01 真配方已取证锁定（2026-06-14）：`--minority-oversample 3.0 --supported-downsample 0.50 --seed 42`（split-mode=random 默认、val-ratio 0.1）→ `data/train_sft.jsonl` md5 `3c5deb9b702b735267233cf85f61fe4f`、12741 行 / 9750 Supported、split.json md5 `7c17d2f2`。** build 是**确定性（seeded）**、默认逐句路径 Jun2→今**字节不变**，原始 traindev 完好 → s01 可**逐字节再生**。**之前把 ds 反推成 ~0.6 是错的**（0.6→13804=full_grid，正是 s33/s51/s52 输败那份；句子级 vs 记录级口径搞错）——这个未记+错推的参数害我们烧了 s33/s49-54 共 7 次重训复刻不出 47.80,还误判"softmin 必输/s01 不可复现"。**纪律:派生数据 build 命令+md5 进 `submissions_log.md`;反推值标 `(GUESS)` 并立即 build 校验;关键产物用带 md5 的不可变文件名,勿复用同路径。**

## 9. 待办 / 开放项（2026-06-04 大改：3 个 multi-agent 调研重定方向，详见 notes/）

> **已证伪、别再投入**（见 `notes/submissions_log.md` 总表 + `notes/breakthrough_plan.md`）：
> - **堆 union 成员死了**：规模多样性(q4b s18)、架构多样性(InternVL s19-21 / Gemma s23/s28/s29)全 ≤50.26;**模型越大越糟**(31B 单模 s27=29.28 < 26B s26=44.69)。50.26 是 naive-union 天花板。
> - **外部数据增强死了**(`notes/data_augmentation_research.md`)：SciFact/SciNLI/HaluEval/FactCC 模态+标签双错位。
> - **抑制类后处理 / 置信门控 union 死了**：testp1 PEM 召回受限(s16 砍单票 −2.97);门控 PEM-safe 版退化成 s15。
> **根因(已坐实)**：训练数据「≤1 少数类/段」伪先验(长段仅 0.46% 含≥2;s01 testp1 仅 3.8% 段开≥2)→ 系统性漏召回。

- [x] **★P0 GATE-B 离线评测台 PHASE A+B 完成**(`notes/gate_b_plan.md` / `notes/gate_b_phaseB_result.md`)：`build_dataset --split-mode component_aware --val-ratio 0.15 --out data/devbench`(501 dev,零-sha)→ `scripts/reshape_devbench.py` 产 raw/reshaped/**densematch** 3 变体(trivial 全-Sup 试金石:densematch 17.2≈testp1 18.85 → 召回-critical 对)。**PHASE B(4×H200,16 锚点,`scripts/phaseB_run.sh`+`phaseB_synth_rankcorr.sh`+`rankcorr_meta.py`)**:**densematch 最佳 Spearman +0.590**(raw −0.02 / reshaped +0.22 单调升 → shaping 方向对),但 <0.8 可信门 —— **旧 adapter 记忆泄漏封顶**(s10 二轮过拟合 dev 98.05/testp1 41.70 反相关)。**⇒ 旧 adapter 不能选型;唯一干净路 = P1 累积协议。**
- [ ] **★P1 密度匹配训练 = PHASE C 累积协议(主攻,+3~6,评测台与破局合一)**(`notes/breakthrough_plan.md`)：把真少数类句插进长段、joint 训练,让单模在密集长段 catch 多个真少数类（造 union 没有的新正确召回）。**只在 `data/devbench/train_sft.jsonl`(image-disjoint train',从不见 dev')训 → `data/devbench/dev_gold_densematch.jsonl` 无泄漏离线评 + 同时交 testp1**,累积无泄漏 (dev,testp1) 对(第一个干净点既是破局模型又校准评测台)。densematch +0.59 是泄漏下界,干净后应更高。
- [ ] **P2 便宜叠加**(P1 跑通后)：纯加法自洽采样(temp>0 多采样、只增不删,需给 infer.py 加 --temperature/--n)；OCR 数字通道(表格数字转写,UE/Contra 接地)。
- [ ] Phase-2（2026-06-11 放出）：勿过拟合 Phase-1；**保 s01(47.80)/s15(50.26) 永久保底**。

## 10. 参考（官方文件路径）

- 指南：`../NLPCC-2026-Task10-Science/GUIDELINES.md`
- 评测器：`../NLPCC-2026-Task10-Science/offline_eval/evaluate.py`（Track1 逻辑见 `evaluate_track1`）
- 基线 prompt：`../NLPCC-2026-Task10-Science/baseline_prompting/prompts/track1_{system,user}.txt`
- dev 构造参考：`../NLPCC-2026-Task10-Science/baseline_prompting/prepare_dev_eval.py`
