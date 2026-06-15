# Codabench Track-1 提交记录(命名规范 + 得分 + 复现命令)

> 官方评测:`score = (Sentence Macro-F1 + Paragraph Exact Match PEM)/2`。平台 = Codabench arena 16666(提交上限 100)。

## 命名规范(务必遵守)
```
testp1_s{NN}_{T}_{desc}.zip   (+ 同名 .jsonl)
```
- `testp1` — 阶段(Phase-1 测试集;Phase-2 用 `testp2_`)。
- **`s{NN}`** — 2 位**提交序号**,按创建顺序递增,稳定唯一 ID(上传 Codabench / 记分时用它指代)。
- **`{T}` — 类型**:`m`=单模(单个训练模型) / `pp`=后处理(单模的规则后处理) / `u`=union(并集集成)。
- **`{desc}` — 简短描述**:单模=模型+配置(`q8b-grid` / `q32b-joint` / `gemma26b` / `iv8b` …);union=成员(`q5` / `q5-gemma26b` / `2of4` …);后处理=手法(`tau020` / `vetoAll` …)。**desc 内部用 `-` 连接,`_` 只分隔三段**。
- zip 内文件**永远是 `track1_pred.jsonl`**(官方扁平 zip 规范;`make_submission_zip.py` 自动处理)。
- 新增提交:取下一个 `s{NN}` + 定类型 → 重命名 zip/jsonl → 追加下面总表一行(序号/类型/描述/score)+ 复现命令。

## 关键背景(先读)
- **testp1 trivial(全 Supported)≈ 19**(实测 s05=18.85),**不是** train-dev 的 43.2。testp1 段落长(均值 8.7、最长 31 句)、少数类密集(~84% 段含 ≥1 gold 少数类)。
- ⇒ **testp1 是召回-critical**:抑制少数类(veto/阈值/降过采样)单调掉分(见 s03–s07)。
- ⇒ 已证实**走不通**:过采样调召回(均匀/逐类都不改 OOD 逐类率)、提分辨率(s02 −3.3)、2 epoch(过拟合)。详见 `memory/testp1-result-v1.md`。

## 环境(H200,命令在服务器跑)
```bash
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache
# 校验+打包:uv run python scripts/make_submission_zip.py --sub <jsonl> --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/testp1_sNN_tag.zip
```

## 得分总表(按序号;类型 **m**=单模 / **pp**=后处理 / **u**=union)
> zip 列省略前缀 `testp1_` 和后缀 `.zip`(+ 同名 `.jsonl`)。`待测` = 已打包未上传 Codabench。

| 序号 | 类型 | 描述 | 文件(submissions/) | Score | MF1 | PEM | #少数类(UCM/UE/SO/Contra) |
|---|---|---|---|---:|---:|---:|---|
| **s01** | m | q8b 句级 grid 最优 ⭐保底 | `s01_m_q8b-grid` | **47.79897** | 52.76518 | 42.83276 | 253 (109/92/29/23) |
| s02 | m | q8b 全分辨率802816(有害−3.3) | `s02_m_q8b-fullres` | 44.46574 | 48.99973 | 39.93174 | 260 (104/108/22/26) |
| s03 | pp | min_logprob<−0.20 少数类→Sup | `s03_pp_tau020` | 43.85382 | 48.28784 | 39.41980 | 190 |
| s04 | pp | min_logprob<−0.03→Sup | `s04_pp_tau003` | 38.47759 | 42.48420 | 34.47099 | 127 |
| s05 | pp | 孤立少数类全 veto(trivial 锚) | `s05_pp_vetoAll` | 18.85468 | 22.00970 | 15.69966 | 46 |
| s06 | pp | 孤立少数类 gated veto(−0.05) | `s06_pp_vetoGated` | 39.57432 | 44.33635 | 34.81229 | 171 |
| s07 | m | q8b 降过采样1.5×(更差) | `s07_m_q8b-gentle15` | 40.90984 | 46.83674 | 34.98294 | 246 |
| s08 | m | q8b 均匀过采样4× | `s08_m_q8b-os4` | 45.10490 | 50.44872 | 39.76109 | 235 (102/80/28/25) |
| s09 | m | q8b 逐类过采样 | `s09_m_q8b-perclass` | 43.83218 | 49.95104 | 37.71331 | 268 (104/112/33/19) |
| s10 | m | q8b 逐类过采样2ep(过拟合最差) | `s10_m_q8b-perclassEp2` | 41.69745 | 46.53484 | 36.86007 | 193 (80/74/17/22) |
| **s11** | u | q8b×3(s01+08+09) | `s11_u_q3` | **48.21778** | 54.79733 | 41.63823 | 339 (127/142/43/27) |
| s12 | m | q8b 段落联合 plain-CE | `s12_m_q8b-joint` | 43.33171 | 48.26752 | 38.39590 | 209 (90/66/31/22) |
| s13 | m | q32b 段落联合 plain-CE | `s13_m_q32b-joint` | 44.43184 | 49.44389 | 39.41980 | 183 (97/40/19/27) |
| s14 | u | 4成员(s01+08+09+s13) | `s14_u_q4` | 49.98915 | 56.80424 | 43.17406 | 365 (137/148/48/32) |
| **s15** | u | **5 Qwen 全 ⭐当前最优** | `s15_u_q5` | **50.25660** | 57.16849 | 43.34471 | 382 (139/155/52/36) |
| s16 | u | ≥2票共识(4成员,差) | `s16_u_2of4` | 47.28613 | 52.59274 | 41.97952 | 253 (110/87/32/24) |
| s17 | m | q4b 单模(4B,证伪) | `s17_m_q4b` | 待测 | — | — | 220 (91/78/27/24) |
| s18 | u | 5Q+q4b(证伪 −0.55) | `s18_u_q5-q4b` | 49.71340 | 57.27662 | 42.15017 | 404 (139/164/58/43) |
| s19 | u | 5Q+iv2b | `s19_u_q5-iv2b` | 48.48465 | 56.35496 | 40.61433 | 448 (146/201/58/43) |
| s20 | u | 5Q+iv8b | `s20_u_q5-iv8b` | 49.35701 | 56.73450 | 41.97952 | 415 (139/171/58/47) |
| s21 | u | 5Q+iv2b+iv8b | `s21_u_q5-iv2b8b` | 47.77223 | 55.95401 | 39.59044 | 472 (146/212/61/53) |
| s22 | u | consensus8(8成员≥2票) | `s22_u_consensus8` | 待测 | — | — | 332 (121/129/48/34) |
| s23 | u | 5Q+gemma26b | `s23_u_q5-gemma26b` | 49.82598 | 57.67243 | 41.97952 | 498 (152/237/65/44) |
| s24 | m | iv2b 单模 InternVL3-2B | `s24_m_iv2b` | 待测 | — | — | 241 (71/134/18/18) |
| s25 | m | iv8b 单模 InternVL3-8B@2tiles | `s25_m_iv8b` | 40.04057 | 45.26884 | 34.81229 | 210 (79/74/31/26) |
| s26 | m | gemma26b 单模 Gemma4-26B(2卡zero3) | `s26_m_gemma26b` | 44.69206 | 51.84146 | 37.54266 | 370 (105/191/46/28) |
| s27 | m | gemma31b 单模 Gemma4-31B(2卡zero3,极保守) | `s27_m_gemma31b` | 29.27701 | 33.98063 | 24.57338 | 87 (45/23/13/6) |
| s28 | u | 5Q+gemma31b | `s28_u_q5-gemma31b` | 待测 | — | — | 397 (145/161/54/37) |
| s29 | u | 5Q+gemma26b+gemma31b | `s29_u_q5-gemma26b31b` | 待测 | — | — | 507 (153/243/66/45) |
| s30 | u | PHASE C 干净并集(train':B0联合+A1句级+E1自洽) | `s30_u_cleanunion` | **43.85** | 48.61 | 39.08 | 269 (108/111/28/22) |

> **s30 = PHASE C 第一个「干净」候选,实测 testp1 = 43.85 / 48.61 / 39.08 —— 低于 s01(47.80),远低于 s15(50.26)。预估(~47–50)错了,负结果。**
> **★第一个干净 (densematch, testp1) 校准点 = (83.90, 43.85)。** 这一发就证伪了 PHASE C 评测台对这类候选的可信度:
> - **densematch 把 s30 排到 83.90(高于所有单模),实测 testp1 只 43.85**;densematch 上「联合 union 升 PEM(87.38>B0 86.41)」,testp1 上 **PEM 反崩到 39.08 < s01 42.83** —— 并集在 testp1 加的是**假阳性**,与 densematch 判断相反。PHASE B 早警告的「rho 仅 +0.59、joint-vs-句级不迁移」**重度兑现**。
> - **根因(echo s18/q4b 教训)**:s30 只有 **2 个真正不同的模型**(A1 与 E1 同基座=相关;B0 联合)+ **train' 仅 90% 数据** + **联合-heavy** → B0 联合(testp1 历史 ~43)拖着整个并集到联合档(43.85≈s12/s13 的 43-44)。对比 s15 = **5 个真异构 full-traindev 成员**。**densematch 高估了联合成员 + 把相关成员当多样。**
> - **结论**:s30 **不是**新最优;**s15(50.26)仍是天花板,s01(47.80)仍是单模最优**。**PHASE C 评测台(densematch)尚不可信**,需更多校准点(单交一个干净的纯句级 A1 会很有信息量——它没被单独测过)。复现:`scripts/phaseC_package_testp1.sh`。详见 `notes/phaseC/CAMPAIGN_SUMMARY.md`。

| s31 | m | PHASE C 干净 softmin 单模(train' 句级,= A0) | `s31_m_clean-softmin` | **34.21** | 39.57 | 28.84 | 164 (57/85/9/13) |
| s32 | m | PHASE C 干净 plain-CE 单模(train' 句级,= A1) | `s32_m_clean-plainCE` | **38.68** | 43.57 | 33.79 | 166 (78/56/19/13) |

> **s31/s32 实测 → 大翻案:densematch 的「干净同条件」排序在 testp1 上成立,我之前的悲观结论是错的。** 三个干净 PHASE C 模型在两台子上**同序**:
> | 模型 | densematch | testp1 |
> |---|--:|--:|
> | s31 softmin (A0) | 78.97 | 34.21 |
> | s32 plain-CE (A1) | 81.13 | 38.68 |
> | s30 union (U3) | 83.90 | 43.85 |
> 1. **plain-CE > softmin testp1 坐实**:38.68 > 34.21 = **+4.47**(比 densematch 的 +2.16 还大)。**「softmin 死」在 testp1 上确认**(无过采样档)。
> 2. **union 有用,我之前「B0 联合拖累 union」的判断错了**:s30 union 43.85 > s32 单模 38.68 = **+5.17**,并集正常加召回。
> 3. **densematch 没那么废**:它把干净同条件模型排对了(softmin<plainCE<union 两台同序),只是①压缩了幅度②给不了绝对分③**不能跨 regime 外推**。s30=43.85 的「失手」根因 = **train'(90%数据)+ 无过采样的让步(~9-13 分)**,不是台子/union 的错 —— s01(47.80)是 full-traindev + 过采样。
> **⇒ 行动路线明确**:PHASE C 配方(plain-CE > softmin、union、自洽)已被 testp1 验证;**要破 s15(50.26)就把它搬到 full-traindev**:s15 的句级成员全是 softmin(grid/os4/perclass),**把它们换成 plain-CE 重训 full-traindev 再 union**,有望 > 50.26。复现:`scripts/phaseC_package_singles.sh`。

| s33 | m | full-traindev plain-CE 单模 grid(os3.0/ds0.6) | `s33_m_grid-plainCE` | **42.87** | 48.72 | 37.03 | 245 (92/108/21/24) |
| s34 | m | full-traindev plain-CE 单模 os4(os4.0/ds0.66) | `s34_m_os4-plainCE` | 待交 | — | — | 219 (88/89/23/19) |
| s35 | m | full-traindev plain-CE 单模 perclass(逐类) | `s35_m_perclass-plainCE` | 待交 | — | — | 283 (115/116/32/20) |
| s36 | m | full-traindev plain-CE 单模 gemma26b | `s36_m_gemma26b-plainCE` | **21.40** | 24.88 | 17.92 | 46 (15/21/7/3) |
| **s37** | u | 5-Qwen plain-CE 并集(grid+os4+perclass plainCE + joint8b + joint32b) | `s37_u_q5-plainCE` | **48.54** | 55.62 | 41.47 | 390 (132/169/50/39) |
| s38 | u | s37 + gemma26b plain-CE | `s38_u_q5-plainCE-gemma` | **48.58** | 55.69 | 41.47 | 405 (135/176/53/41) |

> **🏁 plain-CE 批次终局(s33-s38 全部出分,无一破 s15):** 最佳 = s38/s37 ≈ **48.5 < s15 50.26**;最差 = s36 gemma plain-CE **21.40**(只 46 少数类,塌成近-trivial;vs gemma softmin s26=44.69,**−23**!gemma 上 plain-CE 灾难性欠拟合,与 Qwen 上的「过度开火」相反 —— loss×模型×数据交互不可外推)。**s38(+gemma)48.58 ≈ s37 48.54:gemma 加进 union 净增益 ~0**(印证 s23 的 gemma-没用)。**结论钉死:full-traindev+过采样下 softmin>plain-CE;gemma 不进 union;s01(47.80)/s15(50.26)是天花板,plain-CE 路彻底关闭。**

> **🔴 大翻转(2026-06-05):full-traindev + 过采样这个真实配方下,softmin > plain-CE,plain-CE 重训全军覆没。**
> - **s37(plain-CE 并集)48.54 < s15(softmin 并集)50.26,−1.72**;**s33(grid plain-CE 单模)42.87 << s01(grid softmin 单模)47.80,−4.93**。单模、并集两级都确认。
> - **机制**:s37 少数类 390 > s15 382,但 **PEM 41.47 < 43.34**;s33 PEM 37.03 << s01 42.83(−5.8)。**plain-CE + 过采样 = 过度开火少数类(FP 打碎段落)→ PEM 崩**。**softmin 的全部价值 = 控制过采样诱发的 FP**(PEM-bottleneck loss 恰在「过采样导致过度开火」时才发挥作用)。
> - **⇒ 之前「softmin 死、plain-CE 更好」(s31/s32)是「无过采样」regime 的伪信号**;一旦上真实的过采样配方,softmin 反超 ~5 分。**softmin 没死;s01(47.80)/s15(50.26)仍是天花板,plain-CE 这条路否决。** s36/s38(plain-CE Gemma)训完补全(大概率同样偏低)。复现:`scripts/retrain_member.sh` + `train_gemma4.sh` + `ensemble_union.py`。

## P1.5 CoT-SFT(环节② E0/E1;基座=s01 配方,见 notes/cot_sft_plan.md)

| s39 | m | **E1 = s01 配方 + CoT 教材**(softmin grid os3.0/ds0.6 401408 1ep 2卡 + `<analysis>`;⚠️**train'(devbench 85% image-disjoint)非full-traindev**) | `s39_m_cotE1` | **38.54** | 43.28 | 33.79 | **298 (122/127/11/38)** |
| s40 | m | **E0 = E1 对照,label-only**(同 split/过采样/配方,唯一变量=无CoT) | `s40_m_cotE0` | **43.85** | 49.31 | 38.40 | **233 (102/77/41/13)** |

> **🔴🔴 G2 闸门失败:CoT 有害(2026-06-11 干净对照实测)。** E1(CoT) vs E0(label-only) 唯一变量=+`<analysis>`,同 train'/split/过采样/配方,离线两台同档可比:
> | 评测台 | E1(CoT) | E0(label-only) | **E1−E0** |
> |---|--:|--:|--:|
> | densematch score | 0.735 | **0.840** | **−0.105** |
> | densematch MF1/PEM | 0.743/0.728 | 0.825/0.854 | −0.08/−0.13 |
> | raw-dev score | 0.752 | **0.873** | **−0.122** |
> | raw-dev MF1/PEM | 0.687/0.816 | 0.819/0.928 | −0.13/−0.11 |
> - **🔴 testp1 实分三重确认(2026-06-11):s40(E0)=43.85 > s39(E1 CoT)=38.54,E0−E1=+5.31(MF1 +6.0/PEM +4.6)。** 与离线 densematch(+0.105)、raw-dev(+0.122)完全同向 ⇒ **CoT 有害结论锁定,三个独立信号一致。**
> - **E0 两台两指标全面碾压 E1 ~0.10-0.12,远超噪声。CoT 净效应为负。** testp1 开火:E1 298(122/127/11/38) vs E0 233(102/77/41/13)——E1 多开 65 个少数类但分数低得多 ⇒ **多开的是净假阳性(MF1+PEM 双输)**;且 E1 把 SO 压到 11(E0 41,CoT 反而漏 SO)。
> - **数据档成本量化**:s40(E0,train' label-only)=43.85 vs s01(full-traindev label-only)=47.80,train'(85% image-disjoint)仅 −3.95(过采样缓解了,比预估的 −9~13 小)。即:full-traindev+CoT 乐观上限≈47.8−5.3≈42.5 < s01,**CoT 即便搬 full-traindev 也大概率破不了 s01**。
> - **机制(印证用户从一开始的怀疑)**:E1 loss~0.7(analysis 占)vs E0 loss~0.005(label 近记忆)。固定 1 epoch 预算下,CoT 目标**稀释了 label 拟合**;推理时 analysis 100%出现在少数类预测上(看似承重)却导出**低精度的少数类开火**——「不忠实 CoT」兑现:推理在场但把 label 带偏。
> - **⇒ 路线判定**:**当前 s01 配方下 CoT-SFT 这条路否决**(densematch 同档排序在 s31/s32 已证可信)。GRPO **不在 E1 上跑**(放大已劣化的模型);roadmap 退化方案=E0 上 label-only GRPO(已非 CoT 路,价值低)。**s01(47.80)/s15(50.26) 仍是天花板,永久保底。** 待用户拍板:停 CoT 线 / 试变体(更多 epoch?但 s10 证 2ep 过拟合 / 改 analysis 格式 / E0-GRPO) / 直接回 union 冲榜。E0(s40) testp1 实分待传 Codabench 二次确认(densematch 强预测 E0 > E1 的 38.54)。
> - 推理成本旁证:E1 testp1 推理~73min(analysis 解码) vs E0~20min(label-only),CoT 推理慢~5-6×。

> **s39 = P1.5 第一个 CoT 模型,2026-06-11 出。** 教材 = gpt-5.5 三阶段蒸馏(units.jsonl 20794,少数类覆盖~99%,Stage C 审核 PASS 90.6%);E1 用 `build_dataset --cot` 把 `<analysis>{rationale}</analysis>\n{"label"}` 作训练目标,softmin **只作用末行 label span**(`<analysis>` 走 CE floor;test_softmin T7/T8 验证),其余严格 = s01(归因唯一变量=+CoT)。
> - **格式健康**:testp1 parse_fallback = **1/5096(0.02%)**——CoT 不破坏末行 JSON 解析。
> - **少数类 298 vs s01 单模 253(+45)**:UE 127(+35)/Contra 38(+15)/UCM 122(+13) 都↑,但 **SO 11 << s01 29**(CoT 让范围外推更保守)。落在健康单模区(s01 253/s09 268),非抑制塌陷区。
> - **待**:testp1 实分(用户传 Codabench)+ densematch G2 分(dev 推理中)。判据:E1 testp1 > s01 47.80 且少数类召回不降 = CoT 增益成立 → E2/GRPO 起步用 E1;≈s01 = 教材未起效查 analysis 质量;< s01 = 排查 softmin-span。
> - **复现**:教材 `scripts/cot_distill.py --stage blind|gold|verify`;数据 `build_dataset --cot data/cot/units.jsonl --split-mode component_aware --val-ratio 0.15 --minority-oversample 3.0 --supported-downsample 0.6`;训练 `GPUS=2,3 bash scripts/train_e1.sh`(含 cuDNN SDPA 禁用修复 H200 mha_graph 崩)。adapter=`outputs/cotE1_s01cot/v1-*/checkpoint-1980`。
>
> **🔴 s39 testp1 = 38.54,实测分析(2026-06-11):** 比 s01(47.80)低 9.3,但**这不是干净的 CoT 对照**——**E1 训在 train'(devbench 85% image-disjoint),s01 训在 full-traindev**。数据档差独立值 ~9-13 分(对照: s31 train'-softmin-无过采样=34.21 vs s01 full+过采样=47.80,差 13.6)。s39(train'+过采样+CoT)=38.54 比 s31 +4.3,但 ≈ s32(train' plain-CE 无过采样)38.68。**⇒ 38.54 落在 train' 数据档,CoT 净效应被数据档淹没,testp1 无法归因。** 干净归因唯一办法 = **训 E0(train'+过采样+label-only,跳过了的对照)**,比 E1−E0(testp1 + densematch 同档可比)。**G2 闸门尚未判定;GRPO 暂不启动**(无 E0 = 不知 CoT 增益,RL 只会放大未验证的东西)。⚠️ SO 仅 11(<s01 29)是具体担忧:CoT 让范围外推过保守。下一步:E0 对照 + analysis 因果消融(去 analysis 重推,看 label 是否变=因果承重否)。

## P2 A1 自洽 + softmin/plainCE 干净 λ-ablation(2026-06-14;见 notes/a1_a3_a4_execution_plan.md)

| 序号 | 类型 | 描述 | 文件(submissions/prod/) | Score | MF1 | PEM | #少数类(UCM/UE/SO/Contra) |
|---|---|---|---|--:|--:|--:|---|
| s41 | m | **A1 自洽**:s01 + 温度自洽(T1.0 K=8 add-only union,含 greedy 成员保纯加法) | `testp1_s41_m_a1sc-s01` | **47.75838** | — | — | 318 (119/137/37/25) — vs s01 greedy 253,自洽 +65 firing |

> **🔴 s41 = 47.76 ≈ s01 47.80(Δ −0.04,持平)→ A1 自洽 NULL on testp1。** 离线 densematch 的 +2.10 lift(softmin A0 代理)**未迁移**:densematch in-domain 低 FP 压力 → 高估 added-firing 杠杆(s30 同病,ρ=0.59 不可信)。testp1(OOD/PEM-precision-limited)上,65 个新 firing **净 break-even**(加对的被加错碎 PEM 抵消)。**A1 不破 50.26,也没伤(持平);s15(50.26)仍天花板。** ⇒ s42(A1×s15)同机制大概率也 NULL,不值得交;A1 作为 recall-adder 在 testp1 死。**唯一离线信号 densematch 再次对 added-firing 杠杆 over-promise** —— 教训三次锁定(s30/s16/s41)。
| s49 | m | **λ-ablation plainCE**:A1(devbench train', λ=0)greedy testp1 | `testp1_s49_m_ablate-plainCE-trainp` | **38.68254** | — | — | 166 (79/57/17/13) |
| s50 | m | **λ-ablation softmin**:A0(devbench train', λ=0.5)greedy testp1 | `testp1_s50_m_ablate-softmin-trainp` | **34.59594** | — | — | 166 (59/84/10/13) |
| s51 | m | **full-traindev λ-ablation softmin**:softmin @ s33 同数据(`data/full_grid`)/同配置/同 4214 步,唯一改 λ=0.5;**vs s33(plainCE 42.87)定 softmin 在 full-traindev 有没有用** | `testp1_s51_m_ablate-softmin-full` | **39.14218** | — | — | 242 (82/122/20/18) |

> **🟢 softmin 之谜定论 + 破局线索(2026-06-14):s51(softmin-full @ s33 config)=39.14 < s33(plainCE 同 config)=42.87,plainCE +3.73。** 三台全一致(densematch +2.16 / train' +4.09 / full-traindev +3.73)→ **softmin 处处 < plainCE,确定无用**;"softmin>plainCE" 的旧信念(混淆 s01-vs-s33)彻底证伪。
> **⇒ s01 的 47.80 不是 softmin 给的,是配置给的**:config-A+softmin(s01=47.80) >> config-B+softmin(s51=39.14),**差 +8.66!**(config-A = max_length 4096 / 2卡 DDP global-batch-2 / s01 build;config-B = max_length 10240 / 1卡 / full_grid build)。**配置值 ~+8.66,是迄今最大单一杠杆。**
> **★从没训过的 config-A + plainCE**:期望 ≈ 47.80(s01 config-A softmin)+ plainCE 优势(~+2~4) ≈ **49.8~51.8,有望破 s15(50.26)** 且作单模/union 基座俱佳。**= 当前最高价值实验(下一步)。** 警:s01 build 当年 downsample 未精记(§8),复刻用 os3.0/ds0.6 best-guess。

| s52 | m | **config-A + plainCE**(maxlen4096/2卡;**数据用 `data/full_grid`(13804)≠ s01 的 `train_sft`**;λ=0)| `testp1_s52_m_cfgA-plainCE` | **41.55891** | — | — | 272 (92/137/23/20) |

> **🔴 s52=41.56 外推(~51.5)惨败 → 根因 = 数据 build 用错,不是 config。** s52 训练健康(2107 步 loss 0.06)。证据:s52(full_grid+maxlen4096+2卡+plainCE)=41.56 ≈ s33(full_grid+maxlen10240+plainCE)=42.87 → **maxlen/卡数对 full_grid 几乎无影响,全落 39-43 档**。所有 full_grid 模型(s33/s51/s52)= 39-43;**唯一逃出的是 s01=47.80,它用的是 `data/train_sft.jsonl`(原始 ~12741,78%Sup/22%少数;≠ full_grid 13804)。**
> ⇒ **s01 的 +8.66 优势 = 数据 build(downsample/配比),不是 max_length/2卡(我之前的误判)。** s01 原始数据文件已被覆盖(现 train_sft.jsonl=Jun3 重建 12802,配比与 s01 近似但非同一文件)。**真正的 config-A+plainCE 从没正确训过**(s52 用错了数据)。修法 = 用 `data/train_sft.jsonl` 重训 plainCE(s53)。
> ⚠️ 警示沉淀:本数据集**跨 config 外推极不可靠**(softmin/A1/config 三次外推全错);只信 testp1 实分。s01(47.80)因原始数据丢失**不可复现**(adapter 已存,作提交安全)。

| s53 | m | **修正版 config-A+plainCE**:plainCE @ `data/train_sft.jsonl`(s01 配方近似,12802,2卡 maxlen4096)= s52 用错数据的修正重跑 | `testp1_s53_m_cfgA-plainCE-tsft` | **44.08014** | — | — | 260 (110/104/30/16) — profile≈s01 253(109/92/29/23) |

> **🔴 s53=44.08(plainCE@train_sft)< s01=47.80(softmin@train_sft-原始),低 3.72;但 > s52=41.56(plainCE@full_grid)+2.5。** ⇒ 数据 build 确实影响(train_sft 比 full_grid +2.5),但 **train_sft(44 档)仍达不到 s01(47.80 档)** → 当前 `data/train_sft.jsonl`(Jun3 重建)**≠ s01 原始(Jun2)数据**,s01 的数据真丢了。待 s54(softmin@train_sft 复现对照)定论:s54≈47.80→train_sft=s01 数据(则 plainCE<softmin 反转,需查);s54≈44→train_sft≠s01 数据(s01 不可复现,plainCE≥softmin 维持)。后者概率高(plainCE>softmin 已三台)。
| s54 | m | **复现对照**:softmin @ `data/train_sft.jsonl`(= s01 配方+loss,同 s53 配置只改 λ=0.5);≈47.80 则证 train_sft=s01 配方 | `testp1_s54_m_cfgA-softmin-tsft` | **44.74991** | — | — | 244 (114/82/33/15) |

> **🔴🔴 s01 复现失败 + 实证定论(2026-06-14):s54(softmin@train_sft,与 s01 完全同配方/同 loss)= 44.75,比 s01=47.80 低 3.05。** ⇒ **s01 的 47.80 来自 Jun2 那份已被覆盖的原始数据,不是 config/loss。** 当前 `data/train_sft.jsonl`(Jun3 重建)≠ s01 数据;同配方同 loss 也只到 44.75。**s01 不可从现有数据复现(adapter 已存,作提交安全)。**
> **🟡 softmin vs plainCE 实为数据依赖,之前"softmin 必输"修正:** train_sft 档 s54 softmin 44.75 ≥ s53 plainCE 44.08(softmin +0.67,~打平);full_grid/devbench/densematch 档 plainCE 赢 +2.2~4.1。⇒ **softmin 对 s01 不是错(那档打平);plainCE 只在别的 build 上明显占优。** 净:plainCE ≥ softmin 但 margin 数据依赖、有时打平。
> **数据档总览(testp1):** s01 train_sft-Jun2=47.80 >> {train_sft-Jun3: s53/s54 ≈44} > {full_grid: s33/s51/s52 ≈39-43} > {devbench train': s49/s50 ≈35-39}。**数据 build 是最大杠杆(跨档差 ~8)。**

> **🟢🟢 取证定论(2026-06-14,9-agent workflow + 本地独立验证):s01 完全可恢复,根因 = downsample 参数记错。**
> - **真配方 = `--minority-oversample 3.0 --supported-downsample 0.50 --seed 42`(random split,val 0.1)**;本地 build 验证:**ds=0.50 → md5 `3c5deb9b702b735267233cf85f61fe4f`、12741 行 / 9750 Supported**(精确命中 s01 指纹);重跑同 md5(**确定性 seeded**);ds=0.60 → 13804 = **full_grid**(s33/s51/s52 输败那份)。
> - **之前 ds 反推成 0.6 是错的**(句子级 9750/14775≈0.66 vs 代码记录级下采样)→ 所有复刻喂错数据,从没训过 ds=0.50。**"config-A 的 +8.66"其实就是 ds=0.50 vs 0.60。**
> - H3(unseeded 运气)**证伪**(代码 Random(seed)/ds_rng=Random(seed+1),双跑字节同);H4(逻辑变)**证伪**(Jun2 6d17ad0 与 HEAD builder 默认路径字节等价);数据文件确丢(gitignore+从未提交+hf_data 不同步 random-split)但**可再生**。
> - ⇒ **s55 = ds0.50 + s01 配方重训 = 复现 47.80 + 拿回干净可复现基座**(s01 zip/adapter 一直在,提交安全)。**新杠杆:ds 扫描(0.45/0.50/0.55)× plainCE**,ds=0.50 vs 0.60 = ~8 分,是迄今最大可控杠杆。复刻代价沉淀:一个未记+错推参数 → 7 次重训 + 误判"softmin死/s01不可复现"。

> **🔴 干净 λ-ablation 判定(2026-06-14 testp1 实测):plain-CE 38.68 > softmin 34.60,Δ=+4.09。** 加上 densematch 同对照(plainCE 81.13 > softmin 78.97,+2.16),**两台一致:plain-CE 在 in-domain 和 OOD 上都胜 softmin。** 且 s49/s50 几乎复现早先 s32(plainCE train'≈38.68)/s31(softmin train'≈34.2)——**双重确认**。
> ⇒ **唯一支持"softmin>plainCE"的证据 = 混淆的 s01(47.80) vs s33(42.87)**(差 max_length 4096/10240、数据集、batch 2/1)。**所有干净实验都反过来。** s01 的 47.80 究竟来自 softmin 还是它的配置/数据,需一个 **full-traindev 干净 λ-ablation**(softmin sibling of s33,同配置只改 λ)才能定;若 softmin-full ≈ s33(42.87) → softmin 无用、s01 优势是配置/数据假象;若 ≫ → softmin 真有 full-data 交互。**这是当前信息量最高的实验,且可能换掉 A1 的基座。**

> **A1 离线 gate(densematch 206,leakage-free,2026-06-14):** 自洽(温度采样 + add-only union)在两条 lineage 都做出干净 lift:
> | adapter(train') | greedy | T0.7 K8 | **T1.0 K8** |
> |---|--:|--:|--:|
> | A1 plain-CE | 81.13 | 82.74 (+1.61) | **83.70 (+2.57 / PEM +3.40)** |
> | A0 softmin(=s01 配方代理) | 78.97 | 80.89 (+1.92) | **81.07 (+2.10)** |
> PEM 单调↑证明加的是真阳性。**softmin 比 plain-CE 更尖锐但 lift 不输** → 自洽对生产的 s01(softmin)应迁移 → s41 用 T1.0 K8 套到 s01。infer.py 已加 `--n`(引擎内多采样)/`--records-file`(绕 split.json 干净评子集),已上 main 92f1e1a/f98922d。
>
> **🔬 softmin vs plain-CE 干净 ablation(回应"s01 vs s33 是否只差 loss"):** args.json 实测 **s01-vs-s33 不是干净对照**——除 loss 外还差 max_length(4096 vs 10240)、数据集文件、有效 batch(2 卡 vs 1 卡;step 2000 vs 4214)。**唯一干净 ablation = A0 vs A1**(args.json 只差 λ,同数据/同配置):densematch 上 **plain-CE 81.13 > softmin 78.97(+2.16)**。⇒ **没有干净实验证明 softmin 在 testp1 胜 plain-CE**;唯一干净对照在 in-domain densematch 上反而 plain-CE 赢。s49/s50 = 把这对干净 adapter 交 testp1,看 OOD 上是否翻盘(densematch 低估 FP 压力,softmin 价值假设在 testp1 长段才显现)。
> 全部 ADD-mode;s01=47.80 / s15=50.26 永久保底不动。s42-s48 为 execution_plan 预留(A1×s15 / A3 / A4 GRPO),尚未跑。

## P3 s01 恢复 + A1 multi-target(2026-06-14 夜;ds=0.50 正确数据)

| 序号 | 类型 | 描述 | 文件(submissions/prod/) | Score | MF1 | PEM | #少数类(UCM/UE/SO/Contra) |
|---|---|---|---|--:|--:|--:|---|
| s55 | m | **复现 s01**:ds=0.50(正确数据,12741/9750)+ s01 配方(softmin 2卡 maxlen4096)| `testp1_s55_m_recover-s01-ds050` | **45.48379** | — | — | 241 (106/79/26/30) |
| s56 | m | **A1 multi-target**:ds=0.50 + `--multi-target`(多标签每 gold 各一副本,修 pick_rarest 路由)+ s01 配方 | `testp1_s56_m_multitarget-ds050` | **46.11494** | — | — | 344 (108/130/78/28) — vs s55 SO 26→78(3×)/UE 79→130 |
| s57 | (成员) | ds=0.50 **plain-CE**(SOFTMIN_LAMBDA=0,纯 CE 解相关成员)+ s01 配方 | `outputs/prod/s57_ds050_pce_member` | 未单提 | — | — | 238 |
| s58 | (成员) | ds=0.50 **multi-target plain-CE**(mt+纯CE,SIGHUP 后 setsid 重跑成功)| `outputs/prod/s58_ds050mt_pce_member` | 未单提 | — | — | — |
| s59 | (成员) | ds=0.50 **softmin seed7**(不同切分解相关,UE 偏多)| `outputs/prod/s59_ds050_sm_s7_member` | 未单提 | — | — | 271 (—/132/—/—) UE 重 |
| s60 | u | **ds0.50 舰队 union mv1**(6 成员 s01/s55-s59 OR 单票,召回)| `testp1_s60_u_ds050fleet-mv1` | **46.24557** | — | — | 486 (127/229/78/52) |
| s61 | u | ds0.50 舰队 union **mv2**(≥2 共识,精度)| `testp1_s61_u_ds050fleet-mv2` | **48.72485** | — | — | 350 (116/148/53/33) |
| s62 | u | ds0.50 舰队 union **ContraGate**(默认 mv1,Contra:2)| `testp1_s62_u_ds050fleet-contraGate` | **46.94652** | — | — | 468 (127/230/78/33) |
| s63 | u | ds0.50 舰队 union **SO+ContraGate**(SO:2,Contra:2)| `testp1_s63_u_ds050fleet-soContraGate` | **46.97287** | — | — | 443 (127/230/53/33) |
| **s64** | u | **★augmented 10 成员 mv2**(s15 老5:s01/s08/s09/s12/s13 + 新5:s55-s59,≥2 共识)| `testp1_s64_u_aug10-mv2` | **49.33612** | — | — | 377 (124/158/61/34) ≈s15 预算 |
| s65 | u | augmented 10 成员 **mv3**(≥3 共识,保守)| `testp1_s65_u_aug10-mv3` | 待 Codabench | — | — | 316 (116/123/47/30) |
| s66 | u | augmented 10 成员 **mv2 + UCM 放宽**(UCM:1,余 mv2)| `testp1_s66_u_aug10-ucmLib` | 待 Codabench | — | — | 396 (143/158/61/34) |
| s67 | u | **非对称 union**:s15 为底(全开火保留)+ 新5 ≥2 共识在 s15 漏判处加 | `testp1_s67_u_s15base-K2` | **49.48344** | — | — | 421 (+39: UCM3/UE22/SO11/Contra3) |
| s68 | u | 非对称 union s15 底 + 新5 **≥3** 共识加 | `testp1_s68_u_s15base-K3` | 不传(同源 FP) | — | — | 392 (+10) |
| s69 | u | 非对称 union s15 底 + K2(UE 收紧到 3)| `testp1_s69_u_s15base-K2ueTight` | 不传(同源 FP) | — | — | 404 (+22) |
| s70 | m | 密度匹配 softmin(全 ds0.50 + 68 注入;config-A 2卡)| `testp1_s70_m_density-softmin` | **44.67282** | — | — | 361 (UCM116/UE190/SO24/Contra31);≥2段 8.7% vs s01 3.8% |
| s71 | m | 密度匹配 plainCE(同 base+注入;1卡;解相关)| `testp1_s71_m_density-plaince` | 训练中 | — | — | — |
| s72 | u | s15 5成员 + s70 全部(OR mv1)| `testp1_s72_u_density-aug6` | 不传(+71含UE,s67险) | — | — | 453 (143/213/51/46) |
| s73 | u | s15终稿底 + s70 全部开火(asym K1)| `testp1_s73_u_s15base-s70dens` | 不传(≈s72) | — | — | 453 (+71) |
| **s74** | u | **★s15终稿底 + s70 仅 UCM/Contra(外科,排除FP-prone UE/SO)** | `testp1_s74_u_s15base-s70ucmContra` | **★50.45480** | — | — | 396 (+14: UCM5/Contra9) |

> **🟢🔴 密度匹配 testp1 实测(2026-06-15)= 单模没兑现,但外科 union 首破 s15 顶:**
> - **s70 密度单模 = 44.67 < s01 47.80 < s55(无密度)45.48。** 机制对(testp1 ≥2少数类段 8.7% vs s01 3.8%,密度确实让单模原生开多了),但 testp1 OOD 上多开的火(尤其 UE 190,UE 非密度目标类)大多 FP → 拖垮单模 PEM。**densematch 门控第 4 次过度乐观**(+2.68 没传到 testp1;前车 s30/s16/s41)。「密度→强单模 48-50」未成。
> - **★s74 = 50.4548 = 新最佳,首破 s15(50.26),+0.19。** = s15 终稿 + **只加 s70 在 UCM/Contra(密度强化类)的开火**(+14: UCM5/Contra9),排除 s70 的 UE/SO(FP-prone)。说明 s70 的 UCM/Contra 开火里确有 s15 漏的正确召回。**部分推翻 s67「加firing必亏」**:加对类(密度强化的UCM/Contra)有用,加全部(s72/s73 +71含UE)才亏。
> - s72/s73(s15+s70全部,453,+71含UE56)不传——s67 模式大概率 < s15。**新最佳 = s74 50.45。** 下一步:s71(2nd密度)好了 → s76 = s15 + (s70+s71) UCM/Contra,冲过 50.45。s01(47.80)/s15(50.26)永久保底。

> **🏁 密度增强 union 完整网格(2026-06-15)= 重组法天花板 = s75 50.61(破 s15 +0.35):**
> | 序 | 配方 | 加firing | testp1 |
> |---|---|---|---|
> | s74 | s15 + s70 UCM/Contra | +14 | 50.45 |
> | **s75** | **s15 + (s70+s56) UCM/Contra** | **+17** | **★50.61 (峰值)** |
> | s77 | s15 + 6源(+舰队)UCM/Contra | +38 | 49.59(舰队Contra=FP) |
> | s78 | s15 + (s70+s56) UCM/Contra+**SO** | +43 | 49.90(SO=FP) |
> | s80 | s75 + 11个**高置信UE**(≥−0.05) | +28 | 50.22(高置信UE也FP) |
> | s79 | s75 + 30个UE(≥−0.2) | +47 | 不传(必更差) |
> - **逐类定论:UCM/Contra 是密度唯一能加对的类(s70/s56 两源 +17 最优,加舰队/3+源就FP);UE 加任何量+任何置信门控都FP(testp1 OOD 模型自信地错);SO 无高置信子集全FP;砍=召回受限亏(s16/21冲突项)。** ⇒ **重组法封顶 s75=50.61。** 到 51.85(+1.24)需新信号(OCR数字接地 / 更强单模),deadline 内不可行。**P1 最终 = s75 50.61。** 工具:`scripts/asym_union.py`(外科加)+ `scripts/conf_gated_union.py`(置信门控,证 UE 关死)。s01/s15 永久保底。

> **🔴 s82 = OCR 数字接地(推理时,轨道A)= 负结果(2026-06-15):** GPT-5.5 当读图模型转写全 654 testp1 证据图(质量好、0失败,`scripts/gpt5_transcribe.py`,key 在服务器 `.env`/gitignore)→ 注入 caption → Qwen s01 重推增强证据(不重训)。结果:OCR 只改 s01 **1.1%(58/5096)** 预测,主要**多开 UE(+35,FP-prone)**、UCM/Contra 仅 +6;外科加进 s15 净 **+1**(1 Contra)→ **s82 ≈ s15 50.26 < s75 50.61,不传。** 根因:**s01 没训过转写 = OOD,不会用 augmented 证据**,只更爱过开火 UE。⇒ **推理时 OCR 死;真 OCR 杠杆 = 增强证据重训(轨道B)= P2 主攻**(转写全 traindev + 重训,需 `build_dataset --ocr-augment`)。GPT-5.5 当分类器也死(densematch 25,过开火;Contra 精度实测 0.29,非记忆中 82%)。**P1 最终仍 = s75 50.61。**

> **🟢 PHASE C 密度匹配门控 PASS(2026-06-15)= 项目首个过门的「造新召回」杠杆。** 在 devbench(image-disjoint,干净)上,matched control vs density(同 ds0.50 plainCE config-A 1卡,**仅差 47 个注入密集记录**):
> | | densematch score | MacroF1 | PEM | sent-acc |
> |---|---|---|---|---|
> | control(无注入) | 0.8251 | 0.8103 | 0.8398 | 0.9684 |
> | **density(+47 注入)** | **0.8519** | **0.8397** | **0.8641** | **0.9733** |
> | Δ | **+2.68** | **+2.94** | **+2.43** | +0.49 |
> - **三项全涨,PEM 也涨 +2.43**——不是「MF1 涨/PEM 平或跌」(那是 s30/s16/s41 过度乐观的特征),而是模型在密集长段**开对了**少数类、**补全**了更多段。这是质变信号。
> - **构造(`scripts/build_density_inject.py`,label-clean SIMI 子集)**:供体=train' 单标签 **UCM/Contra** 少数类句(无 Table/Figure 引用),宿主=len≥6 且恰好 1 个单标签 UCM/Contra 少数类的段,**跨域(CV/NLP/RL…)+ 零共享专有名词**配对,逐句注入(非 joint,避 −4.5 PEM 税),证据=anchor∪donor 并集。UCM/Contra 标签加无关证据不被满足→标签有效。`expand_record` 的 paragraph sampler 按 channel 分组→步数=记录数(3759 train' + 47 注入 = 3806,0 丢样验证)。
> - ⚠️ **densematch 是域内、必要非充分**(synthesis 警告 + s30/s16/s41 前车之鉴);testp1 OOD,+2.68 大概率打折。**s70/s71 = 全 traindev(ds0.50 base 12802 + 68 注入/600 样本 `data/density_inject_full.jsonl`)重训,交 testp1 实测。** s70=softmin(s01 配方+density,ceiling 高)、s71=plainCE(门控同配方+解相关)。目标:单模 ≥48 → 再 union s15 成员博 52。复现:`scripts/build_density_inject.py --split ALL` + `/tmp/density_prod.sh`(throwaway)。

> **🔴🔴 s67 = union 杠杆死亡判决书(2026-06-15)**:s67 是 s15 的**严格超集**(0 删、仅 +39 firing,已校验),却 **49.48 < s15 50.26(−0.78)**。逻辑闭环:**往 s15 上加任何 firing 都净亏 → 那 39 个新舰队共识开火是 FP → s15 漏判处「漏得对」(真是 Supported)→ s15=50.26 是 union 硬顶。** s68/s69 同源不传(省配额)。**到 52 的唯一路 = 一个原生更强的单模(密度匹配训练),不是 union/后处理/加成员。** 与用户判断完全一致。

> **🔴 s60-s66 实测判读(2026-06-15)= union 顶已确认枯竭,转密度匹配训练主攻:**
> - **共识 >> 单票(对同质新舰队)**:s61 mv2(48.72)比 s60 mv1(46.25)**+2.47**。新 ds0.50 舰队彼此相关,单票 OR 把相关 FP 全放进碎 PEM;但老规律(s15 用单票)仍对,因 s15 的 5 成员**高度异构**(per-sent/joint、8b/32b)。**规律:成员越异构,单票越安全;越同质,越要共识门控。**
> - **s64 aug10-mv2(49.34)= 本批最高但仍 < s15(50.26),差 −0.93。** 根因:mv2 over 10 成员把 s15 老成员各算 1 票、需 2 票才开火 → **误删了 s15「单成员独抓」的正确召回**(那正是 s15 50.26 的底)。
> - **修法 = 非对称 union(s67-s69)**:以 s15 为不可动底座(0 删,已校验),只在 s15 漏判处加新5 强共识。但**叠加空间已量化几乎枯竭**:K=2 仅净加 39 firing(其中 24 落 s15 全-Supported 净段,质量不定)、K=3 仅 10。→ **s67/s68 最多在 50.26 上下微动 ±0.5,撑不到 52。**
> - **★伪先验确证(免费实算)**:traindev GOLD 仅 **0.2%** 段含 ≥2 少数类(长段 0.7%);s01 预测 3.8%,s15 union **9.9%**。**s15 的 47.80→50.26 增益,本质是 union 用「多异构成员各开 1 个不同少数类」凑出 9.9% 的 ≥2 密度**——union 在「假装」破先验。testp1 是长密段,真实 ≥2 比例远高于 9.9% → 这是到 52 的召回缺口。
> - **★战略锁定(用户判断 + 数据一致):union 顶 = 50.26 已榨干,因底层单模顶 47.80。要 union 到 52,必先把单模抬到 48-50。** 唯一未用杠杆 = **密度匹配训练**(造长密少数类段、joint 训,让单模原生敢在密集长段开多个少数类,而非靠 union 凑)。拦路 = 脏标(少数类标签相对各自证据)。已起 workflow `wh31d06bo` 设计+对抗核实 label-valid 构造。build_dataset 无密度支持(待实现)。
> - 上传优先:**s67(K2,+39)**先试(博 50.26 上方);若 >50.26 说明加对了 → s69;若 ≤ → s68(更近 s15)兜底。s60/s64 已知 <s15 不必再传。s01/s15 永久保底。

> **🟡 s55/s56 实测判读(2026-06-15):**
> - **s55=45.48 ≠ s01=47.80(差 −2.3)→ s01 未精确复现。** 数据内容已字节验证对(ds=0.50,12741/9750),配方/loss/seed 全同,故 2.3 主因 = **单模 run-to-run 方差**(2卡 DDP + GPU 浮点非结合性,在脆性 PEM 上 ±~2)+ 可能残留(s01 用 GPU0/1、s55 用 GPU6/7)。⇒ **s01 的 47.80 是 ~45-48 方差带的高位有利抽样**;ds=0.50 确把档位拉到 45-46(远高于错数据 39-44 档),方向对,但精确 47.80 含运气。**校正"完全可复现→47.80"的过强结论:数据可复现,但训练出的分有 ±2 噪声。**
> - **s56=46.11 > s55=45.48(+0.63),matched pair(仅差 multi-target)→ multi-target 方向为正、SO 开火 3× 且未碎 PEM**(预测的"安全一档"兑现)。但 **+0.63 在 ±2 噪声带内,单跑不能定论**。
> - **两者均 < s01(47.80)< s15(50.26)。深层教训:单模杠杆的效应(~0-2)常小于 run 方差(±2),且 densematch(ρ.59)分辨不了 → 这正是 +3-5 极难、s15 难破的根因。** s15 的 +2.5(union>单模)是唯一稳健杠杆。
> - 下一步候选:s56(SO 解相关,firing 与 s15 成员不同)**并入 s15 union 一发**(博 SO 召回叠加,union 是唯一稳健 +);或收手保 s15。s01/s15 永久保底不动。

> **s55/s56 = 夜间自动链。** s55 复现 s01(期望 ~47.80,确认数据恢复闭环;ckpt-2000 = s01 同步数)。s56 = A1 multi-target(SO target 507→1626、Contra 435→579、UCM 747 不变;预检 GREEN)。判据:**s56 vs s55**——s56>s55 = multi-target 净涨(SO/Contra 召回);≈或< = 零和/FP 碎 PEM。⚠️ s56 full-traindev→densematch 泄漏,不可离线门控,胜负以 Codabench 总分为准。ADD-mode;s01/s15 永久保底。

**当前最优 = s15 u_q5(50.26),仍是天花板。** 结论:
- **union(召回叠加)是唯一超 grid 的方向**(s11→s14→s15 单调涨,但仅限**同家族 Qwen 成员**);单模都 ≤grid;后处理(s03-s07)单调掉分(testp1 召回-critical)。
- **⚠️ 架构多样性 union 证伪(2026-06-04)**:往 5-Qwen 基底加任何**非 Qwen**成员都掉分 —— s18(+q4b)49.71、s19(+iv2b)48.48、s20(+iv8b)49.36、s21(+iv2b+iv8b)47.77、**s23(+gemma26b)49.83**(最接近但仍 −0.43)。**统一规律:新成员让 MF1↑ 但 PEM↓**(s23 MF1 57.67 > s15 57.17 **+0.50**,但 PEM 41.98 < 43.34 **−1.36**)→ 新成员加对了召回(MF1)却也加了**假阳性打碎干净段落**(PEM),净亏。s15 的 5-Qwen union 已 catch 住大部分正确少数类,再加只叠 FP。
- gemma26b 单模 44.69 < grid 47.80(连强单模都算不上);InternVL 更弱。**规模/架构多样性都证伪 → s15=50.26 是 naive-union 天花板。**
- 剩余杠杆(若要破顶):**置信门控 union**(新成员的少数类预测只在 min_logprob 高时采纳,滤 FP)或 **per-class 选择性 union**(只采纳新成员擅长的类,如 gemma 的 UE)——核心是「留 MF1 增益、不伤 PEM」。s27(31B)是最后一发 naive-union,预期同 26B(~−0.4)。

---

## 复现命令(按序号)

### s01 grid — 47.80(当前最优,保持榜上)
- 数据:`build_dataset --minority-oversample 3.0 --supported-downsample 0.6`(产出 ~12741 train/9750 Supported;⚠️ 当初 downsample 未记,反推 ~0.6)。
- 训练:2 卡 DDP,`SOFTMIN_BETA=5 SOFTMIN_LAMBDA=0.5 ... scripts/train_softmin.py ... --max_pixels 401408 --use_logits_to_keep false --num_train_epochs 1`(ckpt `outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000`)。
- 推理+打包:`MAX_PIXELS=401408 bash scripts/make_testp1_submission.sh b5_l0.5` → `make_submission_zip.py` 出 `testp1_s01_m_q8b-grid.zip`。

### s02 fullres — 44.47(全分辨率,有害)
`MAX_PIXELS=802816 GPUS=0,1 bash scripts/retrain_fullres.sh`(ulk=true)。

### s03/s04 后处理阈值(零重训;先用 s01 的 grid ckpt 带 logprob 重推)
```bash
CUDA_VISIBLE_DEVICES=5 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --adapter outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000 --engine pt \
  --data-root "$DATA_ROOT" --out outputs/testp1_gridckpt_lp_raw.jsonl
uv run python scripts/threshold_variants.py --raw outputs/testp1_gridckpt_lp_raw.jsonl --out-prefix submissions/thr  # -> tau020/tau003
```

### s05/s06 孤立少数类 veto(同上 raw,零重训)
```bash
uv run python scripts/lone_minority_veto.py --raw outputs/testp1_gridckpt_lp_raw.jsonl --out submissions/s05.jsonl --mode all
uv run python scripts/lone_minority_veto.py --raw outputs/testp1_gridckpt_lp_raw.jsonl --out submissions/s06.jsonl --mode gated --gate -0.05
```

### s07 / s08 / s09 / s10 重训(网格分辨率 401408,scripts/retrain_fullres.sh)
```bash
# s07 gentle1.5: build_dataset --minority-oversample 1.5 --supported-downsample 0.66 ; TAG=gentle1.5_b5_l0.5 MAX_PIXELS=401408 GPUS=0,1 bash scripts/retrain_fullres.sh
# s08 os4:       build_dataset --minority-oversample 4.0 --supported-downsample 0.66 ; TAG=os4_b5_l0.5 ... bash scripts/retrain_fullres.sh
# s09 perclass:  build_dataset --supported-downsample 0.6 --minority-oversample-per-class "Contradiction:6,Scope Overgeneralization:5,Unsupported Causal Mechanistic:3,Unsupported Entity:2" ; TAG=perclass_b5_l0.5 ...
# s10 perclassEp2: 同 s09 数据 ; TAG=perclass_ep2_b5_l0.5 EPOCHS=2 ...
```

### s11 ensU union ensemble(零重训,纯后处理)
```bash
uv run python scripts/ensemble_union.py --out submissions/testp1_s11_u_q3.jsonl \
  submissions/testp1_s01_m_q8b-grid.jsonl submissions/testp1_s08_m_q8b-os4.jsonl submissions/testp1_s09_m_q8b-perclass.jsonl
```

### s12 / s13 段落联合(plain-CE,scripts/train_joint.sh;先 build_dataset --joint)
```bash
PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data --joint --joint-hard-oversample 2.0
# s12 8B:  TAG=joint8b  GPUS=0,1 MAX_PIXELS=401408 bash scripts/train_joint.sh
# s13 32B: MODEL_ID=Qwen/Qwen3-VL-32B-Instruct TAG=joint32b GPUS=0,1 MAX_PIXELS=401408 bash scripts/train_joint.sh
#   注:train_joint.sh 的 infer 步必须传 --model(否则 32B adapter 加载到默认 8B 基座会 state_dict 不匹配)。
```

## 约定
1. 提交前 `validate_submission.py` 必过;`make_submission_zip.py` 打扁平 zip(顶层 `track1_pred.jsonl`)。
2. **数据构造命令务必记**(含 `--supported-downsample`;grid 那次漏记,吃了亏)。
3. 新提交:取下一个 `s{NN}` → 重命名 zip/jsonl → 追加总表一行 + 复现命令 + "为什么试它"。
4. 若平台按最新提交计分,实验后**把 s01(47.80)顶回保底**。

## ensemble 规律(s11/s14/s15/s16;逐项得分见上方总表)
**关键规律:union 成员越多样 → MacroF1 和 PEM 同时单调上涨**(s01 47.8 → s11 48.2 → s14 50.0 → s15 50.3)。≥2票一致(s16)反而更差 → **纯并集(召回最大)是对的**。PEM 在这个少数类密集的集上是**召回受限**(catch 到 gold 少数类才能让长段 PEM=1),不是精度受限。→ 方向:**把架构多样性堆到极致**。

复现:`scripts/ensemble_union.py --min-votes {1|2} --out <jsonl> <成员 jsonl...>`(s11/s14/s15 用 `--min-votes 1`;s16/s22 共识用 `2`)。成员文件名见总表。

## fleet 异构成员推进(s17–s27;目标 50.26→53–56,靠架构多样性;逐项见总表)
> 方向:s15 的 5 成员全是 Qwen3-VL(高相关)。加**架构差异化**的非 Qwen 成员到并集,逼出不相关错误 → 更大并集收益。dev 泄漏不可信,每个新成员是近盲交。

- **⚠️ q4b 证伪(s17 单模 / s18 union):s18(49.71) < s15(50.26),−0.55。** MF1 微升但 **PEM 跌 1.19**:q4b 多出的少数类多是**假阳性**,打碎干净段落。**教训:规模多样性(同 Qwen 4B)无用 —— 错误相关 + 模型弱 → 只叠 FP。要的是架构多样性(非 Qwen)。** 并集基底回到 5 Qwen,q4b 踢出。
- **InternVL3(s24 iv2b / s25 iv8b 单模;s19/s20/s21 union)**:首个非 Qwen(InternViT 编码器)。8B OOM 修复 = `INTERNVL_MAX_PATCHES=2`(见记忆 internvl-fleet-config:MAX_PIXELS=401408、use_logits=false、HF_HOME 可写、lr 5e-5)。union 少数类大涨(s21=472)但**待测**是否破 50.26。
- **Gemma4-26B(s26 单模;s23 union=5Q+26B,498 少数类)**:强 + 架构异构(SigLIP 视觉塔 + MoE),理论最有希望破顶。攻克的坑:**单卡 OOM → 2 卡 deepspeed zero3**(切权重,deepspeed 0.19.1 已装);**多图 record 激活暴涨 OOM → ≤2图过滤数据** `data/train_sft_le2img.jsonl`;**infer 必传 `--template gemma4_nothinking`**(否则 thinking 泄漏 → JSON 解析崩 → 全-Supported)。**s27 = Gemma4-31B 单模仅 29.28**(只 87 少数类→漏 84% 段 gold→PEM 24.57):**模型越大反而越保守越糟,坐实"数据先验(≤1少数类/段)压过模型规模"——堆更大/更多 Gemma 已死路**。s28(5Q+31B,397)/s29(5Q+26B+31B,507)待上传但大概率 ≤50.26。
- 复现:单模 `scripts/train_gemma4.sh`(`DEEPSPEED=zero3 GPUS=4,5 MODEL_ID=google/gemma-4-31B-it TAG=gemma4_31b DATASET=data/train_sft_le2img.jsonl MAX_LENGTH=4096 bash ...`)、`scripts/train_internvl.sh`、q4b 用 `MODEL_ID=Qwen/Qwen3-VL-4B-Instruct ... retrain_fullres.sh`;union 一律 `ensemble_union.py --min-votes 1 <成员 jsonl...>`。
