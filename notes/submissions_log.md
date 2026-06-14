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
| s41 | m | **A1 自洽**:s01 + 温度自洽(T1.0 K=8 add-only union,含 greedy 成员保纯加法) | `testp1_s41_m_a1sc-s01` | **待Codabench** | — | — | 待跑完 |
| s49 | m | **λ-ablation plainCE**:A1(devbench train', λ=0)greedy testp1 | `testp1_s49_m_ablate-plainCE-trainp` | **38.68254** | — | — | 166 (79/57/17/13) |
| s50 | m | **λ-ablation softmin**:A0(devbench train', λ=0.5)greedy testp1 | `testp1_s50_m_ablate-softmin-trainp` | **34.59594** | — | — | 166 (59/84/10/13) |

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
