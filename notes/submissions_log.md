# Codabench Track-1 提交记录(命名规范 + 得分 + 复现命令)

> 官方评测:`score = (Sentence Macro-F1 + Paragraph Exact Match PEM)/2`。平台 = Codabench arena 16666(提交上限 100)。

## 命名规范(务必遵守)
```
testp1_s{NN}_{tag}.zip   (+ 同名 .jsonl)
```
- `testp1` — 阶段(Phase-1 测试集;Phase-2 用 `testp2_`)。
- **`s{NN}`** — 2 位**提交序号**,按创建顺序递增,稳定唯一 ID(上传 Codabench / 记分时用它指代)。
- `{tag}` — 简短配置标识(grid / fullres / tau020 / os4 / perclass / ensU / joint8b / joint32b …)。
- zip 内文件**永远是 `track1_pred.jsonl`**(官方扁平 zip 规范;`make_submission_zip.py` 自动处理)。
- 新增提交:取下一个 `s{NN}`,重命名 zip/jsonl,追加下面总表一行 + 复现命令。

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

## 得分总表(按序号)
| 序号 | zip (submissions/) | Score | Macro-F1 | PEM | #少数类(UCM/UE/SO/Contra) | 配置一句话 |
|---|---|---:|---:|---:|---|---|
| **s01** | `testp1_s01_grid.zip` ⭐ | **47.79897** | 52.76518 | 42.83276 | 253 (109/92/29/23) | grid 最优:β5 λ0.5,过采样 3.0×,max_pixels 401408,use_logits_to_keep=false,1ep |
| s02 | `testp1_s02_fullres.zip` | 44.46574 | 48.99973 | 39.93174 | 260 (104/108/22/26) | 全分辨率 802816 + ulk=true(**有害 −3.3**) |
| s03 | `testp1_s03_tau020.zip` | 43.85382 | 48.28784 | 39.41980 | 190 | 后处理:min_logprob<−0.20 的少数类→Supported |
| s04 | `testp1_s04_tau003.zip` | 38.47759 | 42.48420 | 34.47099 | 127 | 后处理:min_logprob<−0.03→Supported |
| s05 | `testp1_s05_vetoAll.zip` | 18.85468 | 22.00970 | 15.69966 | 46 | 孤立少数类全 veto(≈testp1 trivial 锚点) |
| s06 | `testp1_s06_vetoGated.zip` | 39.57432 | 44.33635 | 34.81229 | 171 | 孤立少数类 veto(仅退最虚 82,gate −0.05) |
| s07 | `testp1_s07_gentle15.zip` | 40.90984 | 46.83674 | 34.98294 | 246 | 降过采样 1.5×(更差→过采样别降) |
| s08 | `testp1_s08_os4.zip` | 45.10490 | 50.44872 | 39.76109 | 235 (102/80/28/25) | 均匀过采样 4×,1ep(<grid) |
| s09 | `testp1_s09_perclass.zip` | 43.83218 | 49.95104 | 37.71331 | 268 (104/112/33/19) | 逐类过采样 Contra6/SO5/UCM3/UE2,ds0.6,1ep(<grid) |
| s10 | `testp1_s10_perclassEp2.zip` | 41.69745 | 46.53484 | 36.86007 | 193 (80/74/17/22) | 逐类过采样,2ep(过拟合,最差) |
| **s11** | `testp1_s11_ensU.zip` ⭐ | **48.21778** | **54.79733** | 41.63823 | 339 (127/142/43/27) | union ensemble(s01+s08+s09)→ **新最优 +0.42**,集成是赢家方向 |
| s12 | `testp1_s12_joint8b.zip` | 43.33171 | 48.26752 | 38.39590 | 209 (90/66/31/22) | 8B 段落联合 plain-CE,1ep(<grid) |
| s13 | `testp1_s13_joint32b.zip` | 44.43184 | 49.44389 | 39.41980 | 183 (97/40/19/27) | 32B 段落联合 plain-CE,0 fallback(<grid,但 >joint8b +1.1:容量在联合内有用) |

**当前最优 = s11 ensU(48.22)**。结论:**ensemble(召回叠加)是唯一超 grid 的方向**;过采样/联合格式/容量在单模型上都 ≤grid。下一步推 ensemble(更多样并集 / ≥2 票一致),见 §ensemble。

---

## 复现命令(按序号)

### s01 grid — 47.80(当前最优,保持榜上)
- 数据:`build_dataset --minority-oversample 3.0 --supported-downsample 0.6`(产出 ~12741 train/9750 Supported;⚠️ 当初 downsample 未记,反推 ~0.6)。
- 训练:2 卡 DDP,`SOFTMIN_BETA=5 SOFTMIN_LAMBDA=0.5 ... scripts/train_softmin.py ... --max_pixels 401408 --use_logits_to_keep false --num_train_epochs 1`(ckpt `outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000`)。
- 推理+打包:`MAX_PIXELS=401408 bash scripts/make_testp1_submission.sh b5_l0.5` → `make_submission_zip.py` 出 `testp1_s01_grid.zip`。

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
uv run python scripts/ensemble_union.py --out submissions/s11.jsonl \
  submissions/testp1_s01_grid.jsonl submissions/testp1_s08_os4.jsonl submissions/testp1_s09_perclass.jsonl
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

## ensemble 推进变体(s14–s16,待提交;基于 s11 ensU=48.22 是赢家方向)
| 序号 | zip | 机制 | #少数类(UCM/UE/SO/Contra) | Score |
|---|---|---|---|---|
| s14 | testp1_s14_ensU4.zip | union(s01+s08+s09+**s13 joint32b**),加 32B 多样性 | 365 (137/148/48/32) | **49.98915** / MF1 56.80424 / PEM 43.17406 |
| **s15** | testp1_s15_ensU5.zip ⭐ | union 全 5(s01+s08+s09+s12+s13),召回最大 | 382 (139/155/52/36) | **50.25660** / MF1 57.16849 / PEM 43.34471 **(新最优)** |
| s16 | testp1_s16_ens2of4.zip | **≥2 票一致**(s01,s08,s09,s13),精度向 | 253 (110/87/32/24) | 47.28613 / MF1 52.59274 / PEM 41.97952 (<ensU,union 胜过 consensus) |

**关键规律:union 成员越多样 → MacroF1 和 PEM 同时单调上涨**(47.8→48.2→50.0→50.3)。≥2票一致(s16)反而更差 → **纯并集(召回最大)是对的**。PEM 在这个少数类密集的集上是**召回受限**(catch 到 gold 少数类才能让长段 PEM=1),不是精度受限。→ 下一步:**把多样性堆到极致**。

复现:scripts/ensemble_union.py --min-votes {1|2} --out <jsonl> <成员 jsonl...>(s14/s15 用 --min-votes 1,s16 用 2)。
