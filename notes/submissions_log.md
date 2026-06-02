# Codabench Track-1 提交记录（得分 + 复现命令）

> 每次提交都登记在这里：官方分数 + **能从头复现该提交的命令**。新提交追加一行表格 + 一节复现步骤。
> 官方评测：`score = (Sentence Macro-F1 + Paragraph Exact Match PEM) / 2`。平台 = Codabench arena 16666（提交上限 100）。

## 关键背景（务必先读）
- **testp1 的 trivial（全 Supported）≈ 19 分**（实测：`veto_all` 99% Supported → 18.85）。**不是** train-dev 的 43.2。
  testp1 段落长（均值 8.7、最长 31 句）且少数类密集（~84% 段含 ≥1 个 gold 少数类）。
- ⇒ **testp1 是「召回-critical」**：必须把少数类句判对才拿得到 PEM。**抑制少数类（veto/阈值/降过采样）单调掉分**。
- ⇒ 方向：**最大化「正确的」少数类召回**（过采样往高、容量、ensemble…），而不是抑制。详见 `memory/testp1-result-v1.md`。

## 环境（H200，所有命令在服务器跑）
```bash
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache
# 打包：uv run python scripts/make_submission_zip.py --sub <jsonl> --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out <zip>
# 校验：uv run python scripts/validate_submission.py --sub <jsonl> --ref "$DATA_ROOT/data/testp1-track-1.jsonl"
```

## 得分总表（按 Score 降序）
| # | 提交文件 (zip) | Score | Macro-F1 | PEM | #少数类 | 配置一句话 |
|---|---|---:|---:|---:|---:|---|
| 1 | `testp1_b5_l0.5_submission.zip` ⭐ | **47.79897** | 52.76518 | 42.83276 | 253 | grid 最优：β5 λ0.5，过采样 **3.0×**，max_pixels 401408（392 tok/img），use_logits_to_keep=false，1 epoch |
| 2 | `testp1_gridckpt_thr_tau-0p20.zip` | 43.85382 | 48.28784 | 39.41980 | 190 | A·置信度阈值：把 min_logprob<−0.20 的少数类退回 Supported |
| 3 | `testp1_fullres_b5_l0.5_submission.zip` | 44.46574 | 48.99973 | 39.93174 | 260 | 全分辨率：max_pixels **802816**（784 tok/img）+ use_logits_to_keep=true（**有害 −3.3**） |
| 4 | `testp1_gentle1.5_b5_l0.5_submission.zip` | 40.90984 | 46.83674 | 34.98294 | 246 | B·降过采样 **1.5×**（更差 → 过采样别降） |
| 5 | `testp1_veto_gated.zip` | 39.57432 | 44.33635 | 34.81229 | 171 | #1·孤立少数类 veto（仅退最虚的 82 个，gate −0.05） |
| 6 | `testp1_gridckpt_thr_tau-0p03.zip` | 38.47759 | 42.48420 | 34.47099 | 127 | A·置信度阈值：min_logprob<−0.03 退回 |
| 7 | `testp1_veto_all.zip` | 18.85468 | 22.00970 | 15.69966 | 46 | #1·孤立少数类 veto（全退 207 个）≈ testp1 trivial |

**结论**：#1 grid(47.8) 至今最优；所有"抑制少数类"的变体都更低，单调随少数类数下降 → testp1 召回-critical。

---

## 复现命令（逐提交）

### #1 grid b5_l0.5 — 47.79897（当前最优，保持为榜上提交）
- **数据**：过采样 3.0× + Supported 下采样（产出 12741 train 样本 / 9750 Supported；⚠️ 当初的 `--supported-downsample` 具体值未记录，反推约 0.55–0.66 区间，逐类过采样确认为 3.0×）。重建近似命令：
  ```bash
  PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data \
    --minority-oversample 3.0 --supported-downsample 0.6   # 调到 Supported≈9750 / 共≈12741
  ```
- **训练**（2 卡 DDP，β5 λ0.5）：经 `scripts/grid_search.py` 跑出；等价单配置命令：
  ```bash
  CUDA_VISIBLE_DEVICES=0,1 SOFTMIN_BETA=5 SOFTMIN_LAMBDA=0.5 \
  PYTHONPATH=src uv run torchrun --nproc_per_node=2 --master_port=29531 scripts/train_softmin.py \
    --model Qwen/Qwen3-VL-8B-Instruct --tuner_type lora --torch_dtype bfloat16 \
    --dataset data/train_sft.jsonl --split_dataset_ratio 0 --loss_type softmin_pem \
    --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --lora_rank 16 --lora_alpha 32 --freeze_vit true \
    --max_length 4096 --max_pixels 401408 --attn_impl sdpa --packing false --padding_free false \
    --use_logits_to_keep false --save_strategy epoch --save_total_limit 1 --output_dir outputs/grid_b5_l0.5
  # checkpoint: outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000
  ```
- **推理+提交**：
  ```bash
  MAX_PIXELS=401408 bash scripts/make_testp1_submission.sh b5_l0.5     # -> outputs/testp1_b5_l0.5_submission.jsonl
  uv run python scripts/make_submission_zip.py --sub submissions/testp1_b5_l0.5_submission.jsonl \
    --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out submissions/testp1_b5_l0.5_submission.zip
  ```

### #3 full-res b5_l0.5 — 44.46574（全分辨率，已证有害）
```bash
MAX_PIXELS=802816 GPUS=0,1 bash scripts/retrain_fullres.sh     # TAG 默认 fullres_b5_l0.5；自带 infer+aggregate+zip
# ckpt: outputs/p0_fullres_b5_l0.5/v0-20260602-135956/checkpoint-2000
# 产物: submissions/testp1_fullres_b5_l0.5_submission.zip
```

### #4 B gentle 1.5× — 40.90984（降过采样，已证更差）
```bash
PYTHONPATH=src uv run python -m nlpcc_t10.build_dataset --data-root "$DATA_ROOT" --out data \
  --minority-oversample 1.5 --supported-downsample 0.66          # 12804 train 样本
TAG=gentle1.5_b5_l0.5 MAX_PIXELS=401408 GPUS=0,1 bash scripts/retrain_fullres.sh
# ckpt: outputs/p0_gentle1.5_b5_l0.5/v0-20260602-151948/checkpoint-1400
# 产物: submissions/testp1_gentle1.5_b5_l0.5_submission.zip
```

### #2 / #6 A·置信度阈值变体（在 grid checkpoint 上后处理，零重训）
```bash
# 1) 用修好 logprob 的 infer.py 重推 grid ckpt（带 min_logprob）
CUDA_VISIBLE_DEVICES=5 MAX_PIXELS=401408 PYTHONPATH=src uv run python -m nlpcc_t10.infer \
  --split testp1 --adapter outputs/grid_b5_l0.5/v0-20260602-095341/checkpoint-2000 \
  --engine pt --data-root "$DATA_ROOT" --out outputs/testp1_gridckpt_lp_raw.jsonl
# 2) 生成阈值变体（自动按 25/50/75 百分位选 tau）
uv run python scripts/threshold_variants.py --raw outputs/testp1_gridckpt_lp_raw.jsonl \
  --out-prefix submissions/testp1_gridckpt_thr
#   -> ..._tau-0p20.jsonl (190 少数类, 43.85)  /  ..._tau-0p03.jsonl (127, 38.48)
# 3) 打包：uv run python scripts/make_submission_zip.py --sub <variant>.jsonl --ref "$DATA_ROOT/data/testp1-track-1.jsonl" --out <variant>.zip
```

### #5 / #7 #1·孤立少数类 veto（同上 raw，零重训）
```bash
uv run python scripts/lone_minority_veto.py --raw outputs/testp1_gridckpt_lp_raw.jsonl \
  --out submissions/testp1_veto_all.jsonl   --mode all                    # 退全部 207 -> 18.85
uv run python scripts/lone_minority_veto.py --raw outputs/testp1_gridckpt_lp_raw.jsonl \
  --out submissions/testp1_veto_gated.jsonl --mode gated --gate -0.05     # 退最虚 82 -> 39.57
# 打包同上
```

---

## 约定（往后每次提交都做）
1. 提交前 `validate_submission.py` 必过；`make_submission_zip.py` 打扁平 zip（顶层 `track1_pred.jsonl`）。
2. **数据构造命令务必记进本表**（grid 那次没记 `--supported-downsample`，吃了亏）。
3. 新提交追加：表格一行 + 一节复现命令 + 该配置的"为什么试它"。
4. 当前最优 = #1（47.80）。若平台按最新提交计分，传完实验变体后**记得把 `testp1_b5_l0.5_submission.zip` 顶回去**。

- **os4_b5_l0.5** (uniform 4x, 1ep, grid res) testp1 per-class: UCM=102 UE=80 SO=28 Contra=25 (minority 235, Supported 4861) | Score TBD (submit AM)

- **perclass_b5_l0.5** (Contra6/SO5/UCM3/UE2, ds0.6, 1ep, grid res) testp1 per-class: UCM=104 UE=112 SO=33 Contra=19 (minority 268, Supported 4828) | Score TBD (submit AM). vs grid UCM109/UE92/SO29/Contra23
