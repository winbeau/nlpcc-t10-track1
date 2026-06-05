# Full-traindev plain-CE retrain + re-union (the shot at > s15 50.26)

## ✅ STATUS = COMPLETE (2026-06-05). 全部 s33-s38 已打包+入库+记 log,不要重复打包/记录。
**结果:plain-CE 路否决。** s37(plain-CE 并集)48.54 < s15 50.26;s33(grid plain-CE)42.87 << s01 47.80 —— full-traindev+过采样下 softmin > plain-CE(plain-CE 过度开火→PEM 崩)。s36(gemma plain-CE 单模)46 少数类(极保守)、s38(+gemma)405 少数类,都 待交、大概率 ≤ s15。**s01(47.80)/s15(50.26)仍天花板。GPU 已被用户收回(显卡停了),gemma 的 aggregate 用 CPU 补完。** 详见 submissions_log.md s33-s38。


2026-06-05. Validated on testp1 (s31 softmin 34.21 < s32 plain-CE 38.68, +4.47): plain-CE > softmin.
Apply to s15's recipe at FULL traindev + oversample (remove the train' handicap).

## GPU constraint: ONLY cards 5,6,7 (user, 2026-06-05). dual-card -> 6+7. GPU0 = another user's vllm, hands-off.

## Members being retrained as PLAIN-CE (SOFTMIN_LAMBDA=0, full traindev, grid res 401408)
- **perclass** (GPU5): per-class oversample (Contra6/SO5/UCM3/UE2, ds0.6) → outputs/testp1_pce_perclass.jsonl
- **grid** (GPU6): `--minority-oversample 3.0 --supported-downsample 0.6` (= s01 recipe, plain-CE) → outputs/testp1_pce_grid.jsonl
- **os4** (GPU7): `--minority-oversample 4.0 --supported-downsample 0.66` → outputs/testp1_pce_os4.jsonl
- **gemma4_26b_pce** (DEFERRED → GPU6+7 zero3, after grid+os4 free them ~2h): `LAMBDA=0 DATASET=train_sft_le2img DEEPSPEED=zero3 GPUS=6,7 MODEL_ID=google/gemma-4-26B-A4B-it TAG=gemma4_26b_pce MAX_PIXELS=401408 bash scripts/train_gemma4.sh` → submissions/testp1_gemma4_26b_pce_submission.jsonl
- ⚠️ DONE/FAILED markers UNRELIABLE for the currently-running jobs (old retrain_member.sh had a buggy ERR trap → spurious FAILED while healthy; fixed in 091181d but running jobs predate it). **Real done-signal = `outputs/testp1_pce_<tag>.jsonl` exists** (only written after infer completes); RUNNING = `pgrep -f "retrain_member.sh <tag>"`.

## Reuse (already plain-CE)
- joint8b = submissions/testp1_s12_m_q8b-joint.jsonl, joint32b = submissions/testp1_s13_m_q32b-joint.jsonl.

## 打包编号方案（singles + unions 全部打包,s{NN} 顺排;上一个是 s32）
**单模(make_submission_zip 每个 testp1_pce_<tag>.jsonl):**
- **s33** = grid plain-CE → `testp1_s33_m_grid-plainCE.{zip,jsonl}`
- **s34** = os4 plain-CE → `testp1_s34_m_os4-plainCE`
- **s35** = perclass plain-CE → `testp1_s35_m_perclass-plainCE`
- **s36** = gemma26b plain-CE → `testp1_s36_m_gemma26b-plainCE`（来源 submissions/testp1_gemma4_26b_pce_submission.jsonl）

**并集(ensemble_union.py --min-votes 1):**
- **s37** = 5-Qwen plain-CE = grid+os4+perclass(新 plain-CE) + joint8b(s12) + joint32b(s13) → `testp1_s37_u_q5-plainCE` ★ 直接对标 s15(50.26)
- **s38** = s37 + gemma26b plain-CE(6 成员) → `testp1_s38_u_q5-plainCE-gemma`

每个打包后 `validate_submission.py` 必过,追加 `notes/submissions_log.md` 一行(序号/类型/desc/待交/少数类数 + 复现)。commit 到 main。用户手动交 Codabench。
轮询节奏:**30min**(正常则等下次,有问题则修)。Drivers: retrain_member.sh / train_gemma4.sh。s01(47.80)/s15(50.26)永久保底。
