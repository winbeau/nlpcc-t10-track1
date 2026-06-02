# Overnight autonomous plan (2026-06-02 night, user asleep)

> Durable record so the /loop and post-compaction me know the state + queue + decision rules.
> Goal: don't waste H200 GPU 0,1; pursue the PROVEN lever (correct minority RECALL — see
> `memory/testp1-result-v1.md`: testp1 trivial≈19, recall-critical, oversample UP not down).
> Best so far = grid b5_l0.5 = **47.80** (`submissions/testp1_b5_l0.5_submission.zip`). Beat it.

## Workflow v2 RESULT (drove the plan below) — `w22wxhs3w` DONE
Ground-truthed from the 7 on-disk submissions: testp1 per-class minority preds (grid 47.8) =
UCM 109 / **UE 92** / **SO 29** / **Contra 23**. Training targets (primary types[0]) =
UCM 276 / UE 544 / **SO 188** / **Contra 101**. ⇒ **SO + Contradiction are the under-fired,
structurally-starved classes; UE is fine (do NOT raise).** Top rec = PER-CLASS oversample
(Contra 6×, SO 5×, UCM 3×, UE 2×, ds 0.6), NOT uniform. Also: 32% image leakage inflates dev
(don't trust dev=88); 32B only via paragraph-joint (per-sentence 32B OOMs). Full result:
`/tmp/claude-1000/.../tasks/w22wxhs3w.output`.

## Running now
- **os4 retrain** (uniform oversample 4.0, 1ep) → `/tmp/p0_retrain_os4.log` → `testp1_os4_b5_l0.5.zip`
  (SUNK COST — uniform inflates UE; submit it but the real move is per-class below).
- **32B download** (`Qwen/Qwen3-VL-32B-Instruct` → ms_cache) → `/tmp/dl_32b.log` (no GPU).
- **scripts/overnight_recall_queue.sh** (REWRITTEN) → `/tmp/overnight_queue.log`: after os4 frees GPU,
  runs **perclass** then **perclass_ep2** (per-class oversample, ds 0.6, 1ep & 2ep).

## Recall-max queue (PER-CLASS, @ grid res 401408)
| TAG | rates | epochs | zip | status |
|---|---|---|---|---|
| (grid) | uniform 3.0× | 1 | testp1_b5_l0.5 | ✅ 47.80 (ref / fallback) |
| (B) | uniform 1.5× | 1 | testp1_gentle1.5 | ✅ 40.91 (worse → up) |
| os4_b5_l0.5 | uniform 4.0× | 1 | testp1_os4_b5_l0.5 | running (sunk) |
| **perclass_b5_l0.5** | Contra6/SO5/UCM3/UE2, ds0.6 | 1 | testp1_perclass_b5_l0.5 | queued (TOP rec) |
| **perclass_ep2_b5_l0.5** | same | 2 | testp1_perclass_ep2_b5_l0.5 | queued |
Target after retrain: SO preds 29→60-90, Contra 23→45-70, UE stays ~85-100 (mine per-class counts
of each new submission; if Contra/SO over-fire on clean paras it back-fires — watch the counts).

## /loop decision rules (every 30 min)
1. Poll `/tmp/p0_retrain_os4.log`, `/tmp/overnight_queue.log`, `/tmp/dl_32b.log`, GPU 0/1.
2. When a `submissions/testp1_<TAG>_submission.zip` appears: `git add -f` it (+ the .jsonl) → commit → push;
   pull local. Append a row to `notes/submissions_log.md` (score TBD — user submits in the morning).
3. If a run **OOM/crashes**: log it, do NOT blindly relaunch; note for morning.
4. When **32B download done** (`DONE_DOWNLOAD` in log) AND **workflow v2 done**: read the v2 roadmap.
   - If v2 says 32B worth it → it REQUIRES paragraph-joint reformat (per-sentence 32B OOMs). That reformat
     is NEW code (build_dataset joint format + aggregate parse + loss port) — implement + SMOKE on 8B first;
     do NOT launch a full 32B train on unvalidated code unattended. If reformat smoke passes + time allows,
     launch 8B-paragraph-joint (validates), then 32B-paragraph-joint.
   - If v2 prioritizes other recall-max levers (per-class, ensemble, caption-OCR) → queue those instead.
5. Keep GPU 0,1 busy; if the queue finishes and 32B isn't ready, consider an ENSEMBLE-union of the best
   2-3 checkpoints (fire minority if any model does) — recall-additive, matches the proven direction.

## Morning summary to prepare
Scoreboard of all overnight retrains (zips ready to submit) + recommended submit order + 32B status +
v2 workflow roadmap digest. User submits the zips (Codabench cap 100, ~10 used).
