# Overnight autonomous plan (2026-06-02 night, user asleep)

> Durable record so the /loop and post-compaction me know the state + queue + decision rules.
> Goal: don't waste H200 GPU 0,1; pursue the PROVEN lever (correct minority RECALL — see
> `memory/testp1-result-v1.md`: testp1 trivial≈19, recall-critical, oversample UP not down).
> Best so far = grid b5_l0.5 = **47.80** (`submissions/testp1_b5_l0.5_submission.zip`). Beat it.

## Running now
- **os4 retrain** (oversample 4.0, ds 0.66, 1ep, grid res 401408, GPU 0,1) → `/tmp/p0_retrain_os4.log`
  → `submissions/testp1_os4_b5_l0.5_submission.zip`.
- **32B download** (`Qwen/Qwen3-VL-32B-Instruct` → ms_cache) detached → `/tmp/dl_32b.log` (no GPU).
- **scripts/overnight_recall_queue.sh** detached → `/tmp/overnight_queue.log`: after os4 frees GPU,
  runs **os5** (oversample 5.0, 1ep) then **ep2** (oversample 3.0, **2 epochs**). Each = build+train+infer+zip.
- **Workflow v2** `w22wxhs3w` (recall re-analysis) — pending; act on its roadmap when it completes.

## Recall-max queue (oversample/epoch axis @ grid res, ds 0.66)
| TAG | oversample | epochs | zip | status |
|---|---|---|---|---|
| (grid) | 3.0 | 1 | testp1_b5_l0.5 | ✅ 47.80 (ref) |
| (B) | 1.5 | 1 | testp1_gentle1.5_b5_l0.5 | ✅ 40.91 (worse → up) |
| os4_b5_l0.5 | 4.0 | 1 | testp1_os4_b5_l0.5 | running |
| os5_b5_l0.5 | 5.0 | 1 | testp1_os5_b5_l0.5 | queued |
| ep2_b5_l0.5 | 3.0 | 2 | testp1_ep2_b5_l0.5 | queued |

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
