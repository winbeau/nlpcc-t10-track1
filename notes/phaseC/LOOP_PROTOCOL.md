# PHASE C loop protocol (resumable; any session can pick this up)

The self-paced `/loop` driving the exploration roadmap. Each wake-up is ONE autonomous turn. State
is the files in this dir + remote markers — **never trust memory; reconstruct every tick**.

## Files
- `queue.jsonl` — experiment configs; each has `state ∈ {QUEUED, LAUNCHED, DONE, FAILED, SKIP}`.
- `status.jsonl` — append-only event journal `{name, state, ts, gpu, score?, note}`.
- `ledger.csv` — written by `eval_local --log`; the numeric record (score/MF1/PEM/per-class F1).
- `loop_state.json` — convenience cursor (not source of truth).
- Remote: `outputs/phaseC/<name>/{DONE,FAILED,run.log}` — completion markers + log.

## Constants (server: tmux session `nlpcc-t10-track1-h200`, repo /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1)
- GPU CAP = **2** (shared box; yield if others occupy cards). Train = single-GPU (`PHASEC_GPU=<g>`).
- Harness: `scripts/phaseC_run_experiment.py <name>` (train|infer_only|soup). `kind:analysis` runs
  `scripts/phaseC_error_analysis.py` directly (CPU, no harness).
- Launch (detached): `nohup env PHASEC_GPU=<g> python3 scripts/phaseC_run_experiment.py <name> > outputs/phaseC/<name>/run.log 2>&1 &`

## Wake-up algorithm
1. **Reconstruct**: for each LAUNCHED exp, check remote markers (authoritative).
2. **Poll** (one batched tmux-inject per LAUNCHED exp):
   `O=outputs/phaseC/<name>; test -f $O/DONE && echo DONE || (test -f $O/FAILED && echo FAILED || (pgrep -af "phaseC_run_experiment.py <name>" >/dev/null && echo RUNNING || echo STALLED))`
   + `tail -n 15 $O/run.log`. STALLED only if no-marker+no-proc+no-log-growth across **2** ticks.
3. **Finalize** DONE: read the new `ledger.csv` row for `<tag>` → append `status` RESULT → apply the
   **gate** (below) → flip downstream QUEUED→eligible/SKIP. FAILED: record + **surface to user, no auto-retry**.
4. **Launch** next eligible (deps all DONE, gate allows, not leaky-blocked): `nvidia-smi` free GPUs,
   respect CAP=2, idempotency-guard (no live proc, no DONE). Fill free GPUs.
5. **ScheduleWakeup**: interval ≈ clamp(0.1×expected_remaining, 60, 600)s. trainings ~8-10min,
   E1 infer ~5min, CPU ~60-90s, gate-blocked ~10min. All terminal → write campaign summary + STOP.

## Decision gate (densematch, CLEAN-vs-CLEAN vs the A0 baseline)
- **KEEP** iff PEM ≥ best_clean_PEM − 0.5 AND macro_f1 > baseline AND class dist moves toward gold
  (Sup85.7/SO7.7/UE2.9/Contra2.1/UCM1.6) without over-firing any minority >2× its gold share.
- **DISCARD** iff score < baseline − 1.0 OR PEM down >1.
- A1 special: if plain-CE ≥ A0 → softmin dead.
- ⚠️ NEVER compare against s15's leaky densematch 92.74. Leaky probes (C1_soup_probe) are not gated.

## Candidate packaging (no Codabench submit — user submits manually)
When a clean run clearly beats A0 (KEEP + margin): also infer testp1 → aggregate →
`scripts/make_submission_zip.py` → `submissions/testp1_s{NN}_{m|u|pp}_{desc}.{zip,jsonl}` (next seq
`s{NN}`), and append a `notes/submissions_log.md` row with the densematch metrics + a **predicted
testp1 high/low** band (vs s01=47.80 / s15=50.26; reasoning from PEM-held + recall-up). Sharpens as
the user reports real testp1 scores (accumulating clean (densematch,testp1) pairs).

## Waves (queue more as gates pass)
- W1: A0, A1, D1, E1(dep A0), C1_soup_probe.  W2: A2 calib-loss, A3 SAM+smooth, B1 density-match★.
- W3 (conditional): D2 if D1 image-bucket errors dominate; B2 if minorities still starved; E2 logit-adj.
  Final: union orthogonal KEEP winners (each individually KEEP).
