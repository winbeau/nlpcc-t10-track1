# PHASE C campaign summary (2026-06-04, autonomous /loop on 2×H200)

Ran the exploration roadmap as a self-paced loop under the **iron law**: train ONLY on
`data/devbench/train_sft.jsonl` (image-disjoint train', never sees dev'), judge offline on
`data/devbench/dev_gold_densematch.jsonl` (the GATE-B bench), package candidates for testp1 without
ever burning a Codabench submission. Every result below is leakage-free.

## Clean densematch scoreboard

| system | score | MF1 | PEM | verdict |
|---|--:|--:|--:|---|
| A0 softmin (s01 recipe) | 78.97 | 75.42 | 82.52 | **DEAD** — softmin loses to plain-CE |
| A1 plain-CE (per-sentence) | 81.13 | 79.24 | 83.01 | baseline |
| E1 self-consist (3× A1) | 82.53 | 80.10 | 84.95 | KEEP (+1.4) |
| B0 joint plain-CE | 82.89 | 79.37 | 86.41 | best single (joint > per-sentence) |
| B1 density-match | 80.82 | 78.15 | 83.50 | **DEAD** — −2.9 PEM (over-fires minorities) |
| E1b self-consist on B0 | 81.18 | 77.89 | 84.47 | **DEAD** — self-consist hurts joint |
| **U3 = B0 + A1 + E1** | **83.90** | **80.42** | **87.38** | ★ **BEST = the system** |
| (U4 = +E1b) | 82.85 | — | — | worse — weak member drags union |

## What WORKED
- **Diverse-model union is the lever** (exactly the testp1 s15 winner, now reconfirmed on clean
  models). U3 = union(joint B0 + per-sentence A1 + self-consistency E1) = 83.90, +1.0 over the best
  single on BOTH score and PEM — added minorities are true positives, not FP. Robust → good for Phase-2.
- **Self-consistency** (temp=0.7, 3 samples, union) on a **per-sentence** model: +1.4 (additive recall).
- **plain-CE > softmin** — softmin was a relic of the refuted precision framing tuned on the leaky dev.
- **joint > per-sentence on the bench** (PEM +3.4) — but see the caveat.

## What DIED (killed cleanly, ZERO Codabench burn)
- **softmin** (A1 > A0). **B1 density-matching** — injecting 359 synthetic ≥2-minority paragraphs
  taught over-firing → −2.9 PEM (the roadmap's "recall-for-FP" risk, materialized). **D2 chart-reading**
  — clean minority misses split 47% image / 53% table, NOT image-bound → no OCR/chart channel.
  **E1b** (self-consistency on a joint model — varies the whole array → adds FP).

## The robust bottleneck
Rare-class **recall** persists even in the best single: Contradiction R≈22% (fires ~4 of 9, P≈100%),
SO R≈63%. The union lifts it via diverse-member recall; it is NOT a vision/chart problem.

## Deliverable: s30 (first CLEAN candidate)
`submissions/testp1_s30_u_cleanunion.{zip,jsonl}` = U3 on testp1 (269 minorities). densematch 83.90.
**Predicted testp1 ≈ 47–50, likely ≤ s15 (50.26)** (only 3 members + train'-only; 269 min < s15's 382).
⚠️ densematch's "joint > per-sentence" may not transfer (testp1 history: joint8b 43.33 < grid 47.80;
PHASE B rho only +0.59) — so this is NOT claimed as a break past s15. **The real value: submitting s30
yields the FIRST clean (densematch, testp1) calibration point** → makes the bench trustworthy and starts
the cumulative protocol.

## If resumed (remaining levers)
- More **diverse union members** (the proven lever scales): e.g. a 2nd per-sentence model (different
  seed/oversample), other clean configs → add to the union.
- **E2 inference logit-adjust** toward Contra/SO at decode (the recall bottleneck) — needs per-class
  decode biasing in the generative setting.
- Retrain the U3 members on **full traindev** (not just train') for the actual final submission once the
  bench-vs-testp1 calibration is known.

s01 (47.80) / s15 (50.26) remain permanent floors. The loop never auto-submits — the user submits s30.
