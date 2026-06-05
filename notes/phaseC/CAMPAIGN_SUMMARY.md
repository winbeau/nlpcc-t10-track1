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

## Deliverable: s30 (first CLEAN candidate) — SUBMITTED, and the bench MIS-PREDICTED it
`submissions/testp1_s30_u_cleanunion` = U3 on testp1. densematch 83.90 (predicted ~47–50).
**ACTUAL testp1 = 43.85 / 48.61 / 39.08 — BELOW s01 (47.80), far below s15 (50.26). A negative result.**

**First clean (densematch, testp1) calibration point = (83.90, 43.85).** It immediately falsifies the
densematch bench for this candidate class:
- densematch ranked U3 #1 (above every single) and said union LIFTS PEM (87.38>86.41); on testp1 the
  union's added minorities were **false positives → PEM collapsed to 39.08 < s01's 42.83**. The PHASE-B
  warning (rho +0.59, joint-vs-per-sentence may not transfer) materialized hard.
- Root cause (echoes s18/q4b): s30 had only **2 truly-distinct models** (A1 & E1 share a base; B0 joint)
  + **train'-only (90% data)** + **joint-heavy** → B0 joint (testp1 ~43) dragged the union to the joint
  tier (43.85 ≈ joint8b/joint32b 43–44). vs s15 = 5 genuinely diverse full-traindev members.

## Honest takeaway
The PHASE-C bench-driven conclusions are **only as trustworthy as densematch — which this point shows is
NOT reliable for the joint-vs-per-sentence / union-composition axis.** Several "kills" (softmin, B1
density, D2) were judged on densematch and should be treated as provisional, not settled. **s15 (50.26)
remains the champion; s01 (47.80) the single-model best.** What's genuinely banked: a real calibration
point + the tooling. Most informative cheap next test = submit the pure per-sentence **A1 alone** on
testp1 (never tested standalone) — if A1 > 43.85, the joint B0 member was the drag (union HURT, like s18).

## If resumed (remaining levers)
- More **diverse union members** (the proven lever scales): e.g. a 2nd per-sentence model (different
  seed/oversample), other clean configs → add to the union.
- **E2 inference logit-adjust** toward Contra/SO at decode (the recall bottleneck) — needs per-class
  decode biasing in the generative setting.
- Retrain the U3 members on **full traindev** (not just train') for the actual final submission once the
  bench-vs-testp1 calibration is known.

s01 (47.80) / s15 (50.26) remain permanent floors. The loop never auto-submits — the user submits s30.
