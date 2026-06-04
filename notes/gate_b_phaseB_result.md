# GATE-B PHASE B result — leaky-adapter rank-corr (2026-06-04)

> Ran on 4×H200 (dasys-h200x8). Inferred 8 Qwen anchors on the image-disjoint devbench dev (501 rec),
> synthesized 4 unions + 4 postproc → 16 anchors, rank-correlated each gold variant's dev score against
> the known testp1 scores. Drivers: `scripts/phaseB_run.sh` (dev inference) + `scripts/phaseB_synth_rankcorr.sh`
> (synthesis + rankcorr). Background: [[gate_b_plan]] PHASE B, [[eval_bench_design]] caveat #2 (legacy leakage).

## Verdict: the bench SHAPE is validated, but leaky legacy adapters cannot trust-rank it.

| gold variant | n_rec | **Spearman(dev_score, testp1)** | dev_MF1 | dev_PEM | B6 |
|---|--:|--:|--:|--:|---|
| raw (traindev shape) | 501 | **−0.016** | +0.222 | −0.141 | ABANDON |
| reshaped (≥8 & 84%, tiny n) | 30 | **+0.222** | +0.432 | +0.191 | ABANDON |
| **densematch (84% density, n=206)** | 206 | **+0.590** | +0.590 | +0.541 | **WEAK smell-test** |

- **Monotonic raw → reshaped → densematch (−0.02 → +0.22 → +0.59)**: the testp1-shaping direction is
  *correct* — density-matching recovers real signal. densematch even crosses into "usable as a smell test."
- **But none reaches 0.8 (trustworthy) or even 0.6 (coarse filter).** Capped by **memorization leakage**:
  the legacy adapters trained on full traindev → they have SEEN these dev' records.

## Why: leakage inverts the ranking (the smoking gun)

dev scores are compressed in **90–98** (vs testp1's 40–50) — massive memorization. The model that *overfit
hardest* memorizes dev best, which **anti-correlates** with testp1 generalization:

| anchor | what | dev_score (raw) | testp1_score |
|---|---|--:|--:|
| **s10** | perclass **2-epoch** (most overfit) | **98.05 (highest)** | **41.70 (3rd worst)** |
| **s15** | 5-Qwen union (testp1 champion) | 91.47 (among lowest) | **50.26 (best)** |

And the union knee inverts for the same reason: on densematch, unions **s11/s14 (dev 94.14) outrank s15
(dev 92.74)**, yet testp1 has s15 > s14 > s11 — because s15's extra member adds testp1-correct minorities
that *don't* match the memorized dev pattern, *lowering* leaky-dev while *raising* testp1. Diverse union =
better testp1 = lower leaky-dev. (Union-knee "MISSES" on all variants; treat as low-confidence anyway —
[[eval_bench_design]]: ~200–500 rec is underpowered for the 50.0–50.3 union top regardless.)

Sanity that the *scale* is right (only the middle is leakage-compressed): on densematch the **trivial
s05 dev = 18.90 ≈ testp1 18.85** — the bench anchors the all-Supported floor exactly where testp1 does.

## densematch — full 16-anchor table (the best variant)

```
  anchor   T    dev_sc  dev_MF1  dev_PEM |   tp_sc  tp_MF1  tp_PEM
  --------------------------------------------------------------------------
  s15      u     92.74    94.23    91.26 |   50.26   57.17   43.34
  s14      u     94.14    95.56    92.72 |   49.99   56.80   43.17
  s11      u     94.14    95.56    92.72 |   48.22   54.80   41.64
  s01      m     92.35    93.44    91.26 |   47.80   52.77   42.83
  s16      u     95.97    96.80    95.15 |   47.29   52.59   41.98
  s08      m     95.78    96.40    95.15 |   45.10   50.45   39.76
  s02      m     92.07    92.89    91.26 |   44.47   49.00   39.93
  s13      m     91.75    92.72    90.78 |   44.43   49.44   39.42
  s03      pp    91.98    92.69    91.26 |   43.85   48.29   39.42
  s09      m     94.73    96.26    93.20 |   43.83   49.95   37.71
  s12      m     92.09    92.91    91.26 |   43.33   48.27   38.40
  s10      m     97.51    98.42    96.60 |   41.70   46.53   36.86
  s07      m     90.63    91.45    89.81 |   40.91   46.84   34.98
  s06      pp    89.98    91.61    88.35 |   39.57   44.34   34.81
  s04      pp    88.37    90.33    86.41 |   38.48   42.48   34.47
  s05      pp    18.90    22.27    15.53 |   18.85   22.01   15.70
```

Anchors: 8 inferred Qwen singles (s01 grid, s02 fullres, s07 gentle, s08 os4, s09 perclass, s10 perclass-ep2,
s12 joint8b, s13 joint32b) + 4 unions (s11/s14/s15=5Q/s16=2of4) + 4 postproc on the s01 grid raw
(s03 τ−0.20, s04 τ−0.03, s05 vetoAll, s06 vetoGated−0.05). All 16 carry known testp1 scores.

## Decision → PHASE C (the only clean path)

The bench's *shape* is right (recall-critical trivial baseline + monotonic rho gain from shaping), but its
*trustworthiness for selection* is gated by leakage. We CANNOT validate it with the existing adapters; the
+0.59 densematch rho is a **lower bound** (leakage suppresses it). The clean route, per plan PHASE C:

> **Cumulative protocol.** Every future candidate is (a) trained ONLY on `data/devbench/train_sft.jsonl`
> (image-disjoint train', never sees dev'), (b) scored on `data/devbench/dev_gold_densematch.jsonl` offline,
> and (c) also submitted to testp1. Accumulate clean (dev, testp1) pairs — rank-corr builds up leakage-free,
> and the first clean point doubles as the **density-matched model** (P1, the breakthrough lever). Use
> **densematch** as the gold variant (best of the three; raw is useless, reshaped is too small).

Until then: densematch may kill *large* bad directions (a candidate that tanks densematch is suspect), but
it cannot pick union-top winners — those still each cost 1 Codabench shot.
