#!/usr/bin/env bash
# GATE-B PHASE B (post-inference): synthesize union + postproc anchors, then rank-correlate each
# gold variant against the known testp1 scores. Run AFTER phaseB_run.sh produces all 8 dev preds.
# CPU-only (no GPU). Idempotent — re-running overwrites the synthesized anchors.
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
P=outputs/devbench_preds
need="s01 s02 s07 s08 s09 s10 s12 s13"
for a in $need; do [ -s $P/$a.jsonl ] || { echo "MISSING $P/$a.jsonl — inference not done"; exit 1; }; done

echo "=== unions ==="
uv run python scripts/ensemble_union.py --min-votes 1 --out $P/s11.jsonl $P/s01.jsonl $P/s08.jsonl $P/s09.jsonl
uv run python scripts/ensemble_union.py --min-votes 1 --out $P/s14.jsonl $P/s01.jsonl $P/s08.jsonl $P/s09.jsonl $P/s13.jsonl
uv run python scripts/ensemble_union.py --min-votes 1 --out $P/s15.jsonl $P/s01.jsonl $P/s08.jsonl $P/s09.jsonl $P/s12.jsonl $P/s13.jsonl
uv run python scripts/ensemble_union.py --min-votes 2 --out $P/s16.jsonl $P/s01.jsonl $P/s08.jsonl $P/s09.jsonl $P/s13.jsonl

echo "=== postproc from s01 grid raw (min_logprob) ==="
# NOTE: --taus=... (with '='): argparse treats a leading-'-' value as a flag otherwise.
uv run python scripts/threshold_variants.py --raw outputs/devbench_raw/s01.jsonl --out-prefix /tmp/thr --taus=-0.20,-0.03
cp /tmp/thr_tau-0p20.jsonl $P/s03.jsonl
cp /tmp/thr_tau-0p03.jsonl $P/s04.jsonl
uv run python scripts/lone_minority_veto.py --raw outputs/devbench_raw/s01.jsonl --out $P/s05.jsonl --mode all
uv run python scripts/lone_minority_veto.py --raw outputs/devbench_raw/s01.jsonl --out $P/s06.jsonl --mode gated --gate -0.05

echo "=== anchors present ==="; ls -1 $P
for v in raw reshaped densematch; do
  echo; echo "############### GOLD VARIANT: $v ###############"
  PYTHONPATH=src uv run python scripts/rankcorr_meta.py --gold data/devbench/dev_gold_$v.jsonl --pred-dir $P --data-root "$DATA_ROOT" --min-anchors 5
done
