#!/usr/bin/env bash
# Package the best clean PHASE-C system = U3 union(B0 joint + A1 per-sentence + E1 self-consist-of-A1)
# on testp1 -> submissions/testp1_s30_u_cleanunion.{zip,jsonl}. Members trained on train' (devbench),
# so this also yields the FIRST clean (densematch 83.90, testp1 ?) calibration point. USER submits.
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache USE_HF=0 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
M=Qwen/Qwen3-VL-8B-Instruct; REF="$DATA_ROOT/data/testp1-track-1.jsonl"
O=outputs/phaseC/package; mkdir -p $O
B0=$(ls -d outputs/phaseC/B0_joint_baseline/ckpt/v*/checkpoint-* | sort | tail -1)
A1=$(ls -d outputs/phaseC/A1_plainCE/ckpt/v*/checkpoint-* | sort | tail -1)
echo "B0=$B0  A1=$A1"

inf(){ CUDA_VISIBLE_DEVICES=$1 MAX_PIXELS=401408 uv run python -m nlpcc_t10.infer --split testp1 \
        --model $M --adapter $2 --engine pt --data-root "$DATA_ROOT" --out $3 "${@:4}"; }
agg(){ uv run python -m nlpcc_t10.aggregate --pred $1 --ref "$REF" --out $2; }

# B0 (joint) on GPU3, in background; A1 + 3 self-consistency samples of A1 (E1) on GPU4
( inf 3 "$B0" $O/B0_raw.jsonl --joint && agg $O/B0_raw.jsonl $O/B0.jsonl && echo "B0 testp1 done" ) &
inf 4 "$A1" $O/A1_raw.jsonl && agg $O/A1_raw.jsonl $O/A1.jsonl && echo "A1 testp1 done"
for k in 0 1 2; do
  inf 4 "$A1" $O/E1_k$k.raw.jsonl --temperature 0.7 --seed $k && agg $O/E1_k$k.raw.jsonl $O/E1_k$k.jsonl
done
uv run python scripts/ensemble_union.py --min-votes 1 --out $O/E1.jsonl $O/E1_k0.jsonl $O/E1_k1.jsonl $O/E1_k2.jsonl
echo "E1 self-consistency union done"
wait  # ensure B0 finished
uv run python scripts/ensemble_union.py --min-votes 1 --out $O/s30.jsonl $O/B0.jsonl $O/A1.jsonl $O/E1.jsonl
uv run python scripts/validate_submission.py --sub $O/s30.jsonl --ref "$REF" 2>&1 | tail -3 || true
uv run python scripts/make_submission_zip.py --sub $O/s30.jsonl --ref "$REF" --out submissions/testp1_s30_u_cleanunion.zip
cp $O/s30.jsonl submissions/testp1_s30_u_cleanunion.jsonl
echo "=== minority count in s30 ==="
python3 -c "import json,collections;d=collections.Counter();[d.update(json.loads(l)['labels']) for l in open('$O/s30.jsonl')];m=sum(v for k,v in d.items() if k!='Supported');print('minorities',m,dict(d))"
touch $O/PACKAGE_DONE
echo "=== PACKAGE DONE -> submissions/testp1_s30_u_cleanunion.zip ==="
