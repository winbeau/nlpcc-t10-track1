#!/usr/bin/env bash
# Submit the clean PER-SENTENCE singles on testp1: A0 softmin -> s31, A1 plain-CE -> s32.
# Purpose: (a) the ONLY clean same-condition softmin-vs-plain-CE head-to-head (both train', per-sentence,
# no oversample); (b) A1 alone isolates whether the joint B0 member dragged s30's union (if A1 > s30 43.85,
# the union HURT). Both trained on image-disjoint train' (90% data) -> expect a train'-only handicap vs
# the full-traindev s01 (47.80). GPU3=A0, GPU4=A1, parallel.
set -uo pipefail
cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
export DATA_ROOT=/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science
export MODELSCOPE_CACHE=/data/chenjiayu/wenbiao_zhao/ms_cache USE_HF=0 PYTHONPATH=src TOKENIZERS_PARALLELISM=false
M=Qwen/Qwen3-VL-8B-Instruct; REF="$DATA_ROOT/data/testp1-track-1.jsonl"
O=outputs/phaseC/package_singles; mkdir -p $O
A0=$(ls -d outputs/phaseC/A0_softmin_clean/ckpt/v*/checkpoint-* | sort | tail -1)
A1=$(ls -d outputs/phaseC/A1_plainCE/ckpt/v*/checkpoint-* | sort | tail -1)
echo "A0(softmin)=$A0  A1(plainCE)=$A1"

inf(){ CUDA_VISIBLE_DEVICES=$1 MAX_PIXELS=401408 uv run python -m nlpcc_t10.infer --split testp1 \
        --model $M --adapter $2 --engine pt --data-root "$DATA_ROOT" --out $3; }
agg(){ uv run python -m nlpcc_t10.aggregate --pred $1 --ref "$REF" --out $2; }

( inf 3 "$A0" $O/A0_raw.jsonl && agg $O/A0_raw.jsonl submissions/testp1_s31_m_clean-softmin.jsonl && echo "A0/s31 infer done" ) &
( inf 4 "$A1" $O/A1_raw.jsonl && agg $O/A1_raw.jsonl submissions/testp1_s32_m_clean-plainCE.jsonl && echo "A1/s32 infer done" ) &
wait
for s in s31_m_clean-softmin s32_m_clean-plainCE; do
  uv run python scripts/validate_submission.py --sub submissions/testp1_$s.jsonl --ref "$REF" 2>&1 | tail -1
  uv run python scripts/make_submission_zip.py --sub submissions/testp1_$s.jsonl --ref "$REF" --out submissions/testp1_$s.zip
  python3 -c "import json,collections;d=collections.Counter();[d.update(json.loads(l)['labels']) for l in open('submissions/testp1_$s.jsonl')];m=sum(v for k,v in d.items() if k!='Supported');print('$s minorities=',m,{k:v for k,v in d.items() if k!='Supported'})"
done
touch $O/DONE
echo "=== SINGLES PACKAGE DONE: submissions/testp1_s31_m_clean-softmin.zip + testp1_s32_m_clean-plainCE.zip ==="
