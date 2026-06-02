#!/usr/bin/env python3
"""Coordinate β/λ grid search for the softmin-PEM LoRA run (H200, 2-GPU DDP).

Phase A: λ=0.5, β∈{2,5,10}  -> pick best β by official dev SCORE.
Phase B: best β, λ∈{0.3,0.7}.
Each config: train 1 epoch (2 GPU) -> infer FULL dev (1 GPU) -> aggregate -> official eval.

Per-config rows append to outputs/grid_results.csv (via eval_local --log); a final
outputs/grid_summary.json + printed ranking name the best (β,λ).

Launch on H200 (long, ~4-5h):
  cd /data/chenjiayu/wenbiao_zhao/nlpcc-t10-track1
  PYTHONPATH=src nohup uv run python scripts/grid_search.py > /tmp/grid.log 2>&1 &

Env (overridable): GRID_GPUS_TRAIN (default "0,1"), GRID_GPU_INFER (default "0"),
GRID_EPOCHS (default 1), MODEL_ID, DATA_ROOT, MODELSCOPE_CACHE.
Idempotent-ish: a config whose checkpoint+score already exist is skipped.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("DATA_ROOT", str(REPO.parent / "NLPCC-2026-Task10-Science"))
MS_CACHE = os.environ.get("MODELSCOPE_CACHE", str(REPO.parent / "ms_cache"))
MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
GPUS_TRAIN = os.environ.get("GRID_GPUS_TRAIN", "0,1")
GPU_INFER = os.environ.get("GRID_GPU_INFER", "0")
EPOCHS = os.environ.get("GRID_EPOCHS", "1")
NPROC = str(len(GPUS_TRAIN.split(",")))
RESULTS = REPO / "outputs" / "grid_results.csv"
os.chdir(REPO)
RESULTS.parent.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], env_extra: dict, log_path: str) -> int:
    env = {**os.environ, "PYTHONPATH": "src", "MODELSCOPE_CACHE": MS_CACHE, **env_extra}
    with open(log_path, "w") as f:
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def train(beta, lam, outdir, logp) -> tuple[int, str | None]:
    cmd = [
        "uv", "run", "torchrun", f"--nproc_per_node={NPROC}", "--master_port=29531",
        "scripts/train_softmin.py",
        "--model", MODEL, "--tuner_type", "lora", "--torch_dtype", "bfloat16",
        "--dataset", "data/train_sft.jsonl", "--split_dataset_ratio", "0",
        "--loss_type", "softmin_pem",
        "--num_train_epochs", EPOCHS, "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1", "--learning_rate", "1e-4",
        "--lora_rank", "16", "--lora_alpha", "32", "--freeze_vit", "true",
        "--max_length", "4096", "--max_pixels", "401408", "--attn_impl", "sdpa",
        "--packing", "false", "--padding_free", "false", "--use_logits_to_keep", "false",
        "--eval_strategy", "no", "--save_strategy", "epoch", "--save_total_limit", "1",
        "--logging_steps", "20", "--dataloader_num_workers", "0", "--output_dir", outdir,
    ]
    rc = _run(cmd, {"CUDA_VISIBLE_DEVICES": GPUS_TRAIN,
                    "SOFTMIN_BETA": str(beta), "SOFTMIN_LAMBDA": str(lam)}, logp)
    ckpts = sorted(Path(outdir).glob("v*/checkpoint-*"))
    return rc, (str(ckpts[-1]) if ckpts else None)


def infer_eval(ckpt, tag) -> float | None:
    raw, sub = f"/tmp/{tag}_raw.jsonl", f"/tmp/{tag}_sub.jsonl"
    if _run(["uv", "run", "python", "-m", "nlpcc_t10.infer", "--split", "dev", "--adapter", ckpt,
             "--engine", "pt", "--data-root", DATA_ROOT, "--out", raw],
            {"CUDA_VISIBLE_DEVICES": GPU_INFER, "MAX_PIXELS": "401408"}, f"/tmp/{tag}.infer.log"):
        print(f"  [{tag}] infer FAILED (see /tmp/{tag}.infer.log)", flush=True)
        return None
    _run(["uv", "run", "python", "-m", "nlpcc_t10.aggregate", "--pred", raw,
          "--ref", "data/dev_gold.jsonl", "--out", sub], {}, f"/tmp/{tag}.agg.log")
    elog = f"/tmp/{tag}.eval.log"
    _run(["uv", "run", "python", "-m", "nlpcc_t10.eval_local", "--pred", sub, "--gold", "data/dev_gold.jsonl",
          "--data-root", DATA_ROOT, "--tag", tag, "--log", str(RESULTS)], {}, elog)
    txt = Path(elog).read_text(errors="ignore")
    m = re.search(r"SCORE=([0-9.]+)", txt) or re.search(r"score\s*[:=]\s*([0-9.]+)", txt)
    return float(m.group(1)) if m else None


def do_config(beta, lam) -> float | None:
    tag = f"b{beta}_l{lam}"
    outdir = f"outputs/grid_{tag}"
    print(f"\n===================== CONFIG {tag} =====================", flush=True)
    rc, ckpt = train(beta, lam, outdir, f"/tmp/grid_{tag}.train.log")
    if rc != 0 or not ckpt:
        print(f"  TRAIN FAILED rc={rc} ckpt={ckpt} (see /tmp/grid_{tag}.train.log)", flush=True)
        return None
    print(f"  trained -> {ckpt}; infer full dev + eval ...", flush=True)
    score = infer_eval(ckpt, f"grid_{tag}")
    print(f"  >>> {tag} dev SCORE = {score}", flush=True)
    return score


def main() -> int:
    print(f"GRID start | model={MODEL} | train GPUs={GPUS_TRAIN} | infer GPU={GPU_INFER} | epochs={EPOCHS}", flush=True)
    print("######## PHASE A: lambda=0.5, beta in {2,5,10} ########", flush=True)
    a = {b: do_config(b, 0.5) for b in (2, 5, 10)}
    valid = {b: s for b, s in a.items() if s is not None}
    best_beta = max(valid, key=valid.get) if valid else 5
    print(f"\n######## PHASE A done: {a} | best_beta={best_beta} ########", flush=True)

    print(f"######## PHASE B: beta={best_beta}, lambda in {{0.3,0.7}} ########", flush=True)
    b_scores = {lam: do_config(best_beta, lam) for lam in (0.3, 0.7)}

    allres = {f"b{bb}_l0.5": a[bb] for bb in a}
    allres.update({f"b{best_beta}_l{lam}": b_scores[lam] for lam in b_scores})
    ranked = sorted(allres.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0.0)))
    print("\n######## GRID DONE — ranking by dev SCORE ########", flush=True)
    for k, v in ranked:
        print(f"  {k}: {v}", flush=True)
    best = ranked[0][0] if ranked and ranked[0][1] is not None else None
    print(f"BEST CONFIG: {best} SCORE={allres.get(best)}", flush=True)
    (REPO / "outputs" / "grid_summary.json").write_text(
        json.dumps({"results": allres, "best": best, "best_beta": best_beta}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
