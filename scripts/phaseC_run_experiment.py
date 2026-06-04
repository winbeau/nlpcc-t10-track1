#!/usr/bin/env python3
"""PHASE C iron-law harness — run ONE experiment end-to-end, server-side.

Reads its config from notes/phaseC/queue.jsonl by name, then:
  train(optional) -> infer devbench DEV -> aggregate -> eval on dev_gold_densematch -> ledger.csv
and writes outputs/phaseC/<name>/{DONE|FAILED} as its last act (the loop's completion signal).

Pure-stdlib (json/os/subprocess); shells out to `uv run` for the heavy ms-swift steps. Launched
detached by the loop:  nohup env PHASEC_GPU=<g> python3 scripts/phaseC_run_experiment.py <name> \
                            > outputs/phaseC/<name>/run.log 2>&1 &

IRON LAW (notes/exploration_roadmap.md §1):
  - train ONLY on data/devbench/train_sft.jsonl (image-disjoint train'; asserted == TRAIN_LINES).
  - cp data/devbench/split.json -> data/split.json BEFORE dev inference (so --split dev = devbench).
  - eval on data/devbench/dev_gold_densematch.jsonl (the GATE-B judge).
  - USE_HF=0 so ms-swift loads cached ModelScope bases (the box's global USE_HF=1 misroutes to HF).

Config schema (one queue.jsonl line); see notes/phaseC/queue.jsonl:
  {name, kind: train|infer_only|soup, tag,
   model, max_pixels, joint(bool), infer_extra(str), union_samples(int, E1),
   # train:
   train_mode: softmin|plaince, beta, lambda,
   # infer_only:
   adapter: "<path>" | "@<expName>"  (resolve to that experiment's checkpoint),
   # soup:
   members: ["<ckpt path>", ...], soup_weights(optional)}
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN_DATA = "data/devbench/train_sft.jsonl"
TRAIN_LINES = 15008  # image-disjoint train' (assert — never train on the leaky data/train_sft.jsonl)
DEV_GOLD = "data/devbench/dev_gold.jsonl"
DENSEMATCH = "data/devbench/dev_gold_densematch.jsonl"
LEDGER = "notes/phaseC/ledger.csv"
DATA_ROOT = os.environ.get("DATA_ROOT", "/data/chenjiayu/wenbiao_zhao/NLPCC-2026-Task10-Science")
MS_CACHE = os.environ.get("MODELSCOPE_CACHE", "/data/chenjiayu/wenbiao_zhao/ms_cache")


def log(msg: str) -> None:
    print(f"[phaseC:{NAME}] {msg}", flush=True)


def base_env() -> dict:
    e = dict(os.environ)
    e.update(USE_HF="0", DATA_ROOT=DATA_ROOT, MODELSCOPE_CACHE=MS_CACHE,
             PYTHONPATH="src", TOKENIZERS_PARALLELISM="false",
             PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return e


def run(cmd: list[str], env: dict | None = None) -> None:
    """Run a subprocess, streaming output; raise on non-zero."""
    log("RUN " + " ".join(cmd))
    r = subprocess.run(cmd, env=env or base_env(), cwd=str(REPO))
    if r.returncode != 0:
        raise RuntimeError(f"step failed (rc={r.returncode}): {' '.join(cmd[:6])}...")


def load_cfg(name: str) -> dict:
    for line in (REPO / "notes/phaseC/queue.jsonl").read_text().splitlines():
        line = line.strip()
        if line and json.loads(line).get("name") == name:
            return json.loads(line)
    raise SystemExit(f"experiment {name!r} not in queue.jsonl")


def resolve_adapter(spec: str) -> str:
    """A literal checkpoint path, or '@<expName>' -> that experiment's latest checkpoint."""
    if spec.startswith("@"):
        exp = spec[1:]
        cks = sorted(glob.glob(f"outputs/phaseC/{exp}/ckpt/v*/checkpoint-*"))
        if not cks:
            raise RuntimeError(f"no checkpoint for dep experiment {exp} (outputs/phaseC/{exp}/ckpt)")
        return cks[-1]
    return spec


# ──────────────────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────────────────

def do_train(cfg: dict, out: Path, gpu: str) -> str:
    ckdir = out / "ckpt"
    existing = sorted(glob.glob(str(ckdir / "v*/checkpoint-*")))
    if existing:  # idempotent: a re-launch after an infer/eval failure must NOT retrain
        log(f"checkpoint already exists, skipping training: {existing[-1]}")
        return existing[-1]
    n = sum(1 for _ in open(REPO / TRAIN_DATA))
    assert n == TRAIN_LINES, f"REFUSING TO TRAIN: {TRAIN_DATA} has {n} lines, expected {TRAIN_LINES} " \
                             f"(image-disjoint train'). Are you about to train on the leaky split?"
    mode = cfg.get("train_mode", "softmin")
    beta = str(cfg.get("beta", 5))
    lam = str(cfg.get("lambda", 0.5 if mode == "softmin" else 0.0))
    mp = str(cfg.get("max_pixels", 401408))
    model = cfg.get("model", "Qwen/Qwen3-VL-8B-Instruct")
    env = base_env()
    env.update(CUDA_VISIBLE_DEVICES=gpu, SOFTMIN_BETA=beta, SOFTMIN_LAMBDA=lam)
    port = str(29500 + (int(gpu.split(",")[0]) if gpu else 0))
    # per-sentence softmin path; LAMBDA=0 => plain mean-CE in the SAME format (loss.py:217). Grid
    # recipe (use_logits_to_keep=false, 401408, 1 epoch, lr 1e-4) — matches the s01 champion single.
    cmd = ["uv", "run", "torchrun", "--nproc_per_node=1", f"--master_port={port}",
           "scripts/train_softmin.py",
           "--model", model, "--tuner_type", "lora", "--torch_dtype", "bfloat16",
           "--dataset", TRAIN_DATA, "--split_dataset_ratio", "0",
           "--loss_type", "softmin_pem",
           "--num_train_epochs", str(cfg.get("epochs", 1)),
           "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
           "--learning_rate", "1e-4", "--lora_rank", "16", "--lora_alpha", "32",
           "--freeze_vit", "true", "--max_length", "10240", "--max_pixels", mp,
           "--attn_impl", "sdpa", "--packing", "false", "--padding_free", "false",
           "--use_logits_to_keep", "false",
           "--eval_strategy", "no", "--save_strategy", "epoch", "--save_total_limit", "1",
           "--logging_steps", "20", "--dataloader_num_workers", "0",
           "--output_dir", str(ckdir)]
    log(f"TRAIN mode={mode} beta={beta} lambda={lam} max_pixels={mp} gpu={gpu} port={port}")
    run(cmd, env)
    cks = sorted(glob.glob(str(ckdir / "v*/checkpoint-*")))
    if not cks:
        raise RuntimeError(f"no checkpoint produced in {ckdir}")
    log(f"trained -> {cks[-1]}")
    return cks[-1]


def do_soup(cfg: dict, out: Path) -> str:
    members = cfg["members"]
    soup_dir = out / "soup_adapter"
    cmd = ["uv", "run", "python", "scripts/soup_adapters.py",
           "--out", str(soup_dir), "--adapters", *members]
    if cfg.get("soup_weights"):
        cmd += ["--weights", cfg["soup_weights"]]
    run(cmd)
    return str(soup_dir)


def do_infer_eval(cfg: dict, out: Path, gpu: str, adapter: str) -> None:
    run(["cp", "data/devbench/split.json", "data/split.json"])  # devbench dev split
    model = cfg.get("model", "Qwen/Qwen3-VL-8B-Instruct")
    mp = str(cfg.get("max_pixels", 401408))
    joint = bool(cfg.get("joint", False))
    n_samp = int(cfg.get("union_samples", 1))
    env = base_env()
    env.update(CUDA_VISIBLE_DEVICES=gpu, MAX_PIXELS=mp)
    extra = cfg.get("infer_extra", "").split() if cfg.get("infer_extra") else []
    base_infer = ["uv", "run", "python", "-m", "nlpcc_t10.infer", "--split", "dev",
                  "--model", model, "--adapter", adapter, "--engine", "pt",
                  "--data-root", DATA_ROOT] + (["--joint"] if joint else [])

    def aggregate(rawp: Path, predp: Path) -> None:
        run(["uv", "run", "python", "-m", "nlpcc_t10.aggregate", "--pred", str(rawp),
             "--ref", DEV_GOLD, "--out", str(predp)])

    pred = out / "dev_pred.jsonl"
    if pred.exists() and pred.stat().st_size > 0:
        log(f"reusing existing {pred.name} (skip infer)")     # idempotent: re-eval without re-infer
    elif n_samp > 1:
        # E1 self-consistency: run infer K times (temperature from infer_extra, fresh --seed k),
        # aggregate each, union (min-votes 1) = additive recall. No reliance on engine n>1.
        ks = []
        for k in range(n_samp):
            kraw, kpred = out / f"dev_raw.k{k}.jsonl", out / f"dev_pred.k{k}.jsonl"
            run(base_infer + ["--out", str(kraw), "--seed", str(k)] + extra, env)
            aggregate(kraw, kpred)
            ks.append(str(kpred))
        run(["uv", "run", "python", "scripts/ensemble_union.py", "--min-votes", "1",
             "--out", str(pred), *ks])
    else:
        raw = out / "dev_raw.jsonl"
        run(base_infer + ["--out", str(raw)] + extra, env)
        aggregate(raw, pred)

    # evaluate.py requires identical gold/pred id sets -> subset the 501-rec dev pred to the
    # 206 densematch ids before scoring (the dev pred legitimately covers more records).
    dm_ids = {json.loads(l)["id"] for l in open(REPO / DENSEMATCH) if l.strip()}
    pred_dm = out / "dev_pred_densematch.jsonl"
    with open(pred) as f, open(pred_dm, "w") as g:
        for line in f:
            if line.strip() and json.loads(line)["id"] in dm_ids:
                g.write(line)

    tag = cfg.get("tag", NAME)
    run(["uv", "run", "python", "-m", "nlpcc_t10.eval_local", "--pred", str(pred_dm),
         "--gold", DENSEMATCH, "--data-root", DATA_ROOT, "--tag", tag,
         "--log", LEDGER, "--keep-result"], env)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(name: str) -> int:
    cfg = load_cfg(name)
    out = REPO / "outputs/phaseC" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    (out / "FAILED").unlink(missing_ok=True)
    gpu = os.environ.get("PHASEC_GPU", "0")
    try:
        kind = cfg.get("kind", "infer_only")
        if kind == "train":
            adapter = do_train(cfg, out, gpu)
        elif kind == "soup":
            adapter = do_soup(cfg, out)
        else:
            adapter = resolve_adapter(cfg["adapter"])
        do_infer_eval(cfg, out, gpu, adapter)
        (out / "DONE").write_text("ok\n")
        log("DONE")
        return 0
    except Exception as exc:  # noqa: BLE001 — last-resort marker
        (out / "FAILED").write_text(f"{type(exc).__name__}: {exc}\n")
        log(f"FAILED: {exc}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: phaseC_run_experiment.py <experiment-name>")
    NAME = sys.argv[1]
    raise SystemExit(main(NAME))
else:
    NAME = "?"
