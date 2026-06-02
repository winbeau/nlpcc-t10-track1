#!/usr/bin/env python3
"""Launcher for the SoftMin (PEM-bottleneck) training run.

WHY a launcher (not plain `swift sft`): the softmin loss needs each optimizer batch to be
ONE complete paragraph, which requires our custom ParagraphGroupSampler inside a
Seq2SeqTrainer subclass. ms-swift 4.2.3 builds a vanilla `Seq2SeqTrainer` internally, so we
replace it with `SoftMinTrainer` BEFORE the swift CLI constructs the trainer. The loss
itself is registered via --loss_type softmin_pem + --custom_register_path (register.py's
top-level code runs on import); paragraph ids reach the loss through the `channel` field
(build_dataset.py contract), which the stock compute_loss already forwards.

Implementation lives in the package src/nlpcc_t10/swift_softmin/ :
  loss.py     -- softmin_pem_loss + group_indices
  sampler.py  -- ParagraphGroupSampler + build_paragraph_trainer_cls
  register.py -- the --custom_register_path entrypoint (registers + patches the trainer)

CONFIRM on the server: whether 4.2.3 exposes a cleaner `--custom_trainer` hook; if so prefer
it over this monkeypatch. Also re-verify swift.__version__ and that
Seq2SeqTrainer.get_train_dataloader / DataLoaderShard signatures match the plugin.

REQUIRED config for this run (see notes/softmin_loss_design.md §6):
  per_device_train_batch_size: 1   (sampler controls the real batch size)
  packing: false  AND  padding_free: false
      packing concatenates sentences into one row (per-sentence L_i unrecoverable) and drops
      `channel`; padding_free flattens rows. Correctness > throughput.

Usage (2xL40):
  SOFTMIN_BETA=5.0 SOFTMIN_LAMBDA=0.5 \
  IMAGE_MAX_TOKEN_NUM=1024 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 \
      uv run python scripts/train_softmin.py --config configs/qwen3vl_lora_sft.yaml \
          --loss_type softmin_pem \
          --custom_register_path src/nlpcc_t10/swift_softmin/register.py \
          --per_device_train_batch_size 1 --packing false --padding_free false

For the MANDATORY ablation baseline (plain CE), just run scripts/train.sh (or set
SOFTMIN_LAMBDA=0) and compare dev SCORE via src/nlpcc_t10/eval_local.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# make `import nlpcc_t10...` work whether or not the package is pip-installed
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    from nlpcc_t10.swift_softmin import register as reg

    # 1+2) Register the loss AND monkeypatch Seq2SeqTrainer -> SoftMinTrainer everywhere
    #       ms-swift references it, BEFORE the CLI constructs the trainer. register.register()
    #       does both (register_loss + patch_trainer, idempotent). It builds the trainer
    #       subclass lazily (imports torch/swift here, on the server).
    reg.register()

    # 3) Hand off to the swift sft entrypoint with the original argv. In ms-swift 4.2.3
    #    sft_main lives in swift.pipelines (swift/cli/sft.py itself does
    #    `from swift.pipelines import sft_main`). For MULTI-GPU, launch this script under
    #    torchrun so register() runs in EVERY worker process, e.g.:
    #      NPROC_PER_NODE=2 torchrun --nproc_per_node=2 scripts/train_softmin.py --config ...
    from swift.pipelines import sft_main
    sft_main()


if __name__ == "__main__":
    main()
