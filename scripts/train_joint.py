#!/usr/bin/env python3
"""Plain-CE LoRA SFT launcher for PARAGRAPH-JOINT data.

Joint samples are INDEPENDENT per-record rows whose target is a JSON array of N labels — standard
ms-swift SFT with standard cross-entropy is exactly right. This differs from train_softmin.py by
NOT registering the softmin-PEM loss or the paragraph-group sampler (those are per-sentence-only).
Run under torchrun for DDP. All hyperparameters come from the CLI (see scripts/train_joint.sh)."""
import os

# Must be set before torch CUDA init in each torchrun worker (env vars don't reach workers).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from swift.pipelines import sft_main  # noqa: E402

if __name__ == "__main__":
    sft_main()
