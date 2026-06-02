"""SoftMin (PEM-bottleneck) loss package for ms-swift 4.2.3.

Modules:
  * ``loss``     -- ``softmin_pem_loss`` + ``group_indices`` (pure-python core, torch deferred).
  * ``sampler``  -- ``ParagraphGroupSampler`` + ``build_paragraph_trainer_cls`` (one batch == one
                    paragraph; rank-sharded for DDP).
  * ``register`` -- ms-swift ``--custom_register_path`` entrypoint (registers ``softmin_pem`` and
                    monkeypatches the trainer).

Design rationale + numerical proofs: ``notes/softmin_loss_design.md``.
Standalone torch test (run on the server): ``test_softmin_loss.py`` at the repo root.
"""

from __future__ import annotations

from .loss import group_indices, softmin_pem_loss
from .sampler import build_paragraph_trainer_cls

__all__ = ["group_indices", "softmin_pem_loss", "build_paragraph_trainer_cls"]
