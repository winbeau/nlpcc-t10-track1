"""ms-swift registration entrypoint for the SoftMin PEM loss.

Pass this file to ``swift sft`` via ``--custom_register_path`` so that ms-swift imports it
and runs its top-level code, which registers the loss under the name ``softmin_pem`` in
``swift.plugin.loss.LOSS_MAPPING``. Then select it with ``--loss_type softmin_pem``.

  swift sft ... \
      --loss_type softmin_pem \
      --custom_register_path src/nlpcc_t10/swift_softmin/register.py

The loss alone, however, is NOT enough: the SoftMin objective requires each optimizer batch
to be ONE complete paragraph. ms-swift constructs a vanilla ``Seq2SeqTrainer``, so the
paragraph-group sampler is installed by monkeypatching ``Seq2SeqTrainer`` with
``SoftMinTrainer`` (from ``sampler.build_paragraph_trainer_cls``) BEFORE the CLI builds the
trainer -- see ``scripts/train_softmin.py``. ``register()`` here also calls
``patch_trainer()`` opportunistically so that, even when this file is imported standalone (as
``--custom_register_path`` does), the trainer is patched as a side effect. The dedicated
launcher remains the recommended path because it guarantees ordering relative to the CLI.

Importable without torch/swift (local unit tests): both ``register_loss`` and
``patch_trainer`` are wrapped in try/except, so importing this module locally is a no-op that
does not raise.
"""

from __future__ import annotations

# Import the loss fn from the sibling module. ``loss`` is torch-free at import time
# (torch is deferred inside its functions), so this import is safe locally.
from .loss import softmin_pem_loss
from .sampler import build_paragraph_trainer_cls

LOSS_NAME = "softmin_pem"

_registered = False
_patched = False


def register_loss():
    """Register ``softmin_pem_loss`` in ms-swift's ``LOSS_MAPPING`` (idempotent)."""
    global _registered
    if _registered:
        return True
    from swift.plugin.loss import register_loss_func

    register_loss_func(LOSS_NAME)(softmin_pem_loss)
    _registered = True
    return True


def patch_trainer():
    """Monkeypatch ``swift.trainers.Seq2SeqTrainer`` (and its re-export) with
    ``SoftMinTrainer`` so the paragraph-group sampler is used. Idempotent.

    NOTE: for reliable ordering relative to the swift CLI, prefer the dedicated launcher
    ``scripts/train_softmin.py`` which patches BEFORE constructing the trainer. This function
    exists so ``--custom_register_path`` alone also patches when imported early enough.
    """
    global _patched
    if _patched:
        return True
    SoftMinTrainer = build_paragraph_trainer_cls()
    import swift.trainers as swift_trainers

    swift_trainers.Seq2SeqTrainer = SoftMinTrainer
    try:  # some versions re-export from swift.trainers.trainers
        import swift.trainers.trainers as _t

        _t.Seq2SeqTrainer = SoftMinTrainer
    except Exception:
        pass
    _patched = True
    return True


def register():
    """Register the loss AND patch the trainer. Safe to call repeatedly."""
    register_loss()
    patch_trainer()


# ms-swift imports this file (via --custom_register_path) and runs module top-level code,
# so register on import. Wrapped in try/except so the file is importable WITHOUT swift
# (e.g. local unit tests of loss.group_indices), where it is a harmless no-op.
try:  # pragma: no cover - exercised only on the server
    register_loss()
except Exception:
    pass
try:  # pragma: no cover - exercised only on the server
    patch_trainer()
except Exception:
    pass
