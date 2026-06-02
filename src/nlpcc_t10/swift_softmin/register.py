"""ms-swift 4.2.3 registration entrypoint for the SoftMin-PEM loss (Option B).

Two things must happen before ms-swift builds the trainer:
  1. register_loss(): put the loss CLASS into ``swift.loss.loss_map['softmin_pem']`` so that
     ``--loss_type softmin_pem`` resolves to it (``mixin.py:996``:
     ``res['compute_loss_func'] = loss_map[args.loss_type](args, self)``).
  2. patch_trainer(): monkeypatch ``swift.trainers.Seq2SeqTrainer`` -> ``SoftMinTrainer`` so the
     train dataloader yields one paragraph per batch. ``trainer_factory`` resolves
     ``'causal_lm' -> 'swift.trainers.Seq2SeqTrainer'`` via ``importlib`` + ``getattr``, so
     overwriting the module attribute BEFORE the factory runs takes effect.

Use the dedicated launcher ``scripts/train_softmin.py`` (it calls ``register()`` then
``swift.cli.sft.sft_main()``) so ordering relative to the CLI is guaranteed. Passing this file
via ``--custom_register_path`` also works because its top-level code runs ``register()`` on
import -- but the launcher is the recommended, order-safe path.

Importable WITHOUT torch/swift (local unit tests): both steps are wrapped in try/except, so
importing this module locally is a no-op that does not raise.
"""

from __future__ import annotations

LOSS_NAME = "softmin_pem"

_registered = False
_patched = False


def register_loss():
    """Put the SoftMin-PEM loss class into ``swift.loss.loss_map`` (idempotent)."""
    global _registered
    if _registered:
        return True
    from swift.loss import loss_map

    from .loss import make_softmin_loss_cls

    loss_map[LOSS_NAME] = make_softmin_loss_cls()  # value is a BaseLoss subclass (a CLASS)
    _registered = True
    return True


def patch_trainer():
    """Monkeypatch ``swift.trainers.Seq2SeqTrainer`` with ``SoftMinTrainer`` (idempotent).

    Builds the subclass FIRST (capturing the original Seq2SeqTrainer to subclass from), THEN
    overwrites the module attribute(s) ms-swift resolves at trainer-build time.
    """
    global _patched
    if _patched:
        return True
    from .sampler import build_paragraph_trainer_cls

    soft_min_trainer = build_paragraph_trainer_cls()  # imports + subclasses the ORIGINAL
    import swift.trainers as swift_trainers

    swift_trainers.Seq2SeqTrainer = soft_min_trainer
    try:  # also patch the defining submodule, in case it is referenced directly
        import swift.trainers.seq2seq_trainer as _seq2seq

        _seq2seq.Seq2SeqTrainer = soft_min_trainer
    except Exception:  # noqa: BLE001
        pass
    _patched = True
    return True


def register():
    """Register the loss AND patch the trainer. Safe to call repeatedly."""
    register_loss()
    patch_trainer()


# --custom_register_path imports this file and runs top-level code, so register on import.
# Wrapped so the module is importable WITHOUT swift (local unit tests), where it is a no-op.
try:  # pragma: no cover - exercised only on the server
    register()
except Exception:  # noqa: BLE001
    pass
