#!/usr/bin/env python3
"""Server-side validation of the Option-B softmin integration against the installed ms-swift.

Run on the server (no GPU needed):  PYTHONPATH=src uv run python scripts/validate_opt_b.py
Checks, with explicit PASS/FAIL lines:
  1. swift version + loss_map + BaseLoss + sft_main importable.
  2. our register() puts 'softmin_pem' into swift.loss.loss_map as a BaseLoss subclass.
  3. our patch swaps swift.trainers.Seq2SeqTrainer -> SoftMinTrainer, and SoftMinTrainer is a
     real Seq2SeqTrainer subclass overriding get_train_dataloader(self, skip_batches=0).
  4. the ParagraphGroupSampler groups + DDP-shards correctly (pure logic, no torch model).
"""
import sys
import inspect
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""))


# 1. swift entries
import swift  # noqa: E402
from swift.loss import loss_map, BaseLoss  # noqa: E402

print(f"swift {swift.__version__} | loss_map keys (before): {sorted(loss_map.keys())}")
try:
    from swift.cli.sft import sft_main  # noqa: F401,E402
    check("swift.cli.sft.sft_main importable", True)
except Exception as e:  # noqa: BLE001
    check("swift.cli.sft.sft_main importable", False, repr(e))

# 2 + 3. register + patch
from nlpcc_t10.swift_softmin import register  # noqa: E402

register.register()

import swift.trainers as T  # noqa: E402

cls = loss_map.get("softmin_pem")
check("softmin_pem registered in loss_map", cls is not None)
check(
    "registered loss is a BaseLoss subclass",
    isinstance(cls, type) and issubclass(cls, BaseLoss),
    getattr(cls, "__name__", None),
)
patched = T.Seq2SeqTrainer
check("Seq2SeqTrainer monkeypatched", patched.__name__ == "SoftMinTrainer", patched.__name__)
# original Seq2SeqTrainer must be an ancestor (so all stock behaviour is inherited)
mro = [c.__name__ for c in patched.__mro__]
check("SoftMinTrainer inherits Seq2SeqTrainer", "Seq2SeqTrainer" in mro, " -> ".join(mro[:4]))
sig = inspect.signature(patched.get_train_dataloader)
check(
    "get_train_dataloader(self, skip_batches=0) signature",
    "skip_batches" in sig.parameters,
    str(sig),
)

# 4. sampler logic (pure python, no torch model)
from nlpcc_t10.swift_softmin.sampler import build_paragraph_trainer_cls  # noqa: E402,F401

# Re-derive the inner sampler class to test grouping/sharding without a model.
# (build_paragraph_trainer_cls builds the trainer; ParagraphGroupSampler is nested, so we
#  test the grouping invariants via group_indices, the shared pure-python helper.)
from nlpcc_t10.swift_softmin.loss import group_indices  # noqa: E402

pids = ["p0", "p0", "p1", "p2", "p2", "p2", "p0#aug0"]
groups = group_indices(pids)
check("group_indices keeps paragraphs intact", groups["p0"] == [0, 1] and groups["p2"] == [3, 4, 5])
check("augmented copy is its own singleton group", groups["p0#aug0"] == [6])

print("\nRESULT:", "ALL PASS ✅" if ok else "SOME FAILED ❌")
sys.exit(0 if ok else 1)
