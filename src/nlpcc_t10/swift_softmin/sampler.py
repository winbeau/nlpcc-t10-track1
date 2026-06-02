"""Paragraph-group batch sampler + SoftMinTrainer for ms-swift 4.2.3.

The SoftMin PEM loss (see ``loss.py``) needs each optimizer batch to contain ALL the
sentence-samples of ONE paragraph, intact, so the per-paragraph soft-min over per-sentence
losses can be computed. ms-swift's stock sampler batches by global index and would split a
paragraph across batches / ranks. This module provides:

  * ``ParagraphGroupSampler`` -- a ``batch_sampler`` that yields one variable-length batch
    per paragraph (round-robin sharded by rank for DDP, so each GPU always sees COMPLETE
    paragraphs).
  * ``build_paragraph_trainer_cls()`` -- a lazy factory returning a ``Seq2SeqTrainer``
    subclass (``SoftMinTrainer``) that only overrides ``get_train_dataloader`` to install
    the sampler. NO ``compute_loss`` override is needed: paragraph ids ride the ``channel``
    field (build_dataset.py contract), which the STOCK ``compute_loss`` already pops and
    forwards to the loss as ``sample_channels`` (verified: trainers.py:168-175,219).

DDP correctness (2xL40): Accelerate does NOT auto-shard a custom ``batch_sampler``, so the
sampler shards WHOLE paragraphs round-robin (``order[rank::world_size]``). Never split a
paragraph across ranks -- a per-rank partial-paragraph LSE is the WRONG bottleneck and fails
silently. Drop-tail keeps every rank's step count equal (else DDP hangs at grad all-reduce).

torch / swift imports are deferred into ``build_paragraph_trainer_cls`` so this module is
importable locally (no torch) for documentation / introspection.
"""

from __future__ import annotations

from collections import defaultdict


def build_paragraph_trainer_cls():
    """Return a ``Seq2SeqTrainer`` subclass whose train dataloader batches one paragraph at a
    time. Call this on the server (inside the train entrypoint), AFTER swift is importable.

    The returned ``SoftMinTrainer`` is monkeypatched in for the stock ``Seq2SeqTrainer`` by
    ``scripts/train_softmin.py`` before the swift CLI constructs the trainer.
    """
    from functools import partial

    import torch.distributed as dist  # noqa: F401  (kept for parity / future use)
    from torch.utils.data import Sampler
    from swift.trainers import Seq2SeqTrainer
    from swift.llm import DataLoaderShard

    class ParagraphGroupSampler(Sampler):
        """A ``batch_sampler``: each ``__iter__`` step yields a LIST of dataset indices that
        together form ONE paragraph (variable length n_p).

        Args:
            paragraph_ids: list aligned with the train dataset; ``paragraph_ids[i]`` is the
                ``channel`` (== paragraph id) of dataset row ``i``.
            shuffle: shuffle the ORDER of paragraphs each epoch (sentences within a paragraph
                stay together; their intra-paragraph order is irrelevant to the soft-min).
            seed: base RNG seed; per-epoch seed is ``seed + epoch`` for reproducibility.
            rank, world_size: DDP sharding -- this rank takes ``order[rank::world_size]``.
        """

        def __init__(self, paragraph_ids, shuffle=True, seed=42, rank=0, world_size=1):
            groups = defaultdict(list)
            for idx, pid in enumerate(paragraph_ids):
                groups[pid].append(idx)
            # one entry per paragraph, each a list of that paragraph's row indices
            self.all_batches = list(groups.values())
            self.shuffle = shuffle
            self.seed = int(seed)
            self.rank = int(rank)
            self.world_size = max(1, int(world_size))
            self.epoch = 0

        def _epoch_order(self):
            """Indices into ``self.all_batches`` for THIS rank this epoch (drop-tail)."""
            import random

            order = list(range(len(self.all_batches)))
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(order)
            # shard WHOLE paragraphs across ranks; round-robin keeps load balanced.
            sharded = order[self.rank :: self.world_size]
            # drop-tail: every rank must yield the SAME number of batches or DDP hangs at
            # the gradient all-reduce. round-robin already gives near-equal counts; trim to
            # the global min so no rank runs ahead.
            n_per_rank = len(order) // self.world_size
            return sharded[:n_per_rank]

        def __iter__(self):
            for i in self._epoch_order():
                yield self.all_batches[i]

        def __len__(self):
            return len(self._epoch_order())

        def set_epoch(self, epoch):
            """Called by the trainer each epoch so shuffling differs across epochs."""
            self.epoch = int(epoch)

    class SoftMinTrainer(Seq2SeqTrainer):
        """Seq2SeqTrainer whose TRAIN dataloader yields one complete paragraph per batch.

        Only ``get_train_dataloader`` is overridden. Paragraph ids reach the loss via the
        ``channel`` field, which the stock ``compute_loss`` already pops + forwards as
        ``sample_channels`` (+ ``trainer=self``) -- so no ``compute_loss`` override is
        needed. Eval uses the stock dataloader and stock CE (the loss degrades to mean-CE
        when ``sample_channels`` is absent), which is correct for model selection by the
        official scorer.
        """

        def get_train_dataloader(self):
            args = self.args
            train_dataset = self.train_dataset
            if train_dataset is None:
                raise ValueError("SoftMinTrainer: training requires a train_dataset.")

            # Group by `channel` (== paragraph_id by our data contract). If a future swift
            # version repurposes `channel`, switch this to a dedicated 'paragraph_id' column.
            try:
                paragraph_ids = list(train_dataset["channel"])
            except (KeyError, TypeError) as e:
                raise RuntimeError(
                    "SoftMinTrainer expects a 'channel' column carrying paragraph ids "
                    "(build_dataset.py writes paragraph_id into row['channel']). "
                    "Confirm packing/padding_free are OFF and the column survived map()."
                ) from e

            world_size = max(1, getattr(args, "world_size", 1) or 1)
            rank = getattr(args, "process_index", 0) or 0
            sampler = ParagraphGroupSampler(
                paragraph_ids,
                shuffle=getattr(args, "train_dataloader_shuffle", True),
                seed=(getattr(args, "data_seed", None) or 42),
                rank=rank,
                world_size=world_size,
            )
            # keep a ref so the training loop can call set_epoch (see _set_para_epoch hook).
            self._para_sampler = sampler

            dataloader_params = {
                "collate_fn": self.data_collator,
                "num_workers": args.dataloader_num_workers,
                "pin_memory": args.dataloader_pin_memory,
                "persistent_workers": args.dataloader_persistent_workers,
                "prefetch_factor": args.dataloader_prefetch_factor,
                "batch_sampler": sampler,
            }
            if args.dataloader_num_workers and args.dataloader_num_workers > 0:
                try:
                    from swift.trainers.mixin import seed_worker

                    dataloader_params["worker_init_fn"] = partial(
                        seed_worker,
                        num_workers=args.dataloader_num_workers,
                        rank=args.process_index,
                    )
                except Exception:
                    pass
            return DataLoaderShard(train_dataset, device=self.accelerator.device, **dataloader_params)

    return SoftMinTrainer
