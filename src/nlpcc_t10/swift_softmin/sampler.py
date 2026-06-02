"""Paragraph-group batch sampler + SoftMinTrainer for ms-swift 4.2.3 (Option B).

The SoftMin PEM loss (``loss.py``) needs each optimizer batch to contain ALL the
sentence-samples of ONE paragraph, intact, so the per-paragraph soft-min over per-sentence
losses can be computed over the whole batch. ms-swift's stock sampler batches by global index
and would split a paragraph across batches / ranks. This module provides:

  * ``ParagraphGroupSampler`` -- a ``batch_sampler`` that yields one variable-length batch per
    paragraph (round-robin sharded by rank for DDP, so each GPU always sees COMPLETE paragraphs).
  * ``build_paragraph_trainer_cls()`` -- a lazy factory returning a ``Seq2SeqTrainer`` subclass
    (``SoftMinTrainer``) that overrides ``get_train_dataloader`` to install the sampler. Because
    each train batch is then exactly one paragraph, the loss treats the whole batch as one
    soft-min group; NO ``compute_loss`` override and NO ``channel``-through-collator plumbing is
    needed (the 3.5.x channel hook does not exist in 4.2.3).

VERIFIED ms-swift 4.2.3 anchors (see notes/ms_swift_4.2.3_integration.md):
  * ``DataLoaderShard`` is in ``swift.dataloader`` (NOT ``swift.llm``).
  * ``seed_worker`` is in ``swift.utils``.
  * ``get_train_dataloader(self, skip_batches=0)`` (mixin.py:1219) -- the resume path calls it
    with ``skip_batches`` (mixin.py:1164), so the override MUST accept it.

DDP correctness (2xL40): Accelerate does NOT auto-shard a custom ``batch_sampler``, so the
sampler shards WHOLE paragraphs round-robin (``order[rank::world_size]``) and drop-tails to an
equal per-rank count (else DDP hangs at the gradient all-reduce). Never split a paragraph across
ranks -- a per-rank partial-paragraph LSE is the WRONG bottleneck and fails silently.

torch / swift imports are deferred into ``build_paragraph_trainer_cls`` so this module is
importable locally (no torch) for documentation / introspection.
"""

from __future__ import annotations

from collections import defaultdict


def build_paragraph_trainer_cls():
    """Return a ``Seq2SeqTrainer`` subclass whose train dataloader batches one paragraph at a
    time. Call on the server (inside the train entrypoint), AFTER swift is importable.

    Monkeypatched in for the stock ``Seq2SeqTrainer`` by ``register.py`` / ``scripts/train_softmin.py``
    BEFORE the swift CLI constructs the trainer (trainer_factory resolves
    ``swift.trainers.Seq2SeqTrainer`` dynamically, so the patch takes effect).
    """
    from functools import partial

    from torch.utils.data import Sampler
    from swift.dataloader import DataLoaderShard
    from swift.trainers import Seq2SeqTrainer

    def _read_channel_column(ds):
        """Read the per-row ``channel`` (== paragraph_id) aligned with dataset index order.

        Works for both the eager HfDataset (column access ``ds['channel']``) and the lazy
        multimodal path (``LazyLLMDataset`` wraps the raw dataset at ``ds.dataset``; index i of
        the lazy dataset encodes raw row i, so ``ds.dataset['channel'][i]`` aligns with i).
        """
        last_err = None
        for obj in (ds, getattr(ds, "dataset", None)):
            if obj is None:
                continue
            try:
                col = list(obj["channel"])
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
            if len(col) == len(ds):
                return col
        raise RuntimeError(
            "SoftMinTrainer: could not read a 'channel' column aligned with the train dataset "
            "(build_dataset.py writes paragraph_id into row['channel']). Ensure packing/"
            "padding_free are OFF and the dataset still carries 'channel'."
        ) from last_err

    class ParagraphGroupSampler(Sampler):
        """A ``batch_sampler``: each step yields a LIST of dataset indices forming ONE paragraph.

        Args:
            paragraph_ids: list aligned with the train dataset; ``paragraph_ids[i]`` is the
                ``channel`` (== paragraph id) of dataset row ``i``.
            shuffle: shuffle the ORDER of paragraphs each epoch (sentences within a paragraph stay
                together; their intra-paragraph order is irrelevant to the soft-min).
            seed: base RNG seed; per-epoch seed is ``seed + epoch``.
            rank, world_size: DDP sharding -- this rank takes ``order[rank::world_size]``.
        """

        def __init__(self, paragraph_ids, shuffle=True, seed=42, rank=0, world_size=1):
            groups = defaultdict(list)
            for idx, pid in enumerate(paragraph_ids):
                groups[pid].append(idx)
            self.all_batches = list(groups.values())  # one entry per paragraph
            self.shuffle = shuffle
            self.seed = int(seed)
            self.rank = int(rank)
            self.world_size = max(1, int(world_size))
            self.epoch = 0

        def _epoch_order(self):
            import random

            order = list(range(len(self.all_batches)))
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(order)
            sharded = order[self.rank :: self.world_size]
            # drop-tail to the global per-rank min so every rank yields the SAME #batches
            # (else DDP hangs at the gradient all-reduce).
            n_per_rank = len(order) // self.world_size
            return sharded[:n_per_rank]

        def __iter__(self):
            for i in self._epoch_order():
                yield self.all_batches[i]

        def __len__(self):
            return len(self._epoch_order())

        def set_epoch(self, epoch):
            self.epoch = int(epoch)

    class SoftMinTrainer(Seq2SeqTrainer):
        """Seq2SeqTrainer whose TRAIN dataloader yields one complete paragraph per batch.

        Only ``get_train_dataloader`` is overridden; the loss is the registered
        ``softmin_pem`` ``BaseLoss`` (selected via ``--loss_type softmin_pem``). Eval uses the
        stock dataloader; the loss returns mean-CE in eval mode, correct for model selection by
        the official scorer.
        """

        def get_train_dataloader(self, skip_batches=0):
            args = self.args
            train_dataset = self.train_dataset
            if train_dataset is None:
                raise ValueError("SoftMinTrainer: training requires a train_dataset.")
            if not hasattr(train_dataset, "__len__"):
                # IterableDataset is incompatible with paragraph batching.
                raise ValueError("SoftMinTrainer requires a map-style (sized) train_dataset.")

            paragraph_ids = _read_channel_column(train_dataset)

            world_size = max(1, getattr(args, "world_size", 1) or 1)
            rank = getattr(args, "process_index", 0) or 0
            sampler = ParagraphGroupSampler(
                paragraph_ids,
                shuffle=getattr(args, "train_dataloader_shuffle", True),
                seed=(getattr(args, "data_seed", None) or 42),
                rank=rank,
                world_size=world_size,
            )
            self._para_sampler = sampler  # keep a ref so set_epoch reaches the inner sampler

            batch_sampler = sampler
            if skip_batches and skip_batches > 0:
                from accelerate.data_loader import SkipBatchSampler

                batch_sampler = SkipBatchSampler(batch_sampler, skip_batches=skip_batches)

            dataloader_params = {
                "collate_fn": self.data_collator,
                "num_workers": args.dataloader_num_workers,
                "pin_memory": args.dataloader_pin_memory,
                "persistent_workers": args.dataloader_persistent_workers,
                "prefetch_factor": args.dataloader_prefetch_factor,
                "batch_sampler": batch_sampler,
            }
            if args.dataloader_num_workers and args.dataloader_num_workers > 0:
                try:
                    from swift.utils import seed_worker

                    dataloader_params["worker_init_fn"] = partial(
                        seed_worker,
                        num_workers=args.dataloader_num_workers,
                        rank=args.process_index,
                    )
                except Exception:  # noqa: BLE001
                    pass

            return DataLoaderShard(
                train_dataset, device=self.accelerator.device, **dataloader_params
            )

    return SoftMinTrainer
