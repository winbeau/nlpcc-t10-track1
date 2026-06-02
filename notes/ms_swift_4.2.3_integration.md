# ms-swift 4.2.3 integration facts (VERIFIED against installed source)

> Verified on the server at `.venv/lib/python3.10/site-packages/swift` (ms-swift **4.2.3**).
> The workflow's `swift_softmin/` was written against **3.5.0.dev0** internals (a `/tmp/VBench`
> copy) whose paths/API differ — it will NOT run as-is. These are the real 4.2.3 anchors.

## File layout (NOT `swift/llm/...`)
- args:     `swift/arguments/base_args/base_args.py`, `swift/arguments/sft_args.py`
- sft:      `swift/pipelines/train/sft.py`
- dataset:  `swift/dataset/utils.py` (LazyLLMDataset), `swift/dataset/__init__.py`
- trainers: `swift/trainers/{seq2seq_trainer,trainer,mixin,trainer_factory}.py`
- loss:     `swift/loss/mapping.py` (`loss_map`), exported `from swift.loss import loss_map, BaseLoss`
- **No `swift/plugin/` dir**, **no `register_loss_func`/`LOSS_MAPPING`** (those are 3.5.x).

## Custom loss hook (the supported path)
- Registry: `swift/loss/mapping.py` → `loss_map = {...}`; `loss_type` arg lists `loss_map.keys()`
  (`trainers/arguments.py:165`).
- Wiring: `mixin.py:988 create_loss_and_eval_metric` → `res['compute_loss_func'] = loss_map[args.loss_type](args, self)`.
  So `loss_map[name]` is a **factory `(args, trainer) -> compute_loss_func`** (a `BaseLoss`).
- Call site (`seq2seq_trainer.py compute_loss`, ~line 190):
  `loss = compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch, loss_scale=loss_scale, trainer=self)`
  → **signature is `(outputs, labels, num_items_in_batch=None, loss_scale=None, trainer=None)`**.
  (NO `sample_channels` kwarg — the workflow's loss signature is wrong.)
  `outputs.logits` + full `labels` are available; when `compute_loss_func` is set, swift pops
  `labels`, runs `outputs = model(**inputs)`, leaves `outputs.loss=None`, then calls our fn.
- To register: import `from swift.loss import loss_map` in a `--custom_register_path` file and
  `loss_map['softmin_pem'] = <factory>`. (`custom_register_path`/`external_plugins` handled in
  `base_args._import_external_plugins`.)

## channel
- `seq2seq_trainer.compute_loss` DOES `channels = inputs.pop('channel', None)`, but only uses it
  for `enable_channel_loss` per-channel **metrics** — NOT a grouping mechanism for a custom loss.
  Don't rely on `channel` reaching our loss; instead make **batch == one paragraph** (below).

## Batch sampler injection
- `mixin.py:1219 get_train_dataloader` builds:
  `BatchSamplerShard(len(train_dataset), batch_size=self._train_batch_size, drop_last=, shuffle=,
   data_seed=, tp_size=)` → `dataloader_params['batch_sampler']` → `DataLoaderShard(train_dataset,
   device=self.accelerator.device, collate_fn=self.data_collator, worker_init_fn=..., **params)`.
- Override `get_train_dataloader` in a Seq2SeqTrainer SUBCLASS, swapping `BatchSamplerShard` for a
  paragraph batch sampler (one paragraph's sentence-indices per batch, DDP-sharded). `BatchSamplerShard`
  is from accelerate (handles cross-rank sharding) — our sampler must shard equivalently.
- `train_dataset['lengths']` column access is used (line 1252) when `group_by_length`, i.e. column
  indexing works on the eager dataset; on the lazy multimodal path read the underlying raw dataset.

## Trainer monkeypatch
- `trainer_factory.py:14`: `'causal_lm': 'swift.trainers.Seq2SeqTrainer'`, resolved via
  `importlib.import_module` + `getattr` (line 54). So patching `swift.trainers.Seq2SeqTrainer`
  BEFORE the factory runs (our launcher) takes effect.

## Other gotchas
- `lazy_tokenize` defaults **True** for multimodal (`base_args._init_lazy_tokenize`: multimodal &&
  !streaming && !packing && !group_by_length → True). Set `lazy_tokenize: false` to get the eager
  path (channel/column access simpler) — but that tokenizes all image samples up front (verify
  memory/time on the smoke run). `packing` & `lazy_tokenize` are mutually exclusive.
- `average_tokens_across_devices` (default False) multiplies final loss by `num_processes`
  (`seq2seq_trainer.py:~219`) — keep False or our per-paragraph-mean loss is inflated ×world_size.
- Outer `trainer.py:66 compute_loss` divides by `gradient_accumulation_steps` when
  `model_accepts_loss_kwargs` — standard; account for it.

## Corrected design (Option B — recommended)
Single Seq2SeqTrainer subclass `SoftMinTrainer` (installed via monkeypatch before trainer_factory):
1. `get_train_dataloader`: replicate 4.2.3's body but use a `ParagraphBatchSampler` so **each batch
   = all sentence-samples of ONE paragraph** (grouping key = `channel` from build_dataset; augmented
   minority copies are singletons). DDP-shard whole paragraphs across ranks.
2. `compute_loss`: batch is one paragraph → pop labels, forward, compute per-sentence length-normalized
   CE for each item, then the verified softmin-LSE aggregate mixed with mean-CE (reuse `loss.py` math).
   No `channel`/`loss_map`/`compute_loss_func` needed — full control in the subclass.
This drops all dependence on the (version-fragile) channel-through-collator + loss-registry plumbing.
