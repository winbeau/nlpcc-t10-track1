"""Sample-level SoftMin (PEM-bottleneck) loss for ms-swift 4.2.3.

WHY (see notes/softmin_loss_design.md, grounded in the OFFICIAL scorer
``NLPCC-2026-Task10-Science/offline_eval/evaluate.py:220-250``):
``score = (sentence Macro-F1 + PEM) / 2`` and PEM is a HARD ``AND`` over a paragraph's
sentences -- a paragraph counts iff EVERY one of its sentences is predicted correctly
(evaluate.py:224,230,236). For a greedy decode ``P(paragraph) = prod_i p_i`` (``p_i`` =
correct-label probability of sentence ``i``), so ``-log P(paragraph) = sum_i l_i``.

Plain mean-CE minimizes the AVERAGE sentence loss ``mean_i l_i`` -- it is happy to leave
one hard sentence wrong if the others are easy. But PEM is decided by the paragraph's
WORST (bottleneck) sentence. So we add a temperature-soft MAXIMUM over the per-sentence
CE losses of one paragraph (= a soft-MINIMUM over the correct-label probs = the soft
bottleneck of ``prod_i p_i``), mixed with mean-CE. The coupling lives ONLY in loss-space:
each sentence is still its own forward pass (shared evidence/claim prefix + ONE target
sentence -> ``{"label": X}``); NO cross-sentence attention, NO joint paragraph output.

OBJECTIVE (mean over the G paragraphs in the global batch):

  (1) per-sentence CE, response-only, LENGTH-NORMALIZED (geometric mean over tokens):
        l_i = -(1/|T_i|) * sum_{t in T_i} log softmax(z_{i,t})[y_{i,t}]
        p_i = exp(-l_i) in (0, 1]                    # token-geo-mean correct-label prob
      Length normalization makes long/short sentences comparable inside the soft-min.

  (2) per-paragraph NORMALIZED LSE soft-max-over-losses (the SOFTMIN core):
        S_p = (1/beta) * [ logsumexp_{i in p}(beta * l_i) - log n_p ]
            = (1/beta) * log( (1/n_p) * sum_{i in p} exp(beta * l_i) )
      Prob reading: exp(beta*l_i) = p_i^{-beta}, so S_p is a power-mean (exponent -beta)
      of the p_i = a soft MINIMUM of the correct-probs. Driving S_p down lifts the
      SMALLEST p_i = the PEM bottleneck.

  (3) total: L = (1 - lambda) * CE_mean + lambda * mean_p(S_p)
        CE_mean = (sum_p sum_{i in p} l_i) / (sum_p n_p)
      The (1 - lambda)*CE_mean floor keeps EVERY sentence (incl. non-bottleneck minority
      classes) learning -> protects Macro-F1 / prevents class collapse. lambda must be < 1.

WHY THE ``-log n_p`` NORMALIZER (the single most important correction):
  * It is a CONSTANT in l, so dS_p/dl_j = softmax_j(beta*l) is UNCHANGED by it (the
    gradient still peaks on the worst sentence). Verified numerically to 4 d.p.
  * But it makes beta->0 collapse to mean_i(l_i) -- a TRUE safe no-op (the UNnormalized
    logsumexp(beta*g)/beta DIVERGES as beta->0).
  * And it removes the n-bias: for an all-equal clean paragraph (l_i == c) the normalized
    S_p == c for ALL n, so LONG clean paragraphs (the protected ~67% majority that the
    precision-critical metric rewards) are NOT systematically over-penalized. The
    unnormalized form grew with n -- exactly the wrong incentive.

WHY LSE, not the weighted ``sum_i softmax(beta*l_i)*l_i`` form:
  The weighted form's gradient w_j + beta*w_j*(l_j - S) goes NEGATIVE on non-bottleneck
  sentences (verified: L=[0.2,1.0,0.25], beta=5 -> grad [-0.050, +1.109, -0.059]) -- it
  would actively push a GOOD sentence's loss UP. The LSE gradient = softmax(beta*l) is in
  [0,1] and sums to 1: always a safe non-negative convex combination. -> we use LSE.

HOW PARAGRAPH GROUPING WORKS (ms-swift 4.2.3, "Option B" -- verified against installed source):
  Grouping is done by the SAMPLER, not by a metadata field riding the batch. The custom
  ``ParagraphGroupSampler`` (sampler.py) makes each TRAIN batch == ALL sentence-samples of
  ONE paragraph (it reads the per-row ``channel`` = paragraph_id from the dataset to group the
  row INDICES). So inside the loss the WHOLE batch is exactly one soft-min group -- no
  ``channel``/``sample_channels`` needs to survive the collator (the 3.5.x ``channel`` pipeline
  the original draft relied on is NOT a usable custom-loss hook in 4.2.3). The
  ``softmin_pem_loss`` function below still supports explicit ``sample_channels`` grouping (used
  by the unit test); the PRODUCTION path is ``make_softmin_loss_cls`` -- a ``swift.loss.BaseLoss``
  that groups the whole batch and gates the soft-min on ``model.training`` (eval -> plain mean-CE).
  packing/padding_free MUST stay OFF (they concatenate sentences -> per-sentence l_i lost);
  use_logits_to_keep MUST be OFF (else the model returns truncated logits and the causal shift
  in ``_per_sentence_ce`` misaligns).

GRACEFUL DEGRADATION (unit-tested):
  * sample_channels is None (eval / missing ids) -> plain CE_mean.
  * lambda <= 0 -> plain CE_mean (lambda becomes a no-op).
  * singleton paragraph (n_p == 1, incl. EVERY oversampled/augmented row, which
    build_dataset.py gives a UNIQUE singleton id "<pid>#augK") -> S_p == l_1 == its CE term
    -> augmented rows cannot pollute a real paragraph's bottleneck.
  * beta->0 -> normalized S_p -> mean -> plain CE.
  * all-Supported clean paragraphs -> l_i tiny and ~equal -> w ~ uniform -> S_p ~ mean ->
    negligible extra gradient -> clean-paragraph PEM is NOT disturbed.

NUMERICAL STABILITY (mandatory):
  always ``torch.logsumexp`` (internally max-subtracts; beta*l can be tens of nats early in
  training -> never exponentiate raw); cast per-sentence to float32 even under bf16;
  ``tok_counts.clamp(min=1)`` so an all-masked sample never divides by zero; the ``-log n_p``
  term is ``math.log(len(idxs))`` subtracted AFTER logsumexp, divided by beta.

The production loss class is registered into ms-swift 4.2.3's ``swift.loss.loss_map`` under the
name ``softmin_pem`` (see ``register.py``) and selected via ``--loss_type softmin_pem``
(``mixin.py:996`` does ``loss_map[loss_type](args, trainer)``). The paragraph batching is
installed by monkeypatching ``swift.trainers.Seq2SeqTrainer`` -> ``SoftMinTrainer`` (sampler.py)
BEFORE the CLI builds the trainer (see ``scripts/train_softmin.py``).
"""

from __future__ import annotations

import os
from collections import OrderedDict

# torch is ONLY available on the server. The pure-python ``group_indices`` helper below is
# importable/testable locally; everything touching torch is deferred into function bodies.


# ---------------------------------------------------------------------------------------
# Pure-python grouping (no torch) -- unit-tested locally.
# ---------------------------------------------------------------------------------------
def group_indices(paragraph_ids):
    """Group batch positions by paragraph id, preserving first-seen order.

    Args:
        paragraph_ids: list of length B; ``paragraph_ids[i]`` is the paragraph id of
            sample ``i`` (for us, the per-row ``channel`` value).

    Returns:
        ``OrderedDict[pid -> list[int]]`` mapping each paragraph id to the (stable-ordered)
        list of batch positions belonging to it. A singleton paragraph maps to a 1-element
        list -> its ``S_p`` reduces to its own per-sentence CE (see module docstring).
    """
    groups: "OrderedDict[str, list]" = OrderedDict()
    for i, pid in enumerate(paragraph_ids):
        groups.setdefault(pid, []).append(i)
    return groups


# ---------------------------------------------------------------------------------------
# Defaults (overridable per-run via env, so the grid sweep needs no code edits).
# ---------------------------------------------------------------------------------------
_DEFAULT_BETA = 5.0      # temperature; larger -> sharper bottleneck (one-hot on worst sentence)
_DEFAULT_LAMBDA = 0.5    # softmin mix weight; keep <= 0.7 so the CE floor (1 - lambda) > 0


def _per_sentence_ce(outputs, labels):
    """Length-normalized, response-only per-sentence cross-entropy. Returns 1-D tensor [B].

    Mirrors ms-swift's default causal shift + ``ignore_index=-100`` masking (prompt tokens
    carry label -100). For each sample we sum the response-token CE and divide by the number
    of response tokens (``T_i``) -> ``l_i`` is the per-token GEOMETRIC-MEAN negative log-prob,
    i.e. ``p_i = exp(-l_i)`` is the geometric-mean correct-label probability. This per-token
    normalization is what makes ``l_i`` a single comparable scalar per sentence regardless of
    sentence length, so the soft-min is not dominated by long sentences via token count alone.
    """
    import torch
    from torch.nn import CrossEntropyLoss

    logits = outputs.logits
    # standard causal shift: predict token t+1 from position t
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=-100)
    per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    ).view(shift_labels.shape)  # [B, T-1]
    valid = shift_labels != -100
    # clamp(min=1): an all-masked / empty sample must never divide by zero.
    tok_counts = valid.sum(dim=-1).clamp(min=1).to(per_token.dtype)  # [B]
    per_sentence = per_token.sum(dim=-1) / tok_counts  # [B]
    return per_sentence


def _resolve_beta(beta_target, trainer):
    """Optional beta-warmup: ramp beta from ``SOFTMIN_BETA_MIN`` -> ``beta_target`` over the
    first ``SOFTMIN_WARMUP_FRAC`` of training (mirrors lr warmup). Early training then behaves
    closer to mean-CE before the bottleneck emphasis sharpens. No-op (returns ``beta_target``)
    when the env frac is <= 0, no trainer is passed, or ``max_steps`` is unknown.
    """
    frac = float(os.environ.get("SOFTMIN_WARMUP_FRAC", "0.0"))
    if frac <= 0.0 or trainer is None:
        return beta_target
    beta_min = float(os.environ.get("SOFTMIN_BETA_MIN", "1.0"))
    state = getattr(trainer, "state", None)
    step = float(getattr(state, "global_step", 0) or 0)
    max_steps = float(getattr(state, "max_steps", 0) or 0)
    if max_steps <= 0:
        return beta_target
    progress = min(1.0, step / max(1.0, frac * max_steps))
    return beta_min + (beta_target - beta_min) * progress


def softmin_pem_loss(
    outputs,
    labels,
    loss_scale=None,
    num_items_in_batch=None,
    sample_channels=None,      # paragraph ids -> stock compute_loss forwards them here
                               # (build_dataset.py writes paragraph_id into row["channel"]).
    trainer=None,              # stock compute_loss passes this alongside sample_channels.
    lambda_pem: float = None,  # mix weight; default env SOFTMIN_LAMBDA or 0.5.
    beta: float = None,        # temperature; default env SOFTMIN_BETA or 5.0.
    paragraph_ids=None,        # alt/legacy keyword; falls back to sample_channels.
    **extra_kwargs,
):
    """ms-swift custom loss. Signature matches the real call
    ``compute_loss_func(outputs, labels, num_items_in_batch=..., **loss_kwargs)``
    (trainers.py:219), where ``loss_kwargs`` may contain ``loss_scale``,
    ``sample_channels`` and ``trainer``.

    Returns a scalar tensor:
        (1 - lambda) * CE_mean + lambda * mean_p(S_p),
    or plain ``CE_mean`` when paragraph ids are unavailable or ``lambda <= 0``.
    """
    import math
    import torch

    lam = (
        float(os.environ.get("SOFTMIN_LAMBDA", str(_DEFAULT_LAMBDA)))
        if lambda_pem is None
        else float(lambda_pem)
    )
    bet = (
        float(os.environ.get("SOFTMIN_BETA", str(_DEFAULT_BETA)))
        if beta is None
        else float(beta)
    )
    bet = _resolve_beta(bet, trainer)

    per_sentence = _per_sentence_ce(outputs, labels).float()  # [B], float32 for a stable LSE
    mean_ce = per_sentence.mean()

    pids = sample_channels if sample_channels is not None else paragraph_ids
    # Graceful degradation: no grouping metadata (eval) or lambda disabled -> plain mean-CE.
    if pids is None or lam <= 0.0:
        return mean_ce

    groups = group_indices(list(pids))
    device = per_sentence.device
    s_terms = []
    for _pid, idxs in groups.items():
        if len(idxs) == 1:
            # singleton (incl. EVERY oversampled/augmented row): S_p == l_i, no coupling.
            s_terms.append(per_sentence[idxs[0]])
            continue
        idx_t = torch.as_tensor(idxs, device=device, dtype=torch.long)
        g = per_sentence.index_select(0, idx_t)  # [n_p]
        # NORMALIZED soft-max over losses. torch.logsumexp internally max-subtracts (stable).
        # The -log(n_p) term is constant in g, so the gradient is exactly softmax(beta*g)
        # (peaked on the WORST sentence) while gaining beta->0 -> mean and n-invariance.
        s_p = (torch.logsumexp(bet * g, dim=0) - math.log(len(idxs))) / bet
        s_terms.append(s_p)

    mean_softmin = torch.stack(s_terms).mean()
    return (1.0 - lam) * mean_ce + lam * mean_softmin


def make_softmin_loss_cls():
    """Build the ms-swift 4.2.3 ``BaseLoss`` subclass for the softmin-PEM objective (Option B).

    Deferred (imports swift/torch inside) so this module stays importable locally for the
    pure-python ``group_indices`` unit test. ``register.py`` puts the returned CLASS into
    ``swift.loss.loss_map['softmin_pem']``; ms-swift instantiates it as ``cls(args, trainer)``
    (``mixin.py:996``) and calls the instance as
    ``(outputs, labels, num_items_in_batch=..., loss_scale=..., trainer=self)``.

    Grouping: with ``ParagraphGroupSampler`` (sampler.py) each TRAIN batch is exactly ONE
    paragraph, so the whole batch is one soft-min group -- no per-row grouping metadata needed.
    During EVAL (stock dataloader, arbitrary batch) we return plain mean-CE, gated on
    ``model.training``, so a spurious cross-paragraph soft-min never fires.
    """
    import math

    import torch
    from swift.loss import BaseLoss

    class SoftMinPEMLoss(BaseLoss):
        def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
            per_sentence = _per_sentence_ce(outputs, labels).float()  # [B], float32 stable LSE
            mean_ce = per_sentence.mean()

            lam = float(os.environ.get("SOFTMIN_LAMBDA", str(_DEFAULT_LAMBDA)))
            bet = float(os.environ.get("SOFTMIN_BETA", str(_DEFAULT_BETA)))
            bet = _resolve_beta(bet, self.trainer)

            model = getattr(self.trainer, "model", None)
            training = bool(getattr(model, "training", True))
            n = int(per_sentence.shape[0])
            # Eval / lambda disabled / singleton batch -> plain mean-CE (no spurious soft-min).
            if (not training) or lam <= 0.0 or n < 2:
                return mean_ce
            # TRAIN: ParagraphGroupSampler guarantees the whole batch == ONE paragraph.
            # Normalized LSE soft-max-over-losses; -log n is constant in l so the gradient is
            # softmax(beta*l), peaked on the worst sentence (see module docstring + unit test).
            s_p = (torch.logsumexp(bet * per_sentence, dim=0) - math.log(n)) / bet
            return (1.0 - lam) * mean_ce + lam * s_p

    return SoftMinPEMLoss
