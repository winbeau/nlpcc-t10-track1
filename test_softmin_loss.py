"""Standalone torch test for the SoftMin PEM loss (src/nlpcc_t10/swift_softmin/loss.py).

torch is NOT installed in the local dev env -- RUN THIS ON THE SERVER (after `uv sync
--extra train`, where torch is present). swift is NOT required: we call the loss function
directly with a fake `outputs` object exposing `.logits`, so the test isolates the math.

    # on the server, from the repo root:
    PYTHONPATH=src python test_softmin_loss.py
    # (or, if the package is installed:  python test_softmin_loss.py)

It asserts, on TWO toy paragraphs of per-sentence samples (shared-prefix + one target
sentence each -> a one-token {"label":X} response, modeled here as a single response token
per sample):

  T1. NUMERICAL STABILITY: the loss is finite for huge logit gaps (CE ~ tens of nats) where a
      naive exp(beta*l) would overflow; logsumexp must keep it finite. Also finite under bf16.
  T2. BOTTLENECK GRADIENT: in a paragraph with one WRONG sentence and one already-CORRECT
      sentence, |d loss / d logits| of the WRONG (worst) sentence >> that of the correct one
      (the soft-min routes gradient to the PEM bottleneck). We also check the softmin loss
      concentrates MORE gradient on the worst sentence than plain mean-CE does.
  T3. beta -> 0 RECOVERS MEAN CE: with lambda=1 (pure softmin term) and beta->0, the loss
      equals plain mean per-sentence CE (the -log n_p normalizer makes this a true no-op).
  T4. SINGLETON IDENTITY: a singleton paragraph (every augmented/oversampled row) contributes
      exactly its own per-sentence CE -- it cannot pollute a real paragraph's bottleneck.
  T5. GRACEFUL DEGRADE: sample_channels=None (eval) and lambda<=0 both return plain mean CE.
  T6. USE_LOGITS_TO_KEEP EQUIVALENCE: per-sentence CE on the FULL [B,T,V] logits == on the
      trailing-K window ms-swift's prepare_logits_to_keep (mixin.py, batch>1 branch) produces
      when use_logits_to_keep=True. Proves the loss stays CORRECT when only the last K=
      T-earliest_response_start+1 positions are kept -> full-resolution training without the
      full-vocab logits OOM.

Exit code 0 and "ALL SOFTMIN TORCH TESTS PASSED" on success; raises AssertionError otherwise.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Make `from nlpcc_t10.swift_softmin...` importable when run from the repo root without install.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nlpcc_t10.swift_softmin.loss import (  # noqa: E402
    _per_sentence_ce,
    group_indices,
    softmin_pem_loss,
)

VOCAB = 8
IGNORE = -100


def make_outputs(logits):
    """Wrap a [B, T, V] logits tensor in the minimal object the loss expects (`.logits`)."""
    return SimpleNamespace(logits=logits)


def build_sample(target_token, conf, prefix_len=2, vocab=VOCAB, dtype=torch.float32):
    """Build one sentence-sample as (logits[T,V], labels[T]).

    The sample is a shared prefix (labels=-100, masked out of the CE) followed by ONE response
    token = the gold label token. `conf` sets how confident the model is on the gold token:
      large positive -> low CE (already correct);  negative/zero -> high CE (wrong / unsure).
    Returns per-position logits and labels for ONE sample; we stack samples into a batch.
    """
    T = prefix_len + 1
    logits = torch.zeros(T, vocab, dtype=dtype)
    labels = torch.full((T,), IGNORE, dtype=torch.long)
    # The response token is the LAST position (labels[-1]). Under the causal shift in
    # _per_sentence_ce (logits[..., :-1] vs labels[..., 1:]) that token is PREDICTED by the
    # logits at the PRECEDING position (-2), so `conf` must live on -2. Putting it on -1 would
    # score a uniform (all-zero) position -> CE = log(VOCAB) regardless of conf (the original
    # fixture bug, which silently turned every sample uniform and made T1/T2/T3 vacuous).
    logits[-2, target_token] = float(conf)
    labels[-1] = target_token
    return logits, labels


def stack_batch(samples):
    """samples: list of (logits[T,V], labels[T]) with equal T -> (outputs, labels[B,T])."""
    logits = torch.stack([s[0] for s in samples], dim=0).requires_grad_(True)
    labels = torch.stack([s[1] for s in samples], dim=0)
    return logits, labels


# ----------------------------------------------------------------------------------------
def test_grouping_pure():
    g = group_indices(["pA", "pA", "pB", "pA", "pC", "pC"])
    assert list(g.keys()) == ["pA", "pB", "pC"], g
    assert g["pA"] == [0, 1, 3] and g["pB"] == [2] and g["pC"] == [4, 5], g
    print("  [T0] grouping order/membership OK")


def test_numerical_stability():
    # Huge logit gaps -> per-sentence CE of tens of nats; exp(beta*l) would overflow,
    # logsumexp must keep S_p (hence the loss) finite. Two paragraphs, n=2 and n=2.
    samples = [
        build_sample(0, conf=40.0),    # ~0 CE (very confident, correct)
        build_sample(1, conf=-40.0),   # huge CE: gold token has logit -40 vs 0 elsewhere
        build_sample(2, conf=40.0),
        build_sample(3, conf=-40.0),
    ]
    logits, labels = stack_batch(samples)
    channels = ["p1", "p1", "p2", "p2"]
    loss = softmin_pem_loss(make_outputs(logits), labels,
                            sample_channels=channels, beta=10.0, lambda_pem=0.5)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    assert torch.isfinite(logits.grad).all(), "non-finite gradient"
    print(f"  [T1] numerically stable with +/-40 logits, beta=10 -> loss={loss.item():.4f} (finite)")

    # bf16 path (skip if no GPU/bf16): cast logits to bf16, loss casts per-sentence to float32.
    if torch.cuda.is_available():
        lb = logits.detach().to("cuda").to(torch.bfloat16).requires_grad_(True)
        lab = labels.to("cuda")
        lbf = softmin_pem_loss(make_outputs(lb), lab, sample_channels=channels,
                               beta=10.0, lambda_pem=0.5)
        assert torch.isfinite(lbf), f"bf16 loss not finite: {lbf}"
        print("       bf16/CUDA path finite too")
    else:
        print("       (bf16/CUDA path skipped: no GPU)")


def test_bottleneck_gradient():
    # ONE paragraph, 2 sentences: sentence 0 already CORRECT (low CE), sentence 1 WRONG (high CE).
    # conf chosen so the correct sentence is NOT fully saturated -> mean-CE still puts real
    # gradient on it, giving the softmin-vs-meanCE comparison a comfortable margin.
    s_correct = build_sample(0, conf=4.0)    # confident-ish correct -> small but nonzero CE grad
    s_wrong = build_sample(1, conf=-1.0)     # gold token logit below others -> large CE
    logits, labels = stack_batch([s_correct, s_wrong])
    channels = ["pX", "pX"]

    # sanity: per-sentence CE -- sentence 1 (worst) has much larger CE than sentence 0.
    with torch.no_grad():
        ce = _per_sentence_ce(make_outputs(logits), labels)
    assert ce[1] > ce[0], f"expected worse CE on the wrong sentence: {ce.tolist()}"

    loss = softmin_pem_loss(make_outputs(logits), labels,
                            sample_channels=channels, beta=5.0, lambda_pem=1.0)  # pure softmin
    loss.backward()
    g0 = logits.grad[0].abs().sum().item()   # gradient magnitude on the CORRECT sentence
    g1 = logits.grad[1].abs().sum().item()   # gradient magnitude on the WORST sentence
    assert g1 > g0, f"worst sentence should get larger grad: g_worst={g1:.4f} g_ok={g0:.4f}"
    # the soft-min routes the vast majority of gradient to the bottleneck.
    assert g1 > 3.0 * g0, f"expected strong concentration: g_worst={g1:.4f} g_ok={g0:.4f}"
    print(f"  [T2] bottleneck grad: |g_worst|={g1:.4f} >> |g_ok|={g0:.4f} "
          f"(ratio {g1 / max(g0, 1e-12):.1f}x)")

    # softmin concentrates MORE on the bottleneck than plain mean-CE does.
    logits_ce = logits.detach().clone().requires_grad_(True)
    ce_loss = softmin_pem_loss(make_outputs(logits_ce), labels,
                               sample_channels=None)  # None -> plain mean CE
    ce_loss.backward()
    c0 = logits_ce.grad[0].abs().sum().item()
    c1 = logits_ce.grad[1].abs().sum().item()
    frac_softmin = g1 / max(g0 + g1, 1e-12)
    frac_meance = c1 / max(c0 + c1, 1e-12)
    assert frac_softmin > frac_meance, (
        f"softmin should put a larger grad FRACTION on the worst sentence: "
        f"softmin={frac_softmin:.3f} vs meanCE={frac_meance:.3f}")
    print(f"       worst-sentence grad fraction: softmin={frac_softmin:.3f} > meanCE={frac_meance:.3f}")


def test_beta_to_zero_recovers_mean_ce():
    # lambda=1 (pure softmin term). As beta->0 the NORMALIZED LSE -> mean_i(l_i) = plain mean CE.
    samples = [build_sample(0, conf=6.0), build_sample(1, conf=1.0), build_sample(2, conf=-2.0)]
    logits, labels = stack_batch(samples)
    channels = ["pZ", "pZ", "pZ"]  # one paragraph of 3 sentences

    with torch.no_grad():
        mean_ce = _per_sentence_ce(make_outputs(logits), labels).mean().item()
        loss_b0 = softmin_pem_loss(make_outputs(logits), labels, sample_channels=channels,
                                   beta=1e-4, lambda_pem=1.0).item()
    assert abs(loss_b0 - mean_ce) < 1e-3, f"beta->0 should recover mean CE: {loss_b0} vs {mean_ce}"
    print(f"  [T3] beta->0 (lambda=1) -> loss={loss_b0:.5f} == mean CE {mean_ce:.5f}")

    # and S_p <= max_i l_i (soft UNDER-estimate of the hard max) at a finite beta.
    with torch.no_grad():
        ce = _per_sentence_ce(make_outputs(logits), labels)
        loss_b5 = softmin_pem_loss(make_outputs(logits), labels, sample_channels=channels,
                                   beta=5.0, lambda_pem=1.0).item()
    assert mean_ce - 1e-6 <= loss_b5 <= ce.max().item() + 1e-6, (
        f"S_p out of [mean,max]: mean={mean_ce} S={loss_b5} max={ce.max().item()}")
    print(f"       finite-beta S_p={loss_b5:.5f} in [mean {mean_ce:.5f}, max {ce.max().item():.5f}]")


def test_singleton_identity():
    # A singleton paragraph (every oversampled/augmented row) contributes exactly its own CE.
    s = build_sample(3, conf=0.5)
    logits, labels = stack_batch([s])
    with torch.no_grad():
        ce = _per_sentence_ce(make_outputs(logits), labels).mean().item()
        loss = softmin_pem_loss(make_outputs(logits), labels, sample_channels=["solo"],
                                beta=5.0, lambda_pem=0.5).item()
    assert abs(loss - ce) < 1e-6, f"singleton S_p must equal its CE: {loss} vs {ce}"
    print(f"  [T4] singleton identity: loss={loss:.5f} == its CE {ce:.5f}")


def test_graceful_degrade():
    samples = [build_sample(0, conf=2.0), build_sample(1, conf=-1.0)]
    logits, labels = stack_batch(samples)
    with torch.no_grad():
        mean_ce = _per_sentence_ce(make_outputs(logits), labels).mean().item()
        # eval: no channels -> plain mean CE
        l_none = softmin_pem_loss(make_outputs(logits), labels, sample_channels=None).item()
        # lambda<=0 -> plain mean CE even with channels present
        l_lam0 = softmin_pem_loss(make_outputs(logits), labels,
                                  sample_channels=["p", "p"], lambda_pem=0.0).item()
    assert abs(l_none - mean_ce) < 1e-6, (l_none, mean_ce)
    assert abs(l_lam0 - mean_ce) < 1e-6, (l_lam0, mean_ce)
    print(f"  [T5] graceful degrade: channels=None -> {l_none:.5f}, lambda=0 -> {l_lam0:.5f} "
          f"== mean CE {mean_ce:.5f}")


def test_use_logits_to_keep_equivalence():
    """T6. The GOLDEN gate for P0 (use_logits_to_keep -> full-resolution training).

    When use_logits_to_keep=True, ms-swift's prepare_logits_to_keep (mixin.py:1122, batch>1
    branch) replaces inputs['labels'] with its trailing K columns (K = T - earliest non--100
    index + 1) and asks the model for logits over only those last K positions. So our loss
    receives logits[B,K,V] + labels[B,K] instead of the full [B,T,*]. Because _per_sentence_ce
    masks -100 and only the causal-shift pairs (logits[p-1], labels[p]) of RESPONSE tokens
    contribute -- and K is chosen so every response token AND its predecessor sit inside the
    window -- the per-sentence CE must be byte-for-byte identical. If this fails, enabling
    use_logits_to_keep would silently change the loss; do NOT ship without this passing.

    Batch mimics a real paragraph batch: responses start at DIFFERENT positions (varying prompt
    lengths) with -100 prompt prefixes and trailing -100 padding."""
    torch.manual_seed(7)
    B, T, V = 4, 16, VOCAB
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), IGNORE, dtype=torch.long)
    spans = [(9, 2), (5, 3), (11, 1), (7, 4)]  # (start, len); earliest start=5 -> K = 16-5+1 = 12
    for b, (start, ln) in enumerate(spans):
        for j in range(ln):
            labels[b, start + j] = (b * 3 + j) % V
    with torch.no_grad():
        full = _per_sentence_ce(make_outputs(logits), labels)
        # Replicate prepare_logits_to_keep's batch>1 slicing exactly:
        first_resp = (labels != IGNORE).int().argmax(-1)          # first non--100 idx per row
        K = T - int(first_resp.min().item()) + 1
        trunc = _per_sentence_ce(make_outputs(logits[:, -K:, :].contiguous()),
                                 labels[:, -K:].contiguous())
    assert torch.allclose(full, trunc, atol=1e-5), (
        f"per-sentence CE differs full vs truncated(K={K}):\n full={full}\n trunc={trunc}")
    print(f"  [T6] use_logits_to_keep equivalence OK (T={T} -> K={K}): per-sentence CE identical "
          f"(full == prepare_logits_to_keep trailing-K window)")


def test_label_span_bottleneck():
    """T7 (E1 CoT): the soft-min bottleneck must track LABEL uncertainty, NOT <analysis>
    perplexity. Build a 2-sentence paragraph where:
      A: HIGH analysis-token CE (badly-modeled rationale) but CONFIDENT, correct label.
      B: LOW analysis-token CE but UNCERTAIN label.
    With label-span restriction, the bottleneck gradient must land on B's label token; the
    naive full-response soft-min would (wrongly) route it to A's analysis token. Also assert
    the (1-lambda) mean-CE FLOOR still flows gradient into A's analysis tokens (so analysis is
    still learned). Span detection uses brace token id == BRACE.
    """
    from nlpcc_t10.swift_softmin.loss import _label_span_mask

    V = VOCAB
    BRACE = 7          # pretend token id 7's piece contains '{'
    LBL_A, LBL_B = 3, 4
    ANALYSIS_TOK = 5
    # layout per row (T=5): [analysis, analysis, '{', label, <pad-to-shift>]
    # response tokens (labels != -100): positions 0..3 are analysis/json; we mask prefix none here.
    def row(analysis_conf, label_tok, label_conf):
        T = 5
        logits = torch.zeros(T, V)
        labels = torch.full((T,), IGNORE, dtype=torch.long)
        # response = positions 1..4 (position 0 is a 'prompt' anchor, masked)
        # tokens predicted at t from logits[t-1]; put confidences on the preceding position.
        labels[1] = ANALYSIS_TOK; logits[0, ANALYSIS_TOK] = analysis_conf
        labels[2] = ANALYSIS_TOK; logits[1, ANALYSIS_TOK] = analysis_conf
        labels[3] = BRACE;        logits[2, BRACE] = 8.0          # '{' always confident
        labels[4] = label_tok;    logits[3, label_tok] = label_conf
        return logits, labels

    # A: analysis_conf LOW (=0 -> high CE) but label_conf HIGH; B: analysis HIGH, label LOW.
    a = row(analysis_conf=0.0, label_tok=LBL_A, label_conf=10.0)
    b = row(analysis_conf=10.0, label_tok=LBL_B, label_conf=0.0)
    logits, labels = stack_batch([a, b])

    restrict = _label_span_mask(labels, {BRACE})
    # span must start at the '{' (pos 3) for both rows -> covers pos 3,4 only
    assert restrict[0].tolist() == [False, False, False, True, True], restrict[0].tolist()

    label_ce = _per_sentence_ce(make_outputs(logits), labels, restrict=restrict)
    full_ce = _per_sentence_ce(make_outputs(logits), labels)
    # label-span: B (uncertain label) must be the worse sentence; full-response: A (bad analysis) is.
    assert label_ce[1] > label_ce[0], (label_ce.tolist())
    assert full_ce[0] > full_ce[1], (full_ce.tolist())

    # bottleneck gradient with label-span lands on B's label token (logits row 1, pos 3, LBL_B).
    logits2 = logits.detach().clone().requires_grad_(True)
    restrict2 = _label_span_mask(labels, {BRACE})
    lce = _per_sentence_ce(make_outputs(logits2), labels, restrict=restrict2).float()
    import math as _m
    s_p = (torch.logsumexp(5.0 * lce, dim=0) - _m.log(2)) / 5.0
    s_p.backward()
    g_label_B = logits2.grad[1, 3].abs().sum().item()   # B label token
    g_label_A = logits2.grad[0, 3].abs().sum().item()   # A label token
    g_analysis_A = logits2.grad[0, :2].abs().sum().item()  # A analysis tokens
    assert g_label_B > g_label_A, (g_label_B, g_label_A)
    assert g_analysis_A < 1e-9, f"analysis got soft-min gradient ({g_analysis_A}); span leaked"

    # FLOOR check: full-response mean-CE DOES flow gradient into A's analysis tokens.
    logits3 = logits.detach().clone().requires_grad_(True)
    floor = _per_sentence_ce(make_outputs(logits3), labels).float().mean()
    floor.backward()
    assert logits3.grad[0, :2].abs().sum().item() > 1e-6, "floor must teach analysis tokens"
    print(f"  [T7] label-span bottleneck: g_labelB={g_label_B:.4f} > g_labelA={g_label_A:.4f}; "
          f"analysis soft-min grad={g_analysis_A:.2e} (~0); floor teaches analysis OK")


def test_label_span_fallback_equals_plain():
    """T8: empty brace_ids (no tokenizer) OR label-only data (first response token is '{') ->
    label-span == full response, so the objective is bit-identical to the pre-CoT s01 loss."""
    V = VOCAB
    # label-only style row: response = ['{', label] (first response token is the brace)
    def row(label_tok, conf):
        T = 4
        logits = torch.zeros(T, V); labels = torch.full((T,), IGNORE, dtype=torch.long)
        labels[2] = 7; logits[1, 7] = 8.0          # '{'
        labels[3] = label_tok; logits[2, label_tok] = conf
        return logits, labels
    logits, labels = stack_batch([row(3, 1.0), row(4, 5.0)])
    full = _per_sentence_ce(make_outputs(logits), labels)
    span = _per_sentence_ce(make_outputs(logits), labels,
                            restrict=__import__("nlpcc_t10.swift_softmin.loss", fromlist=["_label_span_mask"])._label_span_mask(labels, {7}))
    # first response token IS the brace -> span covers the whole response -> identical CE
    assert torch.allclose(full, span), (full.tolist(), span.tolist())
    # empty brace_ids -> whole response too
    span_empty = _per_sentence_ce(make_outputs(logits), labels,
                                  restrict=__import__("nlpcc_t10.swift_softmin.loss", fromlist=["_label_span_mask"])._label_span_mask(labels, set()))
    assert torch.allclose(full, span_empty)
    print("  [T8] label-span fallback == plain full-response CE (E0/label-only & no-tokenizer)")


def main():
    torch.manual_seed(0)
    print("Running SoftMin PEM loss torch tests...")
    test_grouping_pure()
    test_numerical_stability()
    test_bottleneck_gradient()
    test_beta_to_zero_recovers_mean_ce()
    test_singleton_identity()
    test_graceful_degrade()
    test_use_logits_to_keep_equivalence()
    test_label_span_bottleneck()
    test_label_span_fallback_equals_plain()
    print("ALL SOFTMIN TORCH TESTS PASSED")


if __name__ == "__main__":
    main()
