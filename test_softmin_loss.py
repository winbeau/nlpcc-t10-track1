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
    # only the LAST position is a response token contributing to CE
    logits[-1, target_token] = float(conf)
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


def main():
    torch.manual_seed(0)
    print("Running SoftMin PEM loss torch tests...")
    test_grouping_pure()
    test_numerical_stability()
    test_bottleneck_gradient()
    test_beta_to_zero_recovers_mean_ce()
    test_singleton_identity()
    test_graceful_degrade()
    print("ALL SOFTMIN TORCH TESTS PASSED")


if __name__ == "__main__":
    main()
