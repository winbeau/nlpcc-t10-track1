"""DEPRECATED single-file shim -> use the package src/nlpcc_t10/swift_softmin/ instead.

The SoftMin (PEM-bottleneck) loss now lives in the modular package:

    src/nlpcc_t10/swift_softmin/
        loss.py     -- softmin_pem_loss, group_indices, _per_sentence_ce, _resolve_beta
        sampler.py  -- ParagraphGroupSampler, build_paragraph_trainer_cls
        register.py -- the --custom_register_path entrypoint (register loss + patch trainer)

This file is kept ONLY as a backwards-compatible re-export so that any leftover reference to
`nlpcc_t10.softmin_plugin` (or `--custom_register_path src/nlpcc_t10/softmin_plugin.py`) keeps
working. New code, configs and the launcher (scripts/train_softmin.py) reference the package.

PREFER:
    swift sft ... --loss_type softmin_pem \
        --custom_register_path src/nlpcc_t10/swift_softmin/register.py
"""

from __future__ import annotations

# Re-export the public API from the package (single source of truth).
from .swift_softmin.loss import (  # noqa: F401
    _per_sentence_ce,
    _resolve_beta,
    group_indices,
    softmin_pem_loss,
)
from .swift_softmin.sampler import build_paragraph_trainer_cls  # noqa: F401


def register():
    """Backwards-compatible register() -> delegates to swift_softmin.register.register()."""
    from .swift_softmin.register import register as _register

    _register()


# ms-swift runs module top-level code when this file is passed to --custom_register_path.
# Register on import (no-op locally without swift; the package's register module is wrapped).
try:  # pragma: no cover - only meaningful on the server
    register()
except Exception:
    pass


if __name__ == "__main__":
    # Pure-python self-tests (no torch) for the normalized-LSE math, mirroring the loss core.
    import math

    g = group_indices(["p1", "p1", "p2", "p1", "p3", "p3"])
    assert list(g.keys()) == ["p1", "p2", "p3"], g
    assert g["p1"] == [0, 1, 3] and g["p2"] == [2] and g["p3"] == [4, 5], g

    def sm(losses, beta):
        m = max(beta * l for l in losses)
        lse = m + math.log(sum(math.exp(beta * l - m) for l in losses))
        return (lse - math.log(len(losses))) / beta

    def softmax(losses, beta):
        m = max(beta * l for l in losses)
        e = [math.exp(beta * l - m) for l in losses]
        s = sum(e)
        return [x / s for x in e]

    def num_grad(losses, beta, eps=1e-7):
        out = []
        for j in range(len(losses)):
            lp = losses[:]; lp[j] += eps
            lm = losses[:]; lm[j] -= eps
            out.append((sm(lp, beta) - sm(lm, beta)) / (2 * eps))
        return out

    assert abs(sm([0.7], 5.0) - 0.7) < 1e-9  # singleton identity
    L = [0.1, 0.1, 2.0]; n = len(L)
    assert sum(L) / n < sm(L, 5.0)
    assert max(L) - math.log(n) / 5.0 - 1e-9 <= sm(L, 5.0) <= max(L) + 1e-9
    assert 0.0 <= max(L) - sm(L, 200.0) <= math.log(n) / 200.0 + 1e-9
    for LL, b in [([0.2, 1.5, 0.4], 4.0), ([0.1, 0.1, 2.0], 5.0)]:
        ng, sx = num_grad(LL, b), softmax(LL, b)
        assert all(abs(a - c) < 1e-4 for a, c in zip(ng, sx)), (LL, b, ng, sx)
        assert all(w >= -1e-12 for w in sx) and abs(sum(sx) - 1.0) < 1e-9
    assert abs(sm([0.3, 0.7, 1.1], 0.001) - (0.3 + 0.7 + 1.1) / 3) < 1e-3
    for nn in (2, 4, 8):
        assert abs(sm([0.5] * nn, 5.0) - 0.5) < 1e-9
    print("softmin_plugin (shim) pure-python self-tests OK -> use swift_softmin package")
