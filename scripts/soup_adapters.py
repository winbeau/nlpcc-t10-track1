#!/usr/bin/env python3
"""C1 model-soup — average LoRA adapters into one merged adapter (flat-minimum / OOD robustness).

Two modes:
  --mode factor (default, C1b, cheap PROBE): naive mean of the raw lora_A / lora_B factors.
     ⚠️ NOT equal to averaging the effective delta ΔW=(α/r)·B·A (bilinear). Fast sanity probe only.
  --mode delta  (C1a, semantically correct): average effective ΔWᵢ per module, then truncated-SVD
     re-factor back to rank-r. This is the "average the weight deltas" the soup intent actually wants.

Preconditions (asserted, abort otherwise): all adapters share base model + r + lora_alpha +
target_modules + peft_type, and identical safetensors key sets + shapes. Writes a loadable adapter
dir (adapter_model.safetensors + adapter_config.json copied from member 0).

Runs on the server (needs torch + safetensors); CPU only.

  uv run python scripts/soup_adapters.py --out outputs/phaseC/C1/soup_adapter \
      --adapters <ckptA> <ckptB> <ckptC> [--mode delta] [--weights 0.5,0.3,0.2]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def load_adapter(d: Path) -> tuple[dict, dict]:
    sd = load_file(str(d / "adapter_model.safetensors"))
    cfg = json.loads((d / "adapter_config.json").read_text())
    return sd, cfg


def assert_compatible(cfgs: list[dict], sds: list[dict]) -> None:
    keys = [k for k in ("base_model_name_or_path", "r", "lora_alpha", "peft_type") if k in cfgs[0]]
    for c in cfgs[1:]:
        for k in keys:
            assert c.get(k) == cfgs[0].get(k), f"adapter mismatch on {k}: {c.get(k)} != {cfgs[0].get(k)}"
        assert sorted(c.get("target_modules", [])) == sorted(cfgs[0].get("target_modules", [])), \
            "target_modules differ"
    k0 = set(sds[0])
    for sd in sds[1:]:
        assert set(sd) == k0, f"safetensors key sets differ ({len(set(sd) ^ k0)} symmetric-diff keys)"
    for k in k0:
        for sd in sds[1:]:
            assert sd[k].shape == sds[0][k].shape, f"shape mismatch at {k}: {sd[k].shape} vs {sds[0][k].shape}"


def soup_factor(sds: list[dict], w: list[float]) -> dict:
    out = {}
    for k in sds[0]:
        acc = sum(wi * sd[k].float() for wi, sd in zip(w, sds))
        out[k] = acc.to(sds[0][k].dtype)
    return out


def soup_delta(sds: list[dict], cfg: dict, w: list[float]) -> dict:
    """Average effective ΔW per LoRA module, re-factor to rank-r via SVD. Non-LoRA tensors are mean'd."""
    r = int(cfg["r"])
    scaling = float(cfg["lora_alpha"]) / r
    out = {}
    a_keys = [k for k in sds[0] if k.endswith("lora_A.weight")]
    handled = set()
    for ak in a_keys:
        bk = ak.replace("lora_A.weight", "lora_B.weight")
        if bk not in sds[0]:
            continue
        # ΔW_i = scaling * B_i @ A_i ; average ; target = mean_ΔW / scaling
        dW = sum(wi * (sd[bk].float() @ sd[ak].float()) for wi, sd in zip(w, sds))  # [out,in]
        target = dW / scaling
        U, S, Vh = torch.linalg.svd(target, full_matrices=False)
        Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r, :]
        rootS = torch.sqrt(Sr)
        out[bk] = (Ur * rootS).to(sds[0][bk].dtype)              # [out, r]
        out[ak] = (rootS.unsqueeze(1) * Vr).to(sds[0][ak].dtype)  # [r, in]
        handled.add(ak); handled.add(bk)
    for k in sds[0]:
        if k not in handled:
            out[k] = sum(wi * sd[k].float() for wi, sd in zip(w, sds)).to(sds[0][k].dtype)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Average LoRA adapters into one merged adapter.")
    ap.add_argument("--out", required=True, help="output adapter dir")
    ap.add_argument("--adapters", nargs="+", required=True, help="2+ checkpoint dirs")
    ap.add_argument("--mode", choices=["factor", "delta"], default="factor")
    ap.add_argument("--weights", default="", help="comma floats; default = uniform")
    args = ap.parse_args()

    dirs = [Path(a) for a in args.adapters]
    sds, cfgs = zip(*[load_adapter(d) for d in dirs])
    sds, cfgs = list(sds), list(cfgs)
    assert_compatible(cfgs, sds)
    n = len(sds)
    if args.weights:
        w = [float(x) for x in args.weights.split(",")]
        assert len(w) == n, f"--weights has {len(w)} but {n} adapters"
        s = sum(w); w = [x / s for x in w]
    else:
        w = [1.0 / n] * n
    print(f"souping {n} adapters mode={args.mode} weights={[round(x,3) for x in w]}")

    merged = soup_factor(sds, w) if args.mode == "factor" else soup_delta(sds, cfgs[0], w)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out / "adapter_model.safetensors"))
    shutil.copy(dirs[0] / "adapter_config.json", out / "adapter_config.json")
    for extra in ("additional_config.json",):
        if (dirs[0] / extra).exists():
            shutil.copy(dirs[0] / extra, out / extra)
    print(f"wrote merged adapter -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
