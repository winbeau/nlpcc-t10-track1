#!/usr/bin/env python3
"""Incremental archive push to HF winbeau/nlpcc2026-task10 (private).

Uploads in ONE commit:
  - NEW LoRA adapters (inference-essential files ONLY: adapter_model.safetensors + configs;
    NOT optimizer.pt/rng/etc.) -> models/<sNN_desc>/  (the s33-s36 plain-CE retrains + the
    PHASE-C devbench A0/A1/B0/B1 that produced s30/s31/s32 + the densematch findings).
  - ALL testp1 submissions -> submissions/  (HF dedups unchanged; adds the new s20-s38).
  - ALL scripts -> scripts/  (the new phaseC_*/retrain_member/soup/etc.).
  - notes/submissions_log.md -> SUBMISSIONS.md  (the authoritative per-submission reproduce table).
  - README.md: append the PHASE C + plain-CE section (idempotent).

Run on the SERVER (adapters live there; HF token at ~/.cache/huggingface/token). Reproducible /
re-runnable: re-running re-uploads identical content as a no-op diff.

  uv run python scripts/hf_push.py            # do it
  uv run python scripts/hf_push.py --dry-run  # just list what would upload
"""
from __future__ import annotations

import glob
import os
import sys

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

RID = "winbeau/nlpcc2026-task10"

# adapter name -> checkpoint dir (server paths). adapter name = HF models/<name>/.
ADAPTERS = {
    # plain-CE full-traindev retrains -> s33-s36 (+ s37/s38 unions reuse s12/s13 already on HF)
    "s33_grid-plainCE":     "outputs/pce_grid/ckpt/v1-20260605-032646/checkpoint-4214",
    "s34_os4-plainCE":      "outputs/pce_os4/ckpt/v1-20260605-032646/checkpoint-5313",
    "s35_perclass-plainCE": "outputs/pce_perclass/ckpt/v0-20260605-032443/checkpoint-4553",
    "s36_gemma26b-plainCE": "outputs/p0_gemma4_26b_pce/v1-20260605-055519/checkpoint-479",
    # PHASE-C devbench (image-disjoint train') -> produced s30/s31/s32 + the densematch calibration
    "phaseC_A0_softmin":    "outputs/phaseC/A0_softmin_clean/ckpt/v0-20260604-191038/checkpoint-2832",
    "phaseC_A1_plainCE":    "outputs/phaseC/A1_plainCE/ckpt/v0-20260604-191037/checkpoint-2832",
    "phaseC_B0_joint":      "outputs/phaseC/B0_joint_baseline/ckpt/v0-20260604-205949/checkpoint-708",
    "phaseC_B1_density":    "outputs/phaseC/B1_density_match/ckpt/v0-20260604-211348/checkpoint-798",
}
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "additional_config.json", "args.json")

README_SECTION = """## PHASE C + plain-CE 更新 (2026-06-05)

新增 adapter(`models/`)、提交(`submissions/` s20–s38)、脚本与完整复现表(`SUBMISSIONS.md`)。

**结论(testp1 实测,均未破 s15=50.26):**
- **full-traindev + 过采样下 softmin > plain-CE**:s37(5Q plain-CE 并集)48.54 < s15 50.26;s33(grid plain-CE 单模)42.87 < s01(grid softmin)47.80。plain-CE+过采样过度开火少数类→PEM 崩;softmin 的 PEM-bottleneck loss 专门控这个 FP。
- **Gemma 进 union 净增益 ~0**(s38 48.58 ≈ s37);Gemma plain-CE 单模 s36 塌成 21.40(只 46 少数类)。
- **s01(47.80)/ s15(50.26)仍是天花板与保底。** 「densematch 离线台」只能排同条件干净模型、不能跨 regime 外推(详见 SUBMISSIONS.md 的 s30-s38 + notes/phaseC/)。

**新 adapter ↔ 提交映射(复现:加载 base Qwen3-VL-8B/26B + 对应 LoRA → infer testp1):**
| models/ | 配方 | 复现的提交 |
|---|---|---|
| s33_grid-plainCE | grid(过采样3.0/降0.6)plain-CE | s33;s37/s38 成员 |
| s34_os4-plainCE | os4(过采样4.0/降0.66)plain-CE | s34;s37/s38 成员 |
| s35_perclass-plainCE | 逐类过采样 plain-CE | s35;s37/s38 成员 |
| s36_gemma26b-plainCE | Gemma-26B plain-CE(≤2图) | s36;s38 成员 |
| phaseC_A0_softmin / A1_plainCE / B0_joint / B1_density | image-disjoint train' 离线台实验 | s30/s31/s32 + densematch 校准 |

每条提交的精确 build/train/infer/union 命令见 `SUBMISSIONS.md`(= notes/submissions_log.md)。复现脚本:`scripts/retrain_member.sh`(plain-CE 单模)、`scripts/train_gemma4.sh`(Gemma)、`scripts/phaseC_*`(devbench)、`scripts/ensemble_union.py`(并集)、`scripts/phaseC_package_testp1.sh`/`make_submission_zip.py`(打包)。
"""


def main() -> int:
    dry = "--dry-run" in sys.argv
    api = HfApi()
    ops: list[CommitOperationAdd] = []

    for name, ck in ADAPTERS.items():
        if not os.path.isdir(ck):
            print(f"  ⚠️ MISSING {name}: {ck} — skipped")
            continue
        n = 0
        for f in ADAPTER_FILES:
            p = os.path.join(ck, f)
            if os.path.exists(p):
                ops.append(CommitOperationAdd(f"models/{name}/{f}", p)); n += 1
        sz = os.path.getsize(os.path.join(ck, "adapter_model.safetensors")) / 1e6
        print(f"  adapter {name}: {n} files ({sz:.0f} MB) <- {ck}")
        assert n >= 2, f"{name}: only {n} files (need adapter_model + config)"

    for pat in ("submissions/testp1_s*.zip", "submissions/testp1_s*.jsonl"):
        for p in sorted(glob.glob(pat)):
            ops.append(CommitOperationAdd(f"submissions/{os.path.basename(p)}", p))
    for pat in ("scripts/*.py", "scripts/*.sh"):
        for p in sorted(glob.glob(pat)):
            ops.append(CommitOperationAdd(f"scripts/{os.path.basename(p)}", p))
    ops.append(CommitOperationAdd("SUBMISSIONS.md", "notes/submissions_log.md"))

    try:
        readme = open(hf_hub_download(RID, "README.md", repo_type="model")).read()
    except Exception:
        readme = "# nlpcc2026-task10\n\nNLPCC-2026 Task10 Track-1 model/submission archive.\n"
    if "## PHASE C + plain-CE 更新" not in readme:
        readme = readme.rstrip() + "\n\n" + README_SECTION
        open("/tmp/_hf_readme.md", "w").write(readme)
        ops.append(CommitOperationAdd("README.md", "/tmp/_hf_readme.md"))
        print("  README: appending PHASE C section")
    else:
        print("  README: PHASE C section already present (skip)")

    print(f"\n{'DRY-RUN: would commit' if dry else 'committing'} {len(ops)} files to {RID}")
    if dry:
        return 0
    api.create_commit(
        RID, ops, repo_type="model",
        commit_message="PHASE C + plain-CE: adapters s33-s36 + devbench A0/A1/B0/B1, submissions s20-s38, scripts, SUBMISSIONS.md",
    )
    print("DONE -> https://huggingface.co/" + RID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
