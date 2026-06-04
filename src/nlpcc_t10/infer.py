"""句子级批量推理（远程服务器）。

设计契约（CLAUDE.md §5 + softmin_loss_design.md §6）：

  - 输入 split:
      dev    -> 用 data/split.json 取 dev records，回 $DATA_ROOT/data/traindev-track-1.jsonl
               按原始行号取该 record（id = "track1-{idx:06d}"，与 build_dataset.py 完全一致）。
      testp1 -> $DATA_ROOT/data/testp1-track-1.jsonl（自带 id 形如 track1-p1-test-000008；
               sentence_label 只含 {"sentence": ...}，无 types）。
  - 对每个 record：构造与训练 **逐字一致** 的 prompt 前缀（system + label 定义 + evidence
    图/caption + claim_text），再对 record 内每个句子追加 "TARGET sentence: <句>" 解码 1 个 label。
    -> 直接复用 build_dataset.py 的 SYSTEM_PROMPT / build_user_content / make-sample 逻辑，
       train/infer 前缀不会漂移。
  - **KV-cache / prefix 复用**（关键，§5）：同一 record 的 system+evidence+claim 前缀对它的 n
    个句子完全相同，仅末尾 "TARGET sentence: ..." 不同。我们把一个 record 的 n 个句子作为一批
    InferRequest 一起送进引擎，并按 record 顺序处理：
      * --engine vllm（推荐）：开 enable_prefix_caching=True，vLLM 自动复用相同前缀的 KV
        （含相同图片的视觉 token），跨同一 record 的 n 句、甚至跨共享图片的 record 透明命中。
        这是在 Qwen3-VL 视觉 token 与文本 token 交错情况下唯一稳健的 prefix-reuse 方案；
        手工切片 past_key_values 在交错多模态序列里极易错位，已弃用。
      * --engine pt（默认，最稳，便于刚训完的 LoRA adapter 直接验证）：transformers 后端，
        不做跨请求 prefix cache，但按 record 成批 + max_batch_size 摊薄；输出/解析/打分完全一致。
  - 解析输出 JSON {"label": ...}；非法/越界 -> FALLBACK_LABEL（"Supported"，保 PEM，§4/§8）。
  - 输出句子级原始预测 outputs/<split>_raw.jsonl，每行：
      {"id": <record_id>, "sent_index": i, "label": "...", "raw": "<模型原文>",
       "logprob": <float|null>, "min_logprob": <float|null>, "sum_logprob": <float|null>,
       "n_tokens": <int|null>}
    （raw 便于排查解析失败；min_logprob = 全响应最弱 token 的 logprob ≈ 标签置信度，长度无关，
     段落级阈值收口的主信号；sum_logprob = log P(整条响应)；logprob 仅向后兼容，无判别力，§5）。
  - aggregate.py 负责把句子级 -> record 级 {id, labels}。

依赖（仅服务器，本地无 torch）：ms-swift 4.2.3 顶层导出 swift.{TransformersEngine,VllmEngine,
RequestConfig,InferRequest,AdapterRequest}（注意 4.2.3 没有 swift.llm / PtEngine；pt 后端 =
TransformersEngine）。本文件 import torch/swift 全部延迟到函数体，纯 argparse 可本地 import。

CLI:
  # 默认 PtEngine（transformers）：
  uv run python -m nlpcc_t10.infer --split testp1 --data-root "$DATA_ROOT" \
      --adapter outputs/qwen3vl8b_lora --out outputs/testp1_raw.jsonl

  # vLLM + 前缀缓存（吞吐更高，自动复用前缀 KV）：
  uv run python -m nlpcc_t10.infer --split dev --data-root "$DATA_ROOT" \
      --adapter outputs/qwen3vl8b_lora --engine vllm --out outputs/dev_raw.jsonl

  # dry-run（本地可跑：只构造并打印前 K 条请求，不加载模型）：
  uv run python -m nlpcc_t10.infer --split testp1 --data-root "$DATA_ROOT" \
      --adapter outputs/qwen3vl8b_lora --out /tmp/x.jsonl --dry-run --dry-run-k 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# 与 build_dataset.py 复用，保证 train/infer prompt 前缀逐字一致（纯 stdlib，本地可 import）。
try:  # 作为包运行：python -m nlpcc_t10.infer
    from .build_dataset import (
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_JOINT,
        SYSTEM_PROMPT_CORRECTOR,
        TRACK1_LABELS,
        build_user_content,
        build_user_content_joint,
        build_user_content_corrector,
        load_jsonl,
    )
except ImportError:  # 直接运行脚本时的回退
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nlpcc_t10.build_dataset import (  # type: ignore
        SYSTEM_PROMPT,
        SYSTEM_PROMPT_JOINT,
        SYSTEM_PROMPT_CORRECTOR,
        TRACK1_LABELS,
        build_user_content,
        build_user_content_joint,
        build_user_content_corrector,
        load_jsonl,
    )

FALLBACK_LABEL = "Supported"  # 解析失败/越界 -> 保 PEM 的安全默认（§4：trivial all-Supported 已是强 baseline）

# 句子级响应很短：{"label": "Scope Overgeneralization"} 远不到 24 个 token。
DEFAULT_MAX_NEW_TOKENS = 24

_LABEL_LOOKUP = {lbl.lower(): lbl for lbl in TRACK1_LABELS}


# ──────────────────────────────────────────────────────────────────────────────
# 输入：构造每个 record 的 (id, claim_text, evidence_bundle, [sentences])
# ──────────────────────────────────────────────────────────────────────────────

def iter_eval_records(
    split: str,
    data_root: Path,
) -> list[dict[str, Any]]:
    """返回待推理 record 列表，每个 {id, claim_text, evidence_bundle, sentences:[str,...]}。

    dev:    从 data/split.json 取标记为 dev 的原始行号，回 traindev-track-1.jsonl 取该行；
            id = "track1-{idx:06d}"，sentences = [s["sentence"] for s in rec["sentence_label"]]
            （与 build_dataset.make_dev_gold_record 的 id 规则完全一致）。
    testp1: 直接读 testp1-track-1.jsonl；id 用文件自带的 rec["id"]；
            sentences = [s["sentence"] for s in rec["sentence_label"]]（test 无 types）。
    """
    if split == "testp1":
        path = data_root / "data" / "testp1-track-1.jsonl"
        raw = load_jsonl(path)
        out: list[dict[str, Any]] = []
        for rec in raw:
            sents = [s.get("sentence", "") for s in rec.get("sentence_label", [])]
            out.append(
                {
                    "id": rec["id"],
                    "claim_text": rec.get("claim_text", ""),
                    "evidence_bundle": rec.get("evidence_bundle", []),
                    "sentences": sents,
                }
            )
        return out

    # split == "dev"
    split_path = Path("data/split.json")
    if not split_path.exists():
        # 允许 --data-root 旁的 data/，但优先 CWD 下 build_dataset 的输出目录
        alt = data_root.parent / "nlpcc-t10-track1" / "data" / "split.json"
        if alt.exists():
            split_path = alt
        else:
            raise FileNotFoundError(
                f"dev split 需要 data/split.json（由 build_dataset.py 生成），未找到：{split_path}"
            )
    split_map = json.loads(split_path.read_text(encoding="utf-8"))
    dev_ids = {rid for rid, where in split_map.items() if where == "dev"}

    traindev = load_jsonl(data_root / "data" / "traindev-track-1.jsonl")
    out = []
    for idx, rec in enumerate(traindev):
        rid = f"track1-{idx:06d}"
        if rid not in dev_ids:
            continue
        sents = [s.get("sentence", "") for s in rec.get("sentence_label", [])]
        out.append(
            {
                "id": rid,
                "claim_text": rec.get("claim_text", ""),
                "evidence_bundle": rec.get("evidence_bundle", []),
                "sentences": sents,
            }
        )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# prompt 构造：复用 build_dataset 的 build_user_content（去掉 assistant target）
# ──────────────────────────────────────────────────────────────────────────────

def build_messages_for_sentence(
    claim_text: str,
    target_sentence: str,
    evidence_bundle: list[dict[str, Any]],
    data_root: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """返回 (messages, image_paths)。messages 只到 user（无 assistant），交给引擎解码。

    与训练样本前缀逐字一致：system == SYSTEM_PROMPT，user == build_user_content(...)。
    """
    user_text, image_paths = build_user_content(
        claim_text, target_sentence, evidence_bundle, data_root
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return messages, image_paths


# ──────────────────────────────────────────────────────────────────────────────
# 输出解析：{"label": X} -> 合法标签；失败 -> FALLBACK_LABEL
# ──────────────────────────────────────────────────────────────────────────────

def parse_label(text: str) -> tuple[str, bool]:
    """从模型原文解析单一标签。返回 (label, parsed_ok)。

    解析顺序（鲁棒、宽容，但绝不输出非法标签）：
      1) 直接 json.loads，取 obj["label"]，标准化大小写匹配五类之一。
      2) 文本里抓 "label": "..." 正则，标准化匹配。
      3) 全文做合法标签子串扫描（取最先出现且最长匹配，避免 "Unsupported Entity" 被
         "Unsupported ..." 抢先；按标签长度降序匹配）。
      4) 全部失败 -> (FALLBACK_LABEL, False)。
    """
    if not text:
        return FALLBACK_LABEL, False
    s = text.strip()

    # 1) 严格 JSON
    obj = _try_json_obj(s)
    if obj is not None and isinstance(obj, dict) and "label" in obj:
        norm = _normalize_label(str(obj["label"]))
        if norm is not None:
            return norm, True

    # 2) 正则 "label": "..."
    m = re.search(r'"label"\s*:\s*"([^"]+)"', s)
    if m:
        norm = _normalize_label(m.group(1))
        if norm is not None:
            return norm, True

    # 3) 合法标签子串扫描（最长优先，避免前缀误匹配）
    low = s.lower()
    best: tuple[int, str] | None = None  # (起始位置, 标签)
    for lbl in sorted(TRACK1_LABELS, key=len, reverse=True):
        pos = low.find(lbl.lower())
        if pos != -1 and (best is None or pos < best[0]):
            best = (pos, lbl)
    if best is not None:
        return best[1], True

    return FALLBACK_LABEL, False


def _try_json_obj(s: str) -> Any | None:
    """尽力把字符串解析成 JSON 对象：先整体，再抓第一个 {...} 片段。"""
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except Exception:
            return None
    return None


def _normalize_label(raw: str) -> str | None:
    """大小写/空白标准化后匹配五类标签；失败返回 None。"""
    key = " ".join(raw.strip().split()).lower()
    return _LABEL_LOOKUP.get(key)


# ──────────────────────────────────────────────────────────────────────────────
# 引擎装载（仅服务器）
# ──────────────────────────────────────────────────────────────────────────────

def load_engine(
    engine: str,
    model: str,
    adapter: str,
    max_batch_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    lora_rank: int,
    template_type: str | None = None,
):
    """构造 ms-swift InferEngine。返回 (engine_obj, adapter_request_or_None, engine_kind)。

    pt:   PtEngine(model, adapters=[adapter])，LoRA 直接合并进前向；无需 adapter_request。
    vllm: VllmEngine(model, enable_lora=True, enable_prefix_caching=True, ...)，
          adapter 通过每次 infer 传 AdapterRequest 注入；前缀 KV 自动复用。
    """
    if engine == "pt":
        # ms-swift 4.2.3: the transformers backend is TransformersEngine (top-level export);
        # there is no swift.llm.PtEngine. Constructor: model positional + keyword-only
        # adapters/max_batch_size/attn_impl (verified against infer_engine/transformers_engine.py).
        from swift import TransformersEngine  # noqa: WPS433

        eng = TransformersEngine(
            model,
            adapters=[adapter] if adapter else None,
            max_batch_size=max_batch_size,
            attn_impl="sdpa",  # 推理稳妥；如装了 flash-attn 可改 "flash_attn"
            # CRITICAL: match the TRAINING template at inference. Gemma4 26B/31B default to the
            # THINKING template (gemma4) -> emits <|channel|>thought... -> JSON-label parse fails ->
            # all-Supported -> trivial ~19. Pass template_type=gemma4_nothinking (what training used).
            **({"template_type": template_type} if template_type else {}),
        )
        return eng, None, "pt"

    if engine == "vllm":
        from swift import AdapterRequest, VllmEngine  # noqa: WPS433  (4.2.3 top-level exports)

        eng = VllmEngine(
            model,
            enable_lora=bool(adapter),
            max_loras=1,
            max_lora_rank=lora_rank,
            enable_prefix_caching=True,  # 关键：跨同一 record 的 n 句复用前缀 KV
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            limit_mm_per_prompt={"image": 8},  # evidence 图数有限，给足额度
            **({"template_type": template_type} if template_type else {}),
        )
        adapter_request = (
            AdapterRequest(name="default", path=adapter) if adapter else None
        )
        return eng, adapter_request, "vllm"

    raise ValueError(f"未知 --engine: {engine}（支持 pt|vllm）")


# ──────────────────────────────────────────────────────────────────────────────
# 单 record 推理：n 句一批，按 record 顺序处理（prefix 复用友好）
# ──────────────────────────────────────────────────────────────────────────────

def infer_record(
    eng,
    adapter_request,
    engine_kind: str,
    request_config,
    record: dict[str, Any],
    data_root: Path,
):
    """对一个 record 的所有句子一次性 infer，返回 [{id, sent_index, label, raw, logprob}]。

    句子数为 0 时返回空（aggregate.py 会按 ref 句数核对/补齐）。
    """
    from swift import InferRequest  # noqa: WPS433  (4.2.3 top-level export)

    rid = record["id"]
    sentences: list[str] = record["sentences"]
    if not sentences:
        return []

    infer_requests = []
    for sent in sentences:
        messages, image_paths = build_messages_for_sentence(
            record["claim_text"], sent, record["evidence_bundle"], data_root
        )
        kw: dict[str, Any] = {"messages": messages}
        if image_paths:
            kw["images"] = image_paths
        infer_requests.append(InferRequest(**kw))

    infer_kwargs: dict[str, Any] = {}
    if engine_kind == "vllm" and adapter_request is not None:
        infer_kwargs["adapter_request"] = adapter_request

    # 同一 record 的所有句子一起送：vLLM 前缀缓存命中共享前缀；PtEngine 也按 batch 摊薄。
    responses = eng.infer(
        infer_requests, request_config, use_tqdm=False, **infer_kwargs
    )

    # Length guard: if the engine drops/adds a response, enumerate() would silently
    # shift every later sent_index (cascading mislabel with no warning). Fail safe ->
    # the whole record falls back to 'Supported' (protects PEM; review finding #4).
    if len(responses) != len(infer_requests):
        import sys as _sys
        print(
            f"[infer] WARNING id={rid}: {len(responses)} responses for "
            f"{len(infer_requests)} sentences; falling back ALL to 'Supported'.",
            file=_sys.stderr,
        )
        return [
            {"id": rid, "sent_index": i, "label": "Supported", "raw": "",
             "logprob": None, "min_logprob": None, "sum_logprob": None,
             "n_tokens": None, "parsed_ok": False}
            for i in range(len(sentences))
        ]

    rows = []
    for i, resp in enumerate(responses):
        text, conf = _extract_text_and_logprob(resp)
        label, ok = parse_label(text)
        rows.append(
            {
                "id": rid,
                "sent_index": i,
                "label": label,
                "raw": text,
                "logprob": conf["first_logprob"],   # backward-compat (first token; ~useless)
                "min_logprob": conf["min_logprob"],  # primary threshold signal (weakest token)
                "sum_logprob": conf["sum_logprob"],
                "n_tokens": conf["n_tokens"],
                "parsed_ok": ok,
            }
        )
    return rows


def _extract_text_and_logprob(resp) -> tuple[str, dict[str, float | int | None]]:
    """从 ChatCompletionResponse 取 choice0 的文本 + 一组 token 级置信度统计。

    模型输出形如 {"label": "X"}：结构 token（{ " label " : 空格 }）在贪心解码下几乎必然
    （logprob≈0），唯一不确定的是 **标签值** 的 token。所以 review #14 里只取首 token（'{'）
    的 logprob 毫无判别力。这里返回全 token 的统计，供段落级阈值收口（§5）按需选用：
      first_logprob : 首 token（旧行为，向后兼容，基本无用）
      min_logprob   : 全 token 最小 logprob —— 最弱的那个 token，通常就是标签 token；
                      **长度无关**，作为"这条预测有多虚"的主信号最稳。
      sum_logprob   : 全 token logprob 之和 = log P(整条响应) ≈ log P(标签短语)（有长度偏置）。
      n_tokens      : 解码 token 数（便于做 mean = sum/n）。
    """
    conf: dict[str, float | int | None] = {
        "first_logprob": None, "min_logprob": None, "sum_logprob": None, "n_tokens": None,
    }
    try:
        choice = resp.choices[0]
    except Exception:
        return "", conf
    text = choice.message.content if choice.message is not None else ""
    if not isinstance(text, str):
        text = str(text)
    lp = getattr(choice, "logprobs", None)
    # ms-swift logprobs 结构：{"content": [{"token":..,"logprob":float,...}, ...]}
    if isinstance(lp, dict):
        content = lp.get("content")
        if content:
            lps = [float(c["logprob"]) for c in content
                   if isinstance(c, dict) and isinstance(c.get("logprob"), (int, float))]
            if lps:
                conf["first_logprob"] = lps[0]
                conf["min_logprob"] = min(lps)
                conf["sum_logprob"] = float(sum(lps))
                conf["n_tokens"] = len(lps)
    return text, conf


# ──────────────────────────────────────────────────────────────────────────────
# PARAGRAPH-JOINT inference: ONE request per record -> {"labels": [...N...]}
# ──────────────────────────────────────────────────────────────────────────────

def build_messages_for_record_joint(claim_text, sentences, evidence_bundle, data_root):
    """Messages for a whole-paragraph joint request (system_joint + numbered sentences)."""
    user_text, image_paths = build_user_content_joint(
        claim_text, sentences, evidence_bundle, data_root
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_JOINT},
        {"role": "user", "content": user_text},
    ]
    return messages, image_paths


def parse_labels_joint(text: str, n: int) -> tuple[list[str], bool]:
    """Parse {"labels":[...]} into EXACTLY n normalized labels. JSON first, then a regex over quoted
    strings that normalize to a valid label (in order). Pad/truncate to n with Supported (PEM-safe).
    Returns (labels, ok) where ok = parsed a list of the right length with all-valid labels."""
    raw_list = None
    obj = _try_json_obj(text or "")
    if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
        raw_list = obj["labels"]
    if raw_list is None:
        found = []
        for m in re.finditer(r'"([^"]+)"', text or ""):
            nl = _normalize_label(m.group(1))
            if nl is not None:
                found.append(nl)
        raw_list = found if found else None
    if raw_list is None:
        return [FALLBACK_LABEL] * n, False
    norm = [(_normalize_label(str(x)) or FALLBACK_LABEL) for x in raw_list]
    all_valid = all(_normalize_label(str(x)) is not None for x in raw_list)
    ok = (len(norm) == n) and all_valid
    if len(norm) < n:
        norm += [FALLBACK_LABEL] * (n - len(norm))
    elif len(norm) > n:
        norm = norm[:n]
    return norm, ok


def infer_record_joint(eng, adapter_request, engine_kind, request_config, record, data_root):
    """One joint request per record; map the JSON-array response to per-sentence rows.
    Length/parse failure -> all-Supported for the record (PEM-safe, mirrors per-sentence guard)."""
    from swift import InferRequest  # noqa: WPS433

    rid = record["id"]
    sentences: list[str] = record["sentences"]
    if not sentences:
        return []
    messages, image_paths = build_messages_for_record_joint(
        record["claim_text"], sentences, record["evidence_bundle"], data_root
    )
    kw: dict[str, Any] = {"messages": messages}
    if image_paths:
        kw["images"] = image_paths
    infer_kwargs: dict[str, Any] = {}
    if engine_kind == "vllm" and adapter_request is not None:
        infer_kwargs["adapter_request"] = adapter_request

    responses = eng.infer([InferRequest(**kw)], request_config, use_tqdm=False, **infer_kwargs)
    if len(responses) != 1:
        return [{"id": rid, "sent_index": i, "label": "Supported", "raw": "", "logprob": None,
                 "min_logprob": None, "sum_logprob": None, "n_tokens": None, "parsed_ok": False}
                for i in range(len(sentences))]
    text, conf = _extract_text_and_logprob(responses[0])
    labels, ok = parse_labels_joint(text, len(sentences))
    rows = []
    for i, lab in enumerate(labels):
        rows.append({
            "id": rid, "sent_index": i, "label": lab,
            "raw": text if i == 0 else "",  # store the full joint response once (sent_index 0)
            "logprob": None,
            "min_logprob": conf["min_logprob"] if i == 0 else None,
            "sum_logprob": conf["sum_logprob"] if i == 0 else None,
            "n_tokens": conf["n_tokens"] if i == 0 else None,
            "parsed_ok": ok,
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# CORRECTOR inference: one request per record, with the ensemble's candidate labels in the prompt
# ──────────────────────────────────────────────────────────────────────────────

def infer_record_corrector(eng, adapter_request, engine_kind, request_config, record, data_root, candidates_map):
    """Like joint inference but the prompt carries the ensemble's INITIAL label per sentence; the
    corrector outputs the corrected label array. candidates_map: {id -> [labels]}."""
    from swift import InferRequest  # noqa: WPS433

    rid = record["id"]
    sentences: list[str] = record["sentences"]
    if not sentences:
        return []
    cands = candidates_map.get(rid, ["Supported"] * len(sentences))
    if len(cands) != len(sentences):
        cands = (cands + ["Supported"] * len(sentences))[:len(sentences)]
    user_text, image_paths = build_user_content_corrector(
        record["claim_text"], sentences, cands, record["evidence_bundle"], data_root
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CORRECTOR},
        {"role": "user", "content": user_text},
    ]
    kw: dict[str, Any] = {"messages": messages}
    if image_paths:
        kw["images"] = image_paths
    infer_kwargs: dict[str, Any] = {}
    if engine_kind == "vllm" and adapter_request is not None:
        infer_kwargs["adapter_request"] = adapter_request
    responses = eng.infer([InferRequest(**kw)], request_config, use_tqdm=False, **infer_kwargs)
    if len(responses) != 1:
        return [{"id": rid, "sent_index": i, "label": "Supported", "raw": "", "logprob": None,
                 "min_logprob": None, "sum_logprob": None, "n_tokens": None, "parsed_ok": False}
                for i in range(len(sentences))]
    text, conf = _extract_text_and_logprob(responses[0])
    labels, ok = parse_labels_joint(text, len(sentences))
    rows = []
    for i, lab in enumerate(labels):
        rows.append({"id": rid, "sent_index": i, "label": lab, "raw": text if i == 0 else "",
                     "logprob": None, "min_logprob": conf["min_logprob"] if i == 0 else None,
                     "sum_logprob": conf["sum_logprob"] if i == 0 else None,
                     "n_tokens": conf["n_tokens"] if i == 0 else None, "parsed_ok": ok})
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# dry-run（本地、无 torch）：只构造请求并打印，验证 prompt/分组/输出契约
# ──────────────────────────────────────────────────────────────────────────────

def dry_run(records: list[dict[str, Any]], data_root: Path, k: int) -> int:
    """不加载模型：对前 k 个 record 构造 messages 并打印摘要，验证 prefix 一致性与分组。"""
    print(f"[dry-run] {len(records)} records；展示前 {k} 个的请求构造：")
    shown = 0
    total_sent = 0
    for rec in records:
        total_sent += len(rec["sentences"])
    for rec in records[:k]:
        rid = rec["id"]
        n = len(rec["sentences"])
        first_msgs, imgs = build_messages_for_sentence(
            rec["claim_text"],
            rec["sentences"][0] if rec["sentences"] else "",
            rec["evidence_bundle"],
            data_root,
        )
        user_text = first_msgs[1]["content"]
        print("=" * 70)
        print(f"id={rid}  n_sentences={n}  n_images={len(imgs)}")
        print(f"  images: {imgs}")
        print(f"  system[:60]: {first_msgs[0]['content'][:60]!r}")
        print(f"  user[:300]:\n{user_text[:300]}")
        # 解析自检：模拟模型输出
        for sample_out in ('{"label": "Supported"}', "garbage", 'the answer is Contradiction'):
            lbl, ok = parse_label(sample_out)
            print(f"  parse_label({sample_out!r}) -> {lbl!r} ok={ok}")
        shown += 1
    print("=" * 70)
    print(f"[dry-run] 共 {total_sent} 个句子级请求将被解码（{len(records)} records）。")
    print("[dry-run] 未加载模型；真实推理请去掉 --dry-run（仅服务器、需 torch+swift）。")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# CLI / MAIN
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentence-level inference for Track 1.")
    p.add_argument("--split", choices=("dev", "testp1"), required=True)
    p.add_argument("--data-root", default="../NLPCC-2026-Task10-Science")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="基座模型 Hub id 或本地路径（须与训练一致）。",
    )
    p.add_argument("--adapter", default="outputs/qwen3vl8b_lora", help="LoRA adapter 目录")
    p.add_argument(
        "--template",
        default=None,
        help="Override the ms-swift template_type at inference (must MATCH training). "
             "REQUIRED for Gemma4 26B/31B: pass 'gemma4_nothinking' or the THINKING template leaks "
             "<|channel|>thought tokens -> JSON parse fails -> all-Supported. None = model default.",
    )
    p.add_argument("--out", required=True, help="输出句子级 raw jsonl 路径")
    p.add_argument(
        "--engine",
        choices=("pt", "vllm"),
        default="pt",
        help="pt=transformers（默认，最稳）；vllm=开 prefix-caching 复用前缀 KV（吞吐更高）。",
    )
    p.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="PtEngine 批大小（0/1=不限）；一个 record 的 n 句会一并送入。",
    )
    p.add_argument("--max-model-len", type=int, default=4096, help="vLLM max_model_len")
    p.add_argument("--gpu-mem-util", type=float, default=0.9, help="vLLM gpu_memory_utilization")
    p.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="vLLM max_lora_rank，须 >= 训练 lora_rank（configs 默认 16）。",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="采样温度。0.0=贪心(默认,行为不变)。E1 自洽采样用 >0(如 0.7);"
             "多样本由调用方多次跑(不同 --seed)再 union，而非引擎 n>1。",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="采样种子(仅 temperature>0 时透传引擎,使一次采样可复现)。",
    )
    p.add_argument(
        "--joint",
        action="store_true",
        help="PARAGRAPH-JOINT inference: one request per record -> JSON array of N labels "
             "(must match a model trained with build_dataset --joint). max_tokens auto-bumped to 512.",
    )
    p.add_argument(
        "--corrector",
        action="store_true",
        help="CORRECTOR inference: prompt carries the --candidates initial labels per sentence; "
             "outputs the corrected JSON array (model trained on build_corrector_data.py output).",
    )
    p.add_argument(
        "--candidates",
        default=None,
        help="ensemble predictions {id,labels} jsonl — REQUIRED for --corrector (the initial labels).",
    )
    p.add_argument(
        "--no-logprob",
        action="store_true",
        help="关闭 logprob 解码（更快；段落级阈值收口将无置信度可用）。",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只推理前 N 个 record（冒烟测试）。",
    )
    p.add_argument("--dry-run", action="store_true", help="不加载模型，只构造并打印请求（本地可跑）。")
    p.add_argument("--dry-run-k", type=int, default=2, help="dry-run 展示的 record 数。")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    data_root = Path(args.data_root).resolve()

    records = iter_eval_records(args.split, data_root)
    if args.limit is not None:
        records = records[: args.limit]
    print(f"split={args.split}  records={len(records)}  "
          f"sentences={sum(len(r['sentences']) for r in records)}")

    if args.dry_run:
        return dry_run(records, data_root, args.dry_run_k)

    # ---- 以下仅服务器（需要 torch + ms-swift）----
    from swift import RequestConfig  # noqa: WPS433  (4.2.3 top-level export)

    eng, adapter_request, engine_kind = load_engine(
        engine=args.engine,
        model=args.model,
        adapter=args.adapter,
        max_batch_size=args.max_batch_size,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
        lora_rank=args.lora_rank,
        template_type=args.template,
    )

    # Joint mode emits up to N labels (testp1 max 31 sentences) -> bigger budget than the
    # per-sentence 24-token default; 512 covers 31 labels + JSON structure.
    # Default (temperature=0.0) is greedy = unchanged. E1 self-consistency passes temperature>0;
    # `seed` is only attached when sampling so the greedy path stays byte-identical (and we don't
    # depend on RequestConfig accepting `seed` for normal runs).
    rc_kwargs: dict[str, Any] = dict(
        max_tokens=(512 if (args.joint or args.corrector) else args.max_new_tokens),
        temperature=args.temperature,
        logprobs=not args.no_logprob,
        top_logprobs=1 if not args.no_logprob else None,
    )
    if args.temperature and args.temperature > 0 and args.seed is not None:
        rc_kwargs["seed"] = args.seed
    request_config = RequestConfig(**rc_kwargs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_fallback = 0
    label_counter: dict[str, int] = {lbl: 0 for lbl in TRACK1_LABELS}
    with out_path.open("w", encoding="utf-8") as f:
        if args.corrector:
            if not args.candidates:
                print("error: --corrector requires --candidates", file=sys.stderr)
                return 2
            cmap = {r["id"]: r["labels"] for r in (json.loads(line) for line in
                    Path(args.candidates).read_text(encoding="utf-8").splitlines() if line.strip())}
            print(f"[corrector] loaded candidates for {len(cmap)} records from {args.candidates}")

            def _infer(eng, ar, ek, rc, rec, dr):
                return infer_record_corrector(eng, ar, ek, rc, rec, dr, cmap)
        elif args.joint:
            _infer = infer_record_joint
        else:
            _infer = infer_record
        for ri, rec in enumerate(records):
            rows = _infer(
                eng, adapter_request, engine_kind, request_config, rec, data_root
            )
            for row in rows:
                if not row.get("parsed_ok", True):
                    n_fallback += 1
                label_counter[row["label"]] = label_counter.get(row["label"], 0) + 1
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 50 == 0 or n_done == len(records):
                print(f"  [{n_done}/{len(records)}] records done "
                      f"(fallback so far: {n_fallback})")

    print(f"Wrote {out_path}")
    print(f"  records={n_done}  parse_fallbacks={n_fallback}")
    print("  label distribution:")
    for lbl in TRACK1_LABELS:
        print(f"    {lbl}: {label_counter.get(lbl, 0)}")
    print("Next: uv run python -m nlpcc_t10.aggregate "
          f"--pred {out_path} --ref <ref jsonl> --out outputs/{args.split}_submission.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
