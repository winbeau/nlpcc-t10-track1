#!/usr/bin/env python3
"""Local error + multi-model-conflict analysis for NLPCC-2026 Task10 Track1.

Inputs (all local):
  - devbench gold:  data/devbench/dev_gold.jsonl (501), dev_gold_densematch.jsonl (206)
  - devbench preds: data/analysis/perclass_a1ce_dense_raw.jsonl (CLEAN phaseC, densematch)
                    data/analysis/devbench_raw/{s01,s02,s07,s08,s09,s10,s12,s13}.jsonl (LEAKY, full dev)
  - testp1 source:  $DATA_ROOT/data/testp1-track-1.jsonl (586, NO gold)
  - testp1 preds:   data/analysis/testp1_raw/*.jsonl (10 diverse models, per-sentence)
  - images:         data/analysis/sci/data/images/<sha>.jpg

Outputs (data/analysis/out/):
  - metrics.json          per-class P/R/F1 + confusion, clean vs leaky
  - dev_errors.jsonl      every devbench error (clean + leaky-still-wrong) w/ context
  - testp1_conflicts.jsonl per-sentence multi-model vote table + conflict tags
  - cases_dev.json        curated devbench error cases (for image root-cause)
  - cases_testp1.json     curated testp1 conflict cases (for image root-cause)
"""
import json, os, glob, collections, itertools

LABELS = ["Supported", "Unsupported Causal Mechanistic", "Unsupported Entity",
          "Scope Overgeneralization", "Contradiction"]
SHORT = {"Supported":"SUP", "Unsupported Causal Mechanistic":"UCM",
         "Unsupported Entity":"UE", "Scope Overgeneralization":"SO", "Contradiction":"CON"}
DR = os.environ.get("DATA_ROOT", "../NLPCC-2026-Task10-Science")
OUT = "data/analysis/out"
os.makedirs(OUT, exist_ok=True)

def load_raw(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        d.setdefault(r["id"], {})[r["sent_index"]] = {
            "label": r["label"], "min_lp": r.get("min_logprob"),
            "parsed_ok": r.get("parsed_ok", True), "raw": r.get("raw","")}
    return d

def load_src(path):
    """id -> {claim, sentences:[..], captions:[..], shas:[..]}"""
    d = {}
    for line in open(path):
        r = json.loads(line)
        caps, shas = [], []
        for e in r.get("evidence_bundle", []):
            ic = e.get("image_caption", [])
            if isinstance(ic, list): caps.append(" ".join(ic))
            else: caps.append(str(ic))
            p = e.get("img_path","")
            if p: shas.append(os.path.basename(p))
        sl = r["sentence_label"]
        d[r["id"]] = {
            "claim": r.get("claim_text",""),
            "sentences": [s["sentence"] for s in sl],
            "gold": [s.get("types") for s in sl],   # None for testp1
            "captions": caps, "shas": shas}
    return d

# ---------- gold-based scoring (faithful to evaluate.py) ----------
def score_set(raw, src):
    """Return per-sentence rows + confusion + per-class metrics."""
    rows = []
    conf = collections.Counter()        # (gold_primary, pred) -> n
    y_true, y_pred = [], []
    for sid, smeta in src.items():
        if sid not in raw: continue
        golds = smeta["gold"]
        for i, gtypes in enumerate(golds):
            if i not in raw[sid]: continue
            pred = raw[sid][i]["label"]
            if pred not in LABELS: pred = "Supported"   # fallback per CLAUDE.md
            correct = pred in gtypes
            true_lab = pred if correct else gtypes[0]
            y_true.append(true_lab); y_pred.append(pred)
            conf[(gtypes[0], pred)] += 1
            rows.append({"id": sid, "sent": i, "pred": pred, "gold": gtypes,
                         "gold_primary": gtypes[0], "correct": correct,
                         "min_lp": raw[sid][i]["min_lp"],
                         "text": smeta["sentences"][i]})
    # per-class P/R/F1
    per = {}
    for L in LABELS:
        tp = sum(1 for t,p in zip(y_true,y_pred) if t==L and p==L)
        fp = sum(1 for t,p in zip(y_true,y_pred) if t!=L and p==L)
        fn = sum(1 for t,p in zip(y_true,y_pred) if t==L and p!=L)
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        per[L] = {"P":round(prec,4),"R":round(rec,4),"F1":round(f1,4),
                  "tp":tp,"fp":fp,"fn":fn,"support":tp+fn}
    macro = round(sum(per[L]["F1"] for L in LABELS)/len(LABELS),4)
    acc = round(sum(1 for t,p in zip(y_true,y_pred) if t==p)/len(y_true),4) if y_true else 0
    return rows, conf, per, macro, acc

def conf_matrix_str(conf):
    lines = ["          " + " ".join(f"{SHORT[p]:>5}" for p in LABELS) + "   (rows=GOLD primary, cols=PRED)"]
    for g in LABELS:
        row = " ".join(f"{conf.get((g,p),0):>5}" for p in LABELS)
        lines.append(f"{SHORT[g]:>6} {row}")
    return "\n".join(lines)

# ============ DEVBENCH (gold-based) ============
src_dm  = load_src("data/devbench/dev_gold_densematch.jsonl")
src_dev = load_src("data/devbench/dev_gold.jsonl")
raw_clean = load_raw("data/analysis/perclass_a1ce_dense_raw.jsonl")
raw_leaky = load_raw("data/analysis/devbench_raw/s01.jsonl")

clean_rows, clean_conf, clean_per, clean_macro, clean_acc = score_set(raw_clean, src_dm)
leaky_rows, leaky_conf, leaky_per, leaky_macro, leaky_acc = score_set(raw_leaky, src_dev)

metrics = {
  "clean_phaseC_densematch": {"n_sent":len(clean_rows),"macro_f1":clean_macro,"acc":clean_acc,"per_class":clean_per},
  "leaky_s01_fulldev":       {"n_sent":len(leaky_rows),"macro_f1":leaky_macro,"acc":leaky_acc,"per_class":leaky_per},
}
json.dump(metrics, open(f"{OUT}/metrics.json","w"), indent=2, ensure_ascii=False)

print("="*70)
print("DEVBENCH GOLD-BASED ERROR ANALYSIS")
print("="*70)
print(f"\n[CLEAN phaseC model on densematch]  n_sent={len(clean_rows)} macroF1={clean_macro} acc={clean_acc}")
print(conf_matrix_str(clean_conf))
print("  per-class P / R / F1 (support):")
for L in LABELS:
    m=clean_per[L]; print(f"    {SHORT[L]:>4}  P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}  (n={m['support']}, fp={m['fp']}, fn={m['fn']})")

print(f"\n[LEAKY s01 (memorized) on full dev]  n_sent={len(leaky_rows)} macroF1={leaky_macro} acc={leaky_acc}")
print(conf_matrix_str(leaky_conf))
print("  per-class P / R / F1 (support):")
for L in LABELS:
    m=leaky_per[L]; print(f"    {SHORT[L]:>4}  P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f}  (n={m['support']}, fp={m['fp']}, fn={m['fn']})")

# dump all clean errors with context
def err_context(rows, src, tag):
    out=[]
    for r in rows:
        if r["correct"]: continue
        sm=src[r["id"]]
        out.append({"set":tag,"id":r["id"],"sent":r["sent"],"text":r["text"],
                    "pred":r["pred"],"gold":r["gold"],"min_lp":r["min_lp"],
                    "claim":sm["claim"],"captions":sm["captions"],"shas":sm["shas"]})
    return out
clean_errs = err_context(clean_rows, src_dm, "clean_densematch")
leaky_errs = err_context(leaky_rows, src_dev, "leaky_s01")
with open(f"{OUT}/dev_errors.jsonl","w") as f:
    for e in clean_errs+leaky_errs: f.write(json.dumps(e,ensure_ascii=False)+"\n")
print(f"\nclean errors: {len(clean_errs)}  |  leaky-still-wrong errors: {len(leaky_errs)}")

# error-type breakdown for clean
def err_type(r):
    p, g0 = r["pred"], r["gold"][0]
    if g0=="Supported" and p!="Supported": return "FALSE_ALARM (gold=SUP, pred=minority)"
    if g0!="Supported" and p=="Supported": return "MISSED_MINORITY (gold=minority, pred=SUP)"
    if g0!="Supported" and p!="Supported": return "WRONG_MINORITY (minority A->B)"
    return "other"
ce = collections.Counter(err_type(r) for r in clean_rows if not r["correct"])
print("\n[CLEAN error-type breakdown]")
for k,v in ce.most_common(): print(f"    {v:>4}  {k}")

# ============ TESTP1 multi-model conflict (no gold) ============
src_t1 = load_src(os.path.join(DR,"data/testp1-track-1.jsonl"))
MODEL_NAME = {
 "testp1_pce_grid_raw":"Qwen8B-pCE-grid", "testp1_q4b_raw":"Qwen4B",
 "testp1_internvl8b_cap2_raw":"InternVL-8B", "testp1_internvl2b_raw":"InternVL-2B",
 "testp1_gemma4_26b_raw":"Gemma-26B", "testp1_joint8b_raw":"Qwen8B-joint",
 "testp1_joint32b_raw":"Qwen32B-joint", "testp1_os4_b5_l0.5_raw":"Qwen8B-sm-os4",
 "testp1_perclass_b5_l0.5_raw":"Qwen8B-sm-perclass", "testp1_gentle1.5_b5_l0.5_raw":"Qwen8B-sm-gentle"}
t1_models = {}
for f in sorted(glob.glob("data/analysis/testp1_raw/*.jsonl")):
    key = os.path.basename(f)[:-6]
    t1_models[MODEL_NAME.get(key,key)] = load_raw(f)
mnames = list(t1_models.keys())
print("\n"+"="*70)
print(f"TESTP1 MULTI-MODEL CONFLICT  ({len(mnames)} models: {', '.join(mnames)})")
print("="*70)

# per-model minority-call rate
print("\n[per-model minority-call rate over all sentences]")
for m in mnames:
    tot=mino=0
    for sid in t1_models[m]:
        for i in t1_models[m][sid]:
            tot+=1
            if t1_models[m][sid][i]["label"]!="Supported": mino+=1
    print(f"    {m:20s} minority={mino:5d}/{tot} = {mino/tot:.1%}")

# build per-sentence vote table
conflicts=[]
cat_counter=collections.Counter()
for sid, sm in src_t1.items():
    for i in range(len(sm["sentences"])):
        votes={}
        for m in mnames:
            lab = t1_models[m].get(sid,{}).get(i,{}).get("label")
            if lab in LABELS: votes[m]=lab
        if len(votes)<5: continue   # need enough models
        cnt=collections.Counter(votes.values())
        distinct=set(votes.values())
        n=len(votes); top,topn=cnt.most_common(1)[0]
        minority_models=[m for m,l in votes.items() if l!="Supported"]
        minority_labels=set(l for l in votes.values() if l!="Supported")
        if distinct=={"Supported"}: cat="unanimous_SUP"
        elif len(distinct)==1: cat="unanimous_minority"
        elif "Supported" in distinct and len(minority_labels)==1: cat="SUP_vs_one_minority"
        elif "Supported" in distinct and len(minority_labels)>=2: cat="SUP_vs_multi_minority"
        elif "Supported" not in distinct: cat="minority_only_disagree"
        else: cat="other"
        cat_counter[cat]+=1
        disagreement = n - topn   # how many models differ from plurality
        conflicts.append({"id":sid,"sent":i,"text":sm["sentences"][i],
            "votes":votes,"counts":dict(cnt),"cat":cat,
            "disagreement":disagreement,"n_minority":len(minority_models),
            "distinct":sorted(distinct),"claim":sm["claim"],
            "captions":sm["captions"],"shas":sm["shas"]})
with open(f"{OUT}/testp1_conflicts.jsonl","w") as f:
    for c in conflicts: f.write(json.dumps(c,ensure_ascii=False)+"\n")

print(f"\n[conflict category counts over {len(conflicts)} scored sentences]")
for k,v in cat_counter.most_common(): print(f"    {v:>5}  {k}")

# which label-pairs co-occur in disagreements (minority disagreement structure)
pair=collections.Counter()
for c in conflicts:
    ls=set(c["votes"].values())
    for a,b in itertools.combinations(sorted(ls),2):
        pair[(SHORT[a],SHORT[b])]+=1
print("\n[most common disagreeing label-pairs across models]")
for (a,b),v in pair.most_common(12): print(f"    {v:>5}  {a} <-> {b}")

print("\nWrote:", os.listdir(OUT))
