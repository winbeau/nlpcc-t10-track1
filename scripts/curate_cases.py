#!/usr/bin/env python3
"""Refine confusion (errors-only) + curate cases for image-based root-cause."""
import json, os, collections, random
random.seed(42)
LABELS = ["Supported","Unsupported Causal Mechanistic","Unsupported Entity",
          "Scope Overgeneralization","Contradiction"]
SHORT = {"Supported":"SUP","Unsupported Causal Mechanistic":"UCM",
         "Unsupported Entity":"UE","Scope Overgeneralization":"SO","Contradiction":"CON"}
OUT="data/analysis/out"
IMGDIR=os.path.abspath("data/analysis/sci/data/images")

def img_paths(shas):
    return [os.path.join(IMGDIR,s) for s in shas if os.path.exists(os.path.join(IMGDIR,s))]

# ---- errors-only confusion + multilabel adjacency on devbench ----
dev_errs=[json.loads(l) for l in open(f"{OUT}/dev_errors.jsonl")]
for tag in ["clean_densematch","leaky_s01"]:
    sub=[e for e in dev_errs if e["set"]==tag]
    conf=collections.Counter((e["gold"][0],e["pred"]) for e in sub)
    multilabel=sum(1 for e in sub if len(e["gold"])>1)
    print(f"\n[ERRORS-ONLY confusion: {tag}]  total_errors={len(sub)}  (of which gold is multi-label: {multilabel})")
    hdr="        "+" ".join(f"{SHORT[p]:>4}" for p in LABELS)
    print(hdr+"   rows=GOLD[0], cols=PRED")
    for g in LABELS:
        if not any(conf.get((g,p),0) for p in LABELS): continue
        print(f"  {SHORT[g]:>4}  "+" ".join(f"{conf.get((g,p),0):>4}" for p in LABELS))
    # error sub-types
    et=collections.Counter()
    for e in sub:
        g0,p=e["gold"][0],e["pred"]
        if g0=="Supported": et["FALSE_ALARM"]+=1
        elif p=="Supported": et["MISSED_MINORITY"]+=1
        else: et["WRONG_MINORITY"]+=1
    print("   subtypes:",dict(et))

# ---- curate devbench cases: ALL clean errors + interesting leaky ----
clean=[e for e in dev_errs if e["set"]=="clean_densematch"]
leaky=[e for e in dev_errs if e["set"]=="leaky_s01"]
# attach abs image paths; keep only cases with at least one image
def prep_dev(e,src_tag):
    ips=img_paths(e["shas"])
    return {"case_id":f"{src_tag}:{e['id']}:s{e['sent']}","set":e["set"],"id":e["id"],
            "sent":e["sent"],"target_sentence":e["text"],"pred":e["pred"],
            "gold":e["gold"],"gold_short":[SHORT[g] for g in e["gold"]],
            "pred_short":SHORT.get(e["pred"],e["pred"]),
            "min_lp":e["min_lp"],"claim":e["claim"],"captions":e["captions"],
            "image_paths":ips,"has_image":bool(ips)}
dev_cases=[prep_dev(e,"clean") for e in clean]
# pick leaky-still-wrong that are MISSED_MINORITY or WRONG_MINORITY (hardest, capacity present yet wrong)
leaky_hard=[e for e in leaky if not(e["gold"][0]=="Supported")]
random.shuffle(leaky_hard)
dev_cases+= [prep_dev(e,"leaky") for e in leaky_hard[:12]]
dev_cases=[c for c in dev_cases if c["has_image"]]
json.dump(dev_cases,open(f"{OUT}/cases_dev.json","w"),indent=1,ensure_ascii=False)
print(f"\ncurated devbench cases (w/ image): {len(dev_cases)}  ("
      f"clean={sum(1 for c in dev_cases if c['set']=='clean_densematch')}, "
      f"leaky={sum(1 for c in dev_cases if c['set']=='leaky_s01')})")

# ---- curate testp1 conflict cases ----
conf=[json.loads(l) for l in open(f"{OUT}/testp1_conflicts.jsonl")]
by_cat=collections.defaultdict(list)
for c in conf: by_cat[c["cat"]].append(c)
def prep_t1(c):
    ips=img_paths(c["shas"])
    # compact vote view
    vv=collections.Counter(c["votes"].values())
    return {"case_id":f"t1:{c['id']}:s{c['sent']}","id":c["id"],"sent":c["sent"],
            "target_sentence":c["text"],"cat":c["cat"],
            "disagreement":c["disagreement"],"n_minority":c["n_minority"],
            "votes":c["votes"],"vote_counts":{SHORT[k]:v for k,v in vv.items()},
            "distinct":[SHORT[x] for x in c["distinct"]],
            "claim":c["claim"],"captions":c["captions"],
            "image_paths":ips,"has_image":bool(ips)}
picks=[]
# SUP_vs_one_minority: most ambiguous (highest n_minority, i.e. near-even), spread by which minority
s1=[c for c in by_cat["SUP_vs_one_minority"] if img_paths(c["shas"])]
# bucket by the single minority label
minbuck=collections.defaultdict(list)
for c in s1:
    ml=[SHORT[l] for l in set(c["votes"].values()) if l!="Supported"][0]
    minbuck[ml].append(c)
for ml,lst in minbuck.items():
    lst.sort(key=lambda c:-c["n_minority"])
    picks+= [prep_t1(c) for c in lst[:3]]   # top-3 most-flagged per minority class
# SUP_vs_multi_minority: top by disagreement
for c in sorted([c for c in by_cat["SUP_vs_multi_minority"] if img_paths(c["shas"])],
                key=lambda c:-c["disagreement"])[:6]: picks.append(prep_t1(c))
# minority_only_disagree: all w/ image (hardest label confusion)
for c in [c for c in by_cat["minority_only_disagree"] if img_paths(c["shas"])][:5]: picks.append(prep_t1(c))
# unanimous_minority: a few (all 10 agree it's minority -> high-conf)
for c in sorted([c for c in by_cat["unanimous_minority"] if img_paths(c["shas"])],
                key=lambda c:-c["n_minority"])[:3]: picks.append(prep_t1(c))
# dedup by case_id
seen=set(); t1_cases=[]
for c in picks:
    if c["case_id"] in seen: continue
    seen.add(c["case_id"]); t1_cases.append(c)
json.dump(t1_cases,open(f"{OUT}/cases_testp1.json","w"),indent=1,ensure_ascii=False)
print(f"curated testp1 conflict cases (w/ image): {len(t1_cases)}")
cc=collections.Counter(c["cat"] for c in t1_cases)
print("  by category:",dict(cc))
print("\nTotal cases for root-cause:",len(dev_cases)+len(t1_cases))
