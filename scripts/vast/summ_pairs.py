import json, sys, glob, os
run = sys.argv[1]
def base_of(k): return k.split("+")[0].split("#")[0].split("@")[0]
def is_bare(k): return "#bare" in k
rows_all = {}
for mg in sorted(glob.glob(f"{run}/*/multigen.json")):
    pid = mg.split("/")[-2]
    d = json.load(open(mg))
    crp = f"{run}/{pid}/step_candidate_review.json"
    cr = json.load(open(crp))["value"] if os.path.exists(crp) else None
    ver = {v["path"]: v for v in cr["filter"]["verdicts"]} if cr else {}
    agg = {}
    for r in d["runs"]:
        if not r.get("output"): continue
        b = base_of(r["model_key"]); lab = "bare" if is_bare(r["model_key"]) else "system"
        a = agg.setdefault((b, lab), {"attr": [], "itm": [], "rev": [], "pass": 0, "n": 0, "seen": set()})
        for c in r["output"]["candidates"]:
            if c["path"] in a["seen"]: continue
            a["seen"].add(c["path"]); a["n"] += 1
            if c.get("attr_contrast") is not None: a["attr"].append(c["attr_contrast"])
            if c.get("itm_attrs") is not None: a["itm"].append(c["itm_attrs"])
            v = ver.get(c["path"])
            if v: a["rev"].append(v["score"]); a["pass"] += int(v["keep"] and not v["matched_must_not"] and v["score"] >= 0.5)
    m = lambda xs: sum(xs)/len(xs) if xs else float("nan")
    print(f"\n== {pid}  final={(cr['final_path'] or '/-').split('/')[-2] if cr else '-'} ({cr.get('final_source') if cr else ''}) iters={len(cr.get('iterations',[])) if cr else 0}")
    print(f"{'model':14}{'nhánh':8}{'n':>3}{'attr':>7}{'itm':>7}{'rev':>7}{'đạt':>7}")
    for b in dict.fromkeys(base_of(r["model_key"]) for r in d["runs"]):
        for lab in ("bare", "system"):
            a = agg.get((b, lab))
            if not a: continue
            print(f"{b:14}{lab:8}{a['n']:>3}{m(a['attr']):>7.2f}{m(a['itm']):>7.2f}{m(a['rev']):>7.2f}{a['pass']:>4}/{len(a['rev']):<2}")
            rows_all.setdefault((b, lab), []).append((m(a['attr']), m(a['itm']), m(a['rev'])))
print("\n== TRUNG BÌNH qua các prompt")
import math
for (b, lab), xs in sorted(rows_all.items()):
    f = lambda i: sum(x[i] for x in xs if not math.isnan(x[i]))/max(1,len([x for x in xs if not math.isnan(x[i])]))
    print(f"{b:14}{lab:8} attr={f(0):.2f} itm={f(1):.2f} rev={f(2):+.2f}  ({len(xs)} prompt)")
