import json, sys
pid = sys.argv[1]; RUN = sys.argv[2] if len(sys.argv) > 2 else "/workspace/runs/v17"
d = json.load(open(f"{RUN}/{pid}/multigen.json"))
cr = json.load(open(f"{RUN}/{pid}/step_candidate_review.json"))["value"]
ver = {v["path"]: v for v in cr["filter"]["verdicts"]}
for r in d["runs"]:
    if not r.get("output"):
        print(r["model_key"], "ERR", (r.get("error") or "")[:60]); continue
    cs = r["output"]["candidates"]
    def m(k):
        xs = [c[k] for c in cs if c.get(k) is not None]
        return sum(xs) / len(xs) if xs else float("nan")
    best = max(cs, key=lambda c: (c.get("ensemble") or 0))
    kept = [ver[c["path"]] for c in cs if c["path"] in ver]
    ok = sum(1 for v in kept if v["keep"] and not v["matched_must_not"])
    print(f'{r["model_key"]:20} n={len(cs)} attr={m("attr_contrast"):.2f} itm={m("itm_attrs"):.2f} pick={m("aesthetic"):.2f} ens={m("ensemble"):.2f} | best attr={best.get("attr_contrast") or 0:.2f} itm={best.get("itm_attrs") or 0:.2f} | filter {ok}/{len(kept)}')
print("iterations:", len(cr.get("iterations", [])), "| stop:", cr.get("stop_reason"), "| final:", (cr["final_path"] or "/-/-").split("/")[-2:], cr.get("final_source"))
for v in cr["filter"]["verdicts"]:
    print("  ", f'{v["path"].split("/")[-2]:18}', "keep" if v["keep"] else "drop", f'{v["score"]:+.2f}', "miss:", [a[:22] for a in v["missing_must_have"]][:3], "must_not:", [a[:22] for a in v["matched_must_not"]][:2], "ppl:", v.get("people_count"))
for it in cr.get("iterations", []):
    print("  iter", it["n"], it["plan"]["rationale"][:120], "|", it.get("note"))
