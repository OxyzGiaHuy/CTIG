import json, sys
run, pid = sys.argv[1], sys.argv[2]
cr = json.load(open(f"{run}/{pid}/step_candidate_review.json"))["value"]
for v in sorted(cr["filter"]["verdicts"], key=lambda v: -v["score"])[:8]:
    print(f'{v["path"].split("/")[-2]:18} {v["path"].split("/")[-1][:26]:26} {v["score"]:+.2f} keep={v["keep"]} vqa={cr.get("vqa",{}).get(v["path"])}')
    print("     có:", [a[:28] for a in v["matched_must_have"]])
    print("     thiếu:", [a[:28] for a in v["missing_must_have"]], "| must_not:", [a[:24] for a in v["matched_must_not"]])
    print("     VQA:", {k[:22]: p for k, p in (v.get("vqa") or {}).items()})
    print("     lý do:", [r[:60] for r in v["reasons"]][:4])
print("final:", cr["final_path"].split("/")[-2:], "| pool top:", sorted(((round(s,2), k.split("/")[-2]) for k,s in cr["pool"].items()), reverse=True)[:5])
