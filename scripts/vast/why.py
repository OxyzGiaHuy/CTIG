import json, sys
run, pid, model = sys.argv[1], sys.argv[2], sys.argv[3]
cr = json.load(open(f"{run}/{pid}/step_candidate_review.json"))["value"]
per = {x["base_model"]: x for x in cr.get("per_model", [])}
ver = {v["path"]: v for v in cr["filter"]["verdicts"]}
spec = json.load(open(f"{run}/{pid}/step_spec.json"))["value"]
print("must_have chấm:", [a for e in spec["entities"] for a in e["required_attrs_en"]])
print("must_not chấm:", [a for e in spec["entities"] for a in e["forbidden_attrs_en"]])
x = per.get(model)
if not x:
    print("không có", model, list(per)); raise SystemExit
print("\nẢNH CUỐI:", x["final_path"].split("/")[-1], "|", x["final_source"], "| vòng:", len(x["iterations"]))
print("\nPool (điểm Reviewer):")
for pth, sc in sorted(x["pool"].items(), key=lambda kv: -kv[1])[:8]:
    v = ver.get(pth, {})
    row = pth.split("/")[-2]
    print("  %-24s %-30s %+.2f  có=%d thiếu=%s must_not=%s" % (
        row, pth.split("/")[-1][:30], sc, len(v.get("matched_must_have", [])),
        [a[:22] for a in v.get("missing_must_have", [])], [a[:20] for a in v.get("matched_must_not", [])]))
print("\nẢnh BARE của cùng model (Reviewer chấm):")
for pth, v in sorted(ver.items(), key=lambda kv: -kv[1]["score"]):
    if "rbare" in pth and model.split("_")[0] in pth:
        print("  %-30s %+.2f có=%s thiếu=%s" % (pth.split("/")[-1][:30], v["score"],
              [a[:22] for a in v["matched_must_have"]], [a[:22] for a in v["missing_must_have"]]))
