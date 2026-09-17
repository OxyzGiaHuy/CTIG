import json, sys
run, pid = sys.argv[1], sys.argv[2]
only = sys.argv[3] if len(sys.argv) > 3 else ""
cr = json.load(open(f"{run}/{pid}/step_candidate_review.json"))["value"]
ver = {v["path"]: v for v in cr["filter"]["verdicts"]}
print("tổng verdict:", len(ver), "| pass:", sum(1 for v in ver.values() if v.get("ok")))
for x in (cr.get("per_model") or []):
    m = x["base_model"]
    if only and only not in m: continue
    fp = x["final_path"]
    print("\n=== %s === cuối=%s (%s) vòng=%d stop=%s" % (m, fp.split("/")[-1], x["final_source"], len(x["iterations"]), x.get("stop_reason")))
    v = ver.get(fp)
    if not v:
        print("  (không có verdict cho ảnh cuối)"); continue
    print("  score %+.2f ok=%s" % (v["score"], v.get("ok")))
    print("  matched:", v.get("matched_must_have"))
    print("  missing:", v.get("missing_must_have"))
    print("  must_not:", v.get("matched_must_not"))
    vq, vn = v.get("vqa") or {}, v.get("vqa_neg") or {}
    for k in sorted(set(list(vq) + list(vn))):
        print("    VQA %-52s pos=%.2f neg=%s" % (k[:52], vq.get(k, -1), ("%.2f" % vn[k]) if k in vn else "-"))
    print("  notes:", [n[:110] for n in (v.get("notes") or [])][:8])
    print("  desc:", (v.get("description") or "")[:400])
