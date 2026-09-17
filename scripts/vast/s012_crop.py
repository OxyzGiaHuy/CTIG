import sys, os, json
sys.path.insert(0, "/workspace/ctig17")
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents import describe as D
from ctig.schema import from_dict, CulturalSpec

for pid, refdir in (("S012", "/workspace/refs_new/reference_images_simple/selected/S012"),
                    ("S001", "/workspace/refs_new/reference_images_simple/selected/S001")):
    spec = from_dict(CulturalSpec, json.load(open(f"/workspace/runs/v193/{pid}/step_spec.json"))["value"])
    refs = [f"{refdir}/{f}" for f in sorted(os.listdir(refdir))]
    ag = PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0")) if pid == "S012" else ag
    print("===", pid, "| nhãn cắt:", D.subject_labels(spec))
    for lab, labels in (("chỉ người", ["person"]), ("theo thực thể", D.subject_labels(spec))):
        D._CAL_CACHE.clear(); D._CROP_MEMO.clear()
        orig = D.subject_labels
        D.subject_labels = lambda s, _l=labels: _l
        cal = D.calibrate(ag, spec, refs, log=lambda *a: None)
        D.subject_labels = orig
        live = [a for a, c in cal.items() if c["side"] == "have" and c["checkable"]]
        dead = [a for a, c in cal.items() if c["side"] == "have" and not c["checkable"]]
        print("  %-14s kiểm được %d: %s" % (lab, len(live), [a[:30] for a in live]))
        print("  %-14s bỏ        %d: %s" % ("", len(dead), [(a[:30], cal[a]["ref_mean"], cal[a]["ref_spread"]) for a in dead]))
