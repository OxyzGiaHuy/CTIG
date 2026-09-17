"""Hiệu chỉnh ngưỡng nên dựa trên mấy ảnh thật? So 3 ảnh 'selected' với 3+20 ảnh 'candidates'."""
import sys, os, glob, json, statistics as st
sys.path.insert(0, "/workspace/ctig17")
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents.describe import attr_question

RUN = "/workspace/runs/v191/S001"
BASE = "/workspace/refs_new/reference_images_simple"
sel = [f"{BASE}/selected/S001/{f}" for f in sorted(os.listdir(f"{BASE}/selected/S001"))]
can = [f"{BASE}/candidates/S001/{f}" for f in sorted(os.listdir(f"{BASE}/candidates/S001"))]
spec = json.load(open(f"{RUN}/step_spec.json"))["value"]
ent = next(e for e in spec["entities"] if e["kind"] == "object")
NAME, HAVE = ent["name_en"].split("(")[0].strip(), [a for a in ent["required_attrs_en"] if a]

LAB = {"sdxl_base_c1": 0, "realvis_xl_c0": 0, "realvis_xl_c1": 0,
       "realvis_xl_rbare_c0": 1, "realvis_xl_rbare_c1": 1, "realvis_xl_rbare_c4": 1,
       "sdxl_base_rbare_c1": 1, "sdxl_base_rbare_c5": 1}
gen = [(os.path.basename(p).replace("S001_", "").replace("_hr.png", ""), p)
       for p in sorted(glob.glob(f"{RUN}/*/*_hr.png")) if "revision" not in p]
gen = [(n, p) for n, p in gen if n in LAB]

ag = PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"))
print("selected %d, candidates %d, ảnh gen có nhãn %d\n" % (len(sel), len(can), len(gen)), flush=True)

for a in HAVE:
    vs = [ag.vqa_yes(attr_question(NAME, a), p) for p in sel]
    vc = [ag.vqa_yes(attr_question(NAME, a), p) for p in can]
    vall = vs + vc
    g = [(n, ag.vqa_yes(attr_question(NAME, a), p), LAB[n]) for n, p in gen]
    pos = [v for _, v, l in g if l == 1]
    neg = [v for _, v, l in g if l == 0]
    pr = [(x, y) for x in pos for y in neg]
    auc = sum((x > y) + 0.5 * (x == y) for x, y in pr) / len(pr)
    print("=== %s" % a[:70], flush=True)
    print("   selected(3)  mean %.2f  median %.2f  min %.2f" % (st.mean(vs), st.median(vs), min(vs)))
    print("   sel+cand(%d) mean %.2f  median %.2f  p30 %.2f  min %.2f"
          % (len(vall), st.mean(vall), st.median(vall), sorted(vall)[int(.3 * len(vall))], min(vall)))
    print("   ảnh gen: ĐÚNG %s | SAI %s | AUC %.2f"
          % (["%.2f" % v for v in sorted(pos)], ["%.2f" % v for v in sorted(neg)], auc))
    for lab, thr in (("sel mean-0.08 ", st.mean(vs) - 0.08),
                     ("all mean-0.08 ", st.mean(vall) - 0.08),
                     ("all median-.08", st.median(vall) - 0.08),
                     ("all p30       ", sorted(vall)[int(.3 * len(vall))])):
        thr = max(0.55, min(0.90, thr))
        tp, fn = sum(v >= thr for v in pos), sum(v < thr for v in pos)
        fp, tn = sum(v >= thr for v in neg), sum(v < thr for v in neg)
        print("     %s thr %.2f -> giữ đúng %d/%d, loại sai %d/%d" % (lab, thr, tp, tp + fn, tn, tn + fp))
    print(flush=True)
