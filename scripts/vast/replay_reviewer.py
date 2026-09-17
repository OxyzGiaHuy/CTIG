"""Chạy lại Reviewer v1.9.3 trên chính các ứng viên S001 của v191, so điểm với bản đã lưu."""
import sys, os, glob, json, time
sys.path.insert(0, "/workspace/ctig17")
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents import describe as D
from ctig.schema import from_dict, CulturalSpec

RUN = "/workspace/runs/v191/S001"
REF = "/workspace/refs_new/reference_images_simple/selected/S001"
refs = [f"{REF}/{f}" for f in sorted(os.listdir(REF))]
spec = from_dict(CulturalSpec, json.load(open(f"{RUN}/step_spec.json"))["value"])
old = {v["path"]: v for v in json.load(open(f"{RUN}/step_candidate_review.json"))["value"]["filter"]["verdicts"]}
pe = json.load(open(f"{RUN}/step_analysis.json"))["value"].get("prompt_en", "")

paths = [p for p in sorted(glob.glob(f"{RUN}/*/*_hr.png")) if "revision" not in p]
ag = PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"))
t0 = time.time()
res = D.run(ag, paths, spec, pe, kind="candidate", log=print, clip=None, refs=refs)

LAB = {"sdxl_base_c1": "SAI (áo liền quần)", "realvis_xl_c0": "SAI (áo ngắn)", "realvis_xl_c1": "SAI (áo ngắn)",
       "realvis_xl_rbare_c0": "DUNG", "realvis_xl_rbare_c1": "DUNG", "realvis_xl_rbare_c4": "DUNG",
       "sdxl_base_rbare_c1": "DUNG", "sdxl_base_rbare_c5": "DUNG"}
print("\n%-26s %8s %8s   %s" % ("ảnh", "v191", "v1.9.3", "nhãn tay / thiếu"))
rows = []
for v in sorted(res.verdicts, key=lambda v: -v.score):
    n = os.path.basename(v.path).replace("S001_", "").replace("_hr.png", "")
    o = old.get(v.path, {}).get("score")
    rows.append((n, o, v.score, LAB.get(n, "")))
    print("%-26s %+8s %+8.2f   %-20s %s" % (n[:26], ("%.2f" % o) if o is not None else "-", v.score,
          LAB.get(n, ""), "; ".join(a[:26] for a in v.missing_must_have)))

lab = [(n, o, s, l) for n, o, s, l in rows if l]
for name, i in (("v191 (ngưỡng cố định)", 1), ("v1.9.3 (hiệu chỉnh)", 2)):
    pos = [r[i] for r in lab if r[3] == "DUNG"]
    neg = [r[i] for r in lab if r[3].startswith("SAI")]
    if any(x is None for x in pos + neg):
        print("\n%-24s: thiếu điểm cũ" % name); continue
    pr = [(a, b) for a in pos for b in neg]
    auc = sum((x > y) + 0.5 * (x == y) for x, y in pr) / len(pr)
    print("\n%-24s: nhóm DUNG %.2f-%.2f | nhóm SAI %.2f-%.2f | AUC %.2f"
          % (name, min(pos), max(pos), min(neg), max(neg), auc))
print("\n%.0fs" % (time.time() - t0))
