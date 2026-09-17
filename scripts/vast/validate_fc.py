"""So câu có/không với câu trắc nghiệm hai lựa chọn trên toàn bộ ứng viên S001 của v191 + 3 ảnh tham chiếu thật."""
import sys, os, glob, json, time
sys.path.insert(0, "/workspace/ctig17")
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents.describe import attr_question, forced_choice, pair_distractors

RUN = "/workspace/runs/v191/S001"
REF = "/workspace/refs_new/reference_images_simple/selected/S001"

spec = json.load(open(f"{RUN}/step_spec.json"))["value"]
ent = next(e for e in spec["entities"] if e["kind"] == "object")
NAME = ent["name_en"].split("(")[0].strip()
HAVE = [a for a in ent["required_attrs_en"] if a]
NOT = [a for a in ent["forbidden_attrs_en"] if a]
alts = pair_distractors(HAVE, NOT)
print("thuc the:", NAME)
print("=== ghep bang luat tu must_not ===")
for k in HAVE:
    print("  %-52s  ->  %s" % (k[:52], alts.get(k, "(khong co)")))

ag = PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"))

imgs = [(os.path.basename(p).replace("S001_", "").replace("_hr.png", ""), p)
        for p in sorted(glob.glob(f"{RUN}/*/*_hr.png")) if "revision" not in p]
imgs += [("REF-THAT " + f, f"{REF}/{f}") for f in sorted(os.listdir(REF))]

# nhãn tay cho thuộc tính "tà xẻ": ảnh đã xem bằng mắt ở kích thước đầy đủ / trong bảng liên hệ
LAB = {"sdxl_base_c1": 0, "realvis_xl_c0": 0, "realvis_xl_c1": 0,
       "REF-THAT 01.jpg": 1, "REF-THAT 02.jpg": 1, "REF-THAT 03.png": 1,
       "realvis_xl_rbare_c0": 1, "realvis_xl_rbare_c1": 1, "realvis_xl_rbare_c4": 1,
       "sdxl_base_rbare_c1": 1, "sdxl_base_rbare_c5": 1}

t0 = time.time()
for ATTR in HAVE:
    alt = alts.get(ATTR)
    print("\n=== %s ===" % ATTR, flush=True)
    print("    doi ung: %s" % (alt or "(khong co -> bo qua)"), flush=True)
    if not alt:
        continue
    rows = []
    for name, p in imgs:
        yn = ag.vqa_yes(attr_question(NAME, ATTR), p)
        fc = forced_choice(ag, NAME, p, ATTR, alt)
        rows.append((name, yn, fc))
        tag = "" if name not in LAB else ("  [nhan DUNG]" if LAB[name] else "  [nhan SAI]")
        print("    %-26s co/khong %.2f   trac nghiem %.2f%s" % (name[:26], yn, fc, tag), flush=True)
    if "panel" in ATTR or "split" in ATTR:
        pos = [r for r in rows if LAB.get(r[0]) == 1]
        neg = [r for r in rows if LAB.get(r[0]) == 0]
        for lab, i in (("co/khong  ", 1), ("trac nghiem", 2)):
            lo_pos, hi_neg = min(r[i] for r in pos), max(r[i] for r in neg)
            pairs = [(a[i], b[i]) for a in pos for b in neg]
            auc = sum((x > y) + 0.5 * (x == y) for x, y in pairs) / len(pairs)
            print("    %s: nho nhat trong nhom DUNG %.2f | lon nhat trong nhom SAI %.2f | khoang cach %+.2f | AUC %.2f"
                  % (lab, lo_pos, hi_neg, lo_pos - hi_neg, auc), flush=True)
print("\n%.0fs, %d luot goi VLM" % (time.time() - t0, ag.llm.calls))
