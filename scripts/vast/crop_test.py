"""Hỏi VQA trên ẢNH CẮT VÙNG NGƯỜI thay vì ảnh đầy đủ: có tách được nhóm đúng khỏi nhóm sai không?"""
import sys, os, glob, json, time
sys.path.insert(0, "/workspace/ctig17")
from PIL import Image
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents.describe import attr_question

RUN = "/workspace/runs/v191/S001"
REF = "/workspace/refs_new/reference_images_simple/selected/S001"
OUT = "/workspace/crops"
os.makedirs(OUT, exist_ok=True)
spec = json.load(open(f"{RUN}/step_spec.json"))["value"]
ent = next(e for e in spec["entities"] if e["kind"] == "object")
NAME, HAVE = ent["name_en"].split("(")[0].strip(), [a for a in ent["required_attrs_en"] if a]

SAI = ["realvis_xl_c0", "realvis_xl_c1", "sdxl_base_c1"]
DUNG = ["sdxl_base_c0", "sdxl_base_c2", "sdxl_base_c3", "sdxl_base_c5",
        "sdxl_base_rbare_c0", "sdxl_base_rbare_c1", "sdxl_base_rbare_c2", "sdxl_base_rbare_c3",
        "realvis_xl_rbare_c0", "realvis_xl_rbare_c1", "realvis_xl_rbare_c2"]
short = lambda p: os.path.basename(p).replace("S001_", "").replace("_hr.png", "")
paths = [p for p in sorted(glob.glob(f"{RUN}/*/*_hr.png")) if "revision" not in p and short(p) in SAI + DUNG]
refs = [f"{REF}/{f}" for f in sorted(os.listdir(REF))]

ag = PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"))


def person_crop(p):
    """Cắt quanh người, nới 6%, giữ tỉ lệ tối thiểu để không mất bối cảnh trang phục."""
    out = f"{OUT}/{os.path.basename(p)}"
    if os.path.exists(out):
        return out
    im = Image.open(p).convert("RGB")
    W, H = im.size
    boxes = ag.locate(p, ["person"]) or ag.locate(p, ["woman"])
    if not boxes:
        return p
    x0, y0, x1, y1 = max(boxes, key=lambda b: (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]))["bbox"]
    pw, ph = (x1 - x0) * 0.06, (y1 - y0) * 0.06
    box = (max(0, int(x0 - pw)), max(0, int(y0 - ph)), min(W, int(x1 + pw)), min(H, int(y1 + ph)))
    if (box[2] - box[0]) < 0.1 * W or (box[3] - box[1]) < 0.1 * H:
        return p
    im.crop(box).save(out)
    return out


def auc(pos, neg):
    pr = [(x, y) for x in pos for y in neg]
    return sum((x > y) + 0.5 * (x == y) for x, y in pr) / len(pr)


t0 = time.time()
crops = {p: person_crop(p) for p in paths}
rcrops = {p: person_crop(p) for p in refs}
kept = sum(1 for p, c in crops.items() if c != p)
print("cắt được %d/%d ảnh gen, %d/%d ảnh thật (%.0fs)\n" % (kept, len(paths), sum(1 for p, c in rcrops.items() if c != p), len(refs), time.time() - t0), flush=True)

for a in HAVE:
    row = {}
    for mode, src in (("đầy đủ", lambda p: p), ("cắt người", lambda p: crops.get(p, p))):
        vals = {short(p): ag.vqa_yes(attr_question(NAME, a), src(p)) for p in paths}
        rv = [ag.vqa_yes(attr_question(NAME, a), (rcrops[r] if mode == "cắt người" else r)) for r in refs]
        pos = [vals[n] for n in DUNG]
        neg = [vals[n] for n in SAI]
        row[mode] = (auc(pos, neg), sum(rv) / len(rv), min(pos), max(pos), min(neg), max(neg))
    print("=== %s" % a[:66])
    for mode, (u, rm, lo, hi, nlo, nhi) in row.items():
        print("   %-10s AUC %.2f | ảnh thật TB %.2f | ĐÚNG %.2f-%.2f | SAI %.2f-%.2f" % (mode, u, rm, lo, hi, nlo, nhi))
    print(flush=True)
print("%.0fs, %d lượt VLM" % (time.time() - t0, ag.llm.calls))
