"""So các cách tính điểm tổng trên 14 ảnh có nhãn: ngưỡng cứng so với điểm mượt, ảnh đầy đủ so với ảnh cắt người."""
import sys, os, glob, json
sys.path.insert(0, "/workspace/ctig17")
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents.describe import attr_question

RUN, REF, OUT = "/workspace/runs/v191/S001", "/workspace/refs_new/reference_images_simple/selected/S001", "/workspace/crops"
CACHE = "/workspace/score_cache.json"
spec = json.load(open(f"{RUN}/step_spec.json"))["value"]
ent = next(e for e in spec["entities"] if e["kind"] == "object")
NAME = ent["name_en"].split("(")[0].strip()
HAVE = [a for a in ent["required_attrs_en"] if a]
NOT = [a for a in ent["forbidden_attrs_en"] if a]

SAI = ["realvis_xl_c0", "realvis_xl_c1", "sdxl_base_c1"]
DUNG = ["sdxl_base_c0", "sdxl_base_c2", "sdxl_base_c3", "sdxl_base_c5",
        "sdxl_base_rbare_c0", "sdxl_base_rbare_c1", "sdxl_base_rbare_c2", "sdxl_base_rbare_c3",
        "realvis_xl_rbare_c0", "realvis_xl_rbare_c1", "realvis_xl_rbare_c2"]
short = lambda p: os.path.basename(p).replace("S001_", "").replace("_hr.png", "")
paths = [p for p in sorted(glob.glob(f"{RUN}/*/*_hr.png")) if "revision" not in p and short(p) in SAI + DUNG]
refs = [f"{REF}/{f}" for f in sorted(os.listdir(REF))]
crop = lambda p: (f"{OUT}/{os.path.basename(p)}" if os.path.exists(f"{OUT}/{os.path.basename(p)}") else p)

store = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
ag = PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"))


def v(attr, img):
    k = f"{img}||{attr}"
    if k not in store:
        store[k] = ag.vqa_yes(attr_question(NAME, attr), img)
    return store[k]


ATTRS = HAVE + NOT
data = {}
for mode, f in (("đầy đủ", lambda p: p), ("cắt người", crop)):
    data[mode] = {"ref": {a: [v(a, f(r)) for r in refs] for a in ATTRS},
                  "img": {short(p): {a: v(a, f(p)) for a in ATTRS} for p in paths}}
json.dump(store, open(CACHE, "w"))


def auc(sc):
    pos, neg = [sc[n] for n in DUNG], [sc[n] for n in SAI]
    pr = [(x, y) for x in pos for y in neg]
    return sum((x > y) + 0.5 * (x == y) for x, y in pr) / len(pr), min(pos), max(neg)


def score_all(mode, rule, lo=0.25, margin=0.08):
    d = data[mode]
    rm = {a: sum(d["ref"][a]) / len(d["ref"][a]) for a in ATTRS}
    live = [a for a in HAVE if rm[a] >= 0.50]
    w = lambda a: 2.0 if a in HAVE[:2] else 1.0
    out = {}
    for n, vals in d["img"].items():
        tot = sum(w(a) for a in live) or 1.0
        if rule == "ngưỡng cứng":
            s = sum(w(a) for a in live if vals[a] >= max(0.55, rm[a] - margin))
        else:
            s = sum(w(a) * max(0.0, min(1.0, (vals[a] - lo) / max(0.05, rm[a] - lo))) for a in live)
        pen = sum(1.0 for a in NOT if vals[a] >= max(0.70, rm[a] + 0.20))
        out[n] = (s - pen) / tot
    return out


print("%-12s %-14s %6s  %-14s %-14s" % ("ảnh", "cách tính", "AUC", "ĐÚNG thấp nhất", "SAI cao nhất"))
best = None
for mode in ("đầy đủ", "cắt người"):
    for rule in ("ngưỡng cứng", "điểm mượt"):
        sc = score_all(mode, rule)
        a, lo, hi = auc(sc)
        print("%-12s %-14s %6.2f  %-14.2f %-14.2f" % (mode, rule, a, lo, hi))
        if best is None or a > best[0]:
            best = (a, mode, rule, sc)

print("\nsố thuộc tính bị loại vì ảnh thật cũng không đạt:")
for mode in ("đầy đủ", "cắt người"):
    rm = {a: sum(data[mode]["ref"][a]) / len(data[mode]["ref"][a]) for a in ATTRS}
    print("  %-10s %s" % (mode, [(a[:30], round(rm[a], 2)) for a in HAVE if rm[a] < 0.50]))

print("\n=== tốt nhất: %s + %s (AUC %.2f) ===" % (best[1], best[2], best[0]))
for n, s in sorted(best[3].items(), key=lambda kv: -kv[1]):
    print("  %-26s %+.2f  %s" % (n, s, "SAI" if n in SAI else "ĐÚNG"))
