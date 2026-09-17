"""Quét biên độ hiệu chỉnh: chạy Reviewer thật một lần, nhớ mọi câu trả lời VLM, rồi phát lại với từng biên độ."""
import sys, os, glob, json, time
sys.path.insert(0, "/workspace/ctig17")
from ctig.llm.qwen_vl import QwenVLBackend
from ctig.llm.prompt_agent import PromptAgent
from ctig.agents import describe as D
from ctig.schema import from_dict, CulturalSpec

RUN = "/workspace/runs/v191/S001"
REF = "/workspace/refs_new/reference_images_simple/selected/S001"
CACHE = "/workspace/sweep_cache.json"
refs = [f"{REF}/{f}" for f in sorted(os.listdir(REF))]
spec = from_dict(CulturalSpec, json.load(open(f"{RUN}/step_spec.json"))["value"])
pe = json.load(open(f"{RUN}/step_analysis.json"))["value"].get("prompt_en", "")
paths = [p for p in sorted(glob.glob(f"{RUN}/*/*_hr.png")) if "revision" not in p]

# nhãn tay cho "tà xẻ", chỉ ghi ảnh đã nhìn rõ ở kích thước đủ lớn
SAI = ["realvis_xl_c0", "realvis_xl_c1", "sdxl_base_c1"]
DUNG = ["sdxl_base_c0", "sdxl_base_c2", "sdxl_base_c3", "sdxl_base_c5",
        "sdxl_base_rbare_c0", "sdxl_base_rbare_c1", "sdxl_base_rbare_c2", "sdxl_base_rbare_c3",
        "realvis_xl_rbare_c0", "realvis_xl_rbare_c1", "realvis_xl_rbare_c2"]
LAB = {n: 0 for n in SAI} | {n: 1 for n in DUNG}
short = lambda p: os.path.basename(p).replace("S001_", "").replace("_hr.png", "")


class Caching:
    """Bọc agent thật, nhớ mọi câu hỏi VQA và mọi mô tả ảnh để phát lại không tốn GPU."""
    def __init__(self, inner, store):
        self.inner, self.s = inner, store
        self.s.setdefault("vqa", {}); self.s.setdefault("desc", {}); self.s.setdefault("match", {})
        self.misses = 0

    def vqa_yes(self, q, img):
        k = f"{img}||{q}"
        if k not in self.s["vqa"]:
            self.misses += 1
            self.s["vqa"][k] = self.inner.vqa_yes(q, img)
        return self.s["vqa"][k]

    def describe_image(self, path):
        if path not in self.s["desc"]:
            self.misses += 1
            self.s["desc"][path] = self.inner.describe_image(path)
        return self.s["desc"][path]

    def match_descriptors(self, text, have, notd):
        k = json.dumps([text, have, notd], ensure_ascii=False)
        if k not in self.s["match"]:
            self.misses += 1
            self.s["match"][k] = self.inner.match_descriptors(text, have, notd)
        return self.s["match"][k]

    def __getattr__(self, n):
        return getattr(self.inner, n)


store = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
ag = Caching(PromptAgent(QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0")), store)
t0 = time.time()

quiet = lambda *a, **k: None
res0 = D.run(ag, paths, spec, pe, kind="candidate", log=quiet, clip=None, refs=refs)
json.dump(store, open(CACHE, "w"))
print("chạy thật xong %.0fs, %d lượt gọi mới\n" % (time.time() - t0, ag.misses), flush=True)


def auc_of(verdicts):
    sc = {short(v.path): v.score for v in verdicts}
    pos = [sc[n] for n in DUNG if n in sc]
    neg = [sc[n] for n in SAI if n in sc]
    pr = [(x, y) for x in pos for y in neg]
    return (sum((x > y) + 0.5 * (x == y) for x, y in pr) / len(pr), min(pos), max(pos), min(neg), max(neg),
            sum(1 for n in SAI if sc.get(n, 0) >= min(pos)))


print("%-7s %-7s %5s %-18s %-18s %s" % ("biên", "AUC", "bỏ", "nhóm ĐÚNG", "nhóm SAI", "số ảnh SAI lọt vào dải ĐÚNG"))
for margin in (0.04, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30):
    D._CAL_MARGIN = margin
    D._CAL_CACHE.clear()
    r = D.run(ag, paths, spec, pe, kind="candidate", log=quiet, clip=None, refs=refs)
    a, lo, hi, nlo, nhi, overlap = auc_of(r.verdicts)
    dead = r.verdicts[0].unverifiable
    print("%-7.2f %-7.2f %5d %-18s %-18s %d"
          % (margin, a, len(dead), "%.2f - %.2f" % (lo, hi), "%.2f - %.2f" % (nlo, nhi), overlap), flush=True)

print("\nvới biên tốt nhất, điểm từng ảnh:")
D._CAL_MARGIN = 0.08
D._CAL_CACHE.clear()
r = D.run(ag, paths, spec, pe, kind="candidate", log=print, clip=None, refs=refs)
for v in sorted(r.verdicts, key=lambda v: -v.score):
    n = short(v.path)
    print("  %-26s %+.2f  %-6s thiếu=%s" % (n[:26], v.score,
          {0: "SAI", 1: "ĐÚNG"}.get(LAB.get(n), ""), [a[:24] for a in v.missing_must_have]))
json.dump(store, open(CACHE, "w"))
