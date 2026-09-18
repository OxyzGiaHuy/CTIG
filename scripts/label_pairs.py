"""Trang gán nhãn SO CẶP cho bốn nhánh B / T / S / M: mỗi màn hình hai ảnh sinh cạnh nhau, bên trên là ảnh THẬT.

    python scripts/label_pairs.py <run_dir> [run_dir2 ...] -o pairs.html \
        [--pairs M-B,M-T,M-S] [--annotator huy] [--seed 0] [--max-pairs 120] [--refs 3] \
        [--ref-dir /duong/dan_simple,/duong/dan_complex] [--config configs/vast_arms.yaml] \
        [--n-eval 5] [--ids S001,S012]

VÌ SAO VIẾT MỚI THAY VÌ SỬA `scripts/label_tool.py` — đây là lý do quan trọng nhất của cả file này.
Trang cũ hỏi mỗi ảnh một câu tổng thể: *"Tổng thể: ảnh này có đúng là {áo dài} không?"*. Đó là câu hỏi
ĐỊNH DANH THỰC THỂ, không phải câu hỏi về tính đúng văn hoá. Lô nhãn ngày 2026-09-17 lộ ngay chỗ hỏng:
cả bốn ảnh mà mắt người thấy sai văn hoá (hoa văn kiểu Trung Quốc, cổ áo sai, tà áo sai) **đều vẫn là áo
dài**, nên người gán buộc phải trả lời "đúng" — bộ nhãn thu được không phân biệt nổi ảnh tốt với ảnh tệ,
và mọi thứ đo trên nó đều vô nghĩa. Nguyên nhân gốc: hỏi một ảnh đơn lẻ theo thang tuyệt đối thì người gán
không có mốc để neo, nên tự hạ tiêu chuẩn về mức "có phải thứ đó không".

Trang này đổi sang thiết kế 2AFC (buộc chọn một trong hai) có NEO:
  * mốc neo là ẢNH THẬT của đúng prompt ấy, hiện ngay bên trên — người gán so hoa văn, dáng, bối cảnh với
    ảnh thật chứ không so với ý niệm mơ hồ trong đầu;
  * câu hỏi là câu SO SÁNH ("ảnh nào giống văn hoá Việt Nam hơn"), nên "cả hai đều là áo dài" không còn là
    câu trả lời hợp lệ; người gán bắt buộc phải nhìn tới chi tiết mới phân biệt được;
  * đơn vị nhãn là CẶP, khớp đúng với ba câu hỏi nghiên cứu, không phải điểm tuyệt đối của từng ảnh.

Ba phép so, mỗi phép trả lời một câu (xem `scripts/run_arms.py`):
    M–B  phản hồi (feedback) có ích không          M–T  NHÌN ẢNH có ích không
    M–S  PHÂN VAI ba agent có ích không

Ảnh thật lấy bằng `ctig.evaluation.ref_split(...)[1]` — tập CẤT RIÊNG, rời hẳn (theo băm nội dung) bộ
`selected/` mà IP-Adapter đã nhìn lúc sinh. Vì sao phải là tập cất riêng: nếu cho người gán xem đúng những
ảnh đã nuôi bộ sinh thì nhánh nào bắt chước ảnh đó sát nhất sẽ thắng, và ta đo lại chính đầu vào của mình.

Đầu ra JSON (mảng, MỖI BẢN GHI MỘT DÒNG):
    {pair_id, prompt_id, rep, left_arm, right_arm, choice, chosen_arm, is_repeat, annotator, ms}
`choice` ∈ {"left","right","tie"}; `chosen_arm` đã quy đổi sẵn về tên nhánh ("M"/"B"/...; null khi hoà) để
script phân tích không phải tự suy từ trái/phải — chỗ đó rất dễ suy ngược dấu và hỏng cả kết luận.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.bundle import thumb_b64  # noqa: E402
from ctig.evaluation import ref_split  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

#: Ảnh sinh nhúng ở 640 px, ảnh thật 320 px. Ảnh thật chỉ là mốc neo nên không cần to; ảnh sinh phải đủ lớn
#: để thấy hoa văn và đường may — nhưng to hơn nữa thì file HTML vượt trăm MB và trình duyệt ì khi lật trang.
GEN_SIDE, REF_SIDE = 640, 320
GEN_Q, REF_Q = 82, 75

#: Tỉ lệ cặp được hiện LẦN THỨ HAI (trái/phải đảo ngược) để đo độ tự nhất quán của người gán. Không có số này
#: thì không biết nhãn có phải tiếng ồn thuần tuý hay không, và không có cơ sở nào để tin vào hiệu số M–B.
REPEAT_FRAC = 0.10
MIN_REFS = 2          # dưới 2 ảnh thật thì mốc neo quá yếu, thà bỏ prompt còn hơn thu nhãn rác


# ----------------------------------------------------------------------------- gom dữ liệu

def load_arms(runs: list[str], ids: set[str] | None) -> list[dict]:
    """Mọi `arms.json` trong các run. Một bản ghi = một (prompt, lần lặp) với đủ bốn nhánh cùng seed."""
    out, da_co = [], {}
    for run in runs:
        for f in sorted(glob.glob(f"{run}/*/arms.json")):
            try:
                d = json.loads(Path(f).read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  bỏ {f}: đọc không được ({type(exc).__name__})")
                continue
            if ids and d.get("prompt_id") not in ids:
                continue
            # Hai run chạy lại cùng (prompt, lần lặp) sẽ cho hai bản ghi trùng khoá. Phải chặn tại đây: pair_id
            # và khoá localStorage đều dựng từ khoá này, trùng thì hai cặp khác ảnh lại chung một ô trả lời, và
            # `scripts/analyze_arms.py` sẽ tưởng bản sau là cặp lặp của bản trước.
            k = (d.get("prompt_id"), int(d.get("rep", 0)))
            if k in da_co:
                print(f"  bỏ {f}: trùng (prompt, rep) với {da_co[k]} — dùng bản gặp trước")
                continue
            da_co[k] = f
            d["_src"] = f
            out.append(d)
    return out


def prompt_vi_map() -> dict[str, str]:
    """prompt_id -> câu tiếng Việt. Chỉ câu tiếng Việt được hiện: người gán là người Việt, và câu tiếng Anh
    đã nở ra theo Culture-TRIP thì lộ luôn nhánh nào được sửa prompt."""
    m: dict[str, str] = {}
    for f in ("data/prompts_simple.json", "data/prompts_complex.json"):
        p = REPO / f
        if p.exists():
            for rec in json.loads(p.read_text(encoding="utf-8")):
                if rec.get("id"):
                    m[rec["id"]] = rec.get("text_vi", "")
    return m


def resolve_ref_dirs(arg_ref_dir: str | None, arg_config: str | None, runs: list[str]) -> str:
    """Thư mục ảnh thật, theo thứ tự ưu tiên: --ref-dir > --config > config.json ghi lại trong run.

    Vì sao phải dò cả trong run: `arms.json` chỉ ghi đường dẫn ảnh sinh, không ghi kho ảnh thật; còn
    `configs/vast_arms.yaml` trỏ vào `/workspace/...` của máy thuê nên trên máy ở nhà là sai đường dẫn.
    """
    if arg_ref_dir:
        return arg_ref_dir
    if arg_config:
        from ctig.config import Config  # nạp lười: chỉ nhánh này mới cần pyyaml

        rd = Config.load(arg_config).retrieval.ref_dir
        if rd:
            return rd
    for run in runs:
        for f in [f"{run}/config.json"] + sorted(glob.glob(f"{run}/*/config.json")):
            try:
                rd = (json.loads(Path(f).read_text(encoding="utf-8")).get("retrieval") or {}).get("ref_dir")
            except Exception:  # noqa: BLE001
                continue
            if rd:
                print(f"  lấy ref_dir từ {f}")
                return rd
    raise SystemExit(
        "không biết ảnh thật nằm ở đâu. Cho một trong hai:\n"
        "  --ref-dir /duong/dan/reference_images_simple,/duong/dan/reference_images_complex\n"
        "  --config configs/vast_arms.yaml      (chỉ đúng khi chạy trên chính máy đã sinh ảnh)\n"
        "Mỗi thư mục phải có dạng <ref_dir>/candidates/<prompt_id>/*.jpg"
    )


# ----------------------------------------------------------------------------- dựng danh sách cặp

def _coin(seed: int, key: str) -> bool:
    """Tung đồng xu ỔN ĐỊNH theo nội dung khoá, không theo thứ tự gọi.

    Vì sao không dùng thẳng `random.Random(seed)` tuần tự: khi ta đổi --max-pairs, thêm một run, hay bỏ một
    prompt thiếu ảnh thật, thứ tự gọi đổi theo, và nhánh nào đứng bên nào sẽ đổi hết. Băm theo pair_id giữ
    cho mỗi cặp luôn có cùng một cách xếp trái/phải giữa các lần dựng trang — cần thế để hai người gán khác
    nhau (hoặc cùng người gán lần hai) vẫn nhìn thấy cùng một bố cục, và để truy lại được về sau.
    """
    return hashlib.sha1(f"{seed}|{key}".encode()).digest()[0] % 2 == 1


def build_items(recs: list[dict], pairs: list[tuple[str, str]], seed: int, max_pairs: int | None):
    """[(mục màn hình)] đã xáo thứ tự, đã cài cặp lặp. Mỗi mục biết nhánh nào nằm bên nào."""
    base, thieu = [], []
    for d in recs:
        pid, rep = d.get("prompt_id"), int(d.get("rep", 0))
        imgs = d.get("images") or {}
        for a, b in pairs:
            pa, pb = imgs.get(a), imgs.get(b)
            if not (pa and pb and os.path.exists(pa) and os.path.exists(pb)):
                thieu.append((pid, rep, f"{a}-{b}", "thiếu ảnh nhánh "
                              + ",".join(x for x, p in ((a, pa), (b, pb)) if not (p and os.path.exists(p)))))
                continue
            pair_id = f"{pid}_r{rep}_{a}-{b}"
            # nhánh đầu (thường là M) nằm bên trái hay bên phải do đồng xu băm quyết định -> người gán không
            # thể học mẹo "cứ chọn bên trái"; tên nhánh vẫn được ghi lại để chấm ngược.
            trai, phai = (a, b) if _coin(seed, pair_id) else (b, a)
            base.append({"pair_id": pair_id, "prompt_id": pid, "rep": rep, "kind": f"{a}-{b}",
                         "left_arm": trai, "right_arm": phai,
                         "left_path": imgs[trai], "right_path": imgs[phai], "is_repeat": False})

    rng = random.Random(seed)
    rng.shuffle(base)                      # trộn để các cặp cùng prompt không đứng liền nhau

    if max_pairs and len(base) > max_pairs:
        # Cắt theo VÒNG TRÒN qua ba loại phép so, không cắt thẳng đuôi: cắt thẳng dễ làm một phép so
        # (ví dụ M–S) còn quá ít cặp và mất luôn khả năng kết luận về nó.
        xo: dict[str, list[dict]] = {}
        for it in base:
            xo.setdefault(it["kind"], []).append(it)
        chon, keys = [], sorted(xo)
        while len(chon) < max_pairs and any(xo[k] for k in keys):
            for k in keys:
                if xo[k] and len(chon) < max_pairs:
                    chon.append(xo[k].pop(0))
        rng.shuffle(chon)
        base = chon

    seq = list(base)
    n_rep = max(1, round(REPEAT_FRAC * len(base))) if len(base) >= 10 else 0
    # Chỉ lặp lại những cặp nằm ở NỬA ĐẦU, và chèn bản lặp vào phần đuôi: hai lần xem phải cách nhau đủ xa
    # để người gán không nhớ mặt ảnh, nếu không thì con số "tự nhất quán" chỉ đo trí nhớ ngắn hạn.
    nua_dau = base[: max(1, int(len(base) * 0.55))]
    for it in rng.sample(nua_dau, min(n_rep, len(nua_dau))):
        dup = dict(it)
        dup["is_repeat"] = True
        dup["left_arm"], dup["right_arm"] = it["right_arm"], it["left_arm"]     # ĐẢO trái phải
        dup["left_path"], dup["right_path"] = it["right_path"], it["left_path"]
        # Khoảng cách tối thiểu tính từ VỊ TRÍ HIỆN TẠI của bản gốc: các lần chèn trước đã đẩy nó lùi xuống,
        # nên chỉ lấy "60% cuối danh sách" là chưa đủ — đã thấy trường hợp hai lần xem chỉ cách nhau 8 màn hình.
        cach = min(10, max(3, len(seq) // 4))
        lo = min(max(int(len(seq) * 0.6), seq.index(it) + cach), len(seq))
        seq.insert(rng.randrange(lo, len(seq) + 1), dup)
    return seq, thieu


# ----------------------------------------------------------------------------- trang HTML

CSS = """
:root{--fg:#1d1d1f;--mut:#6b6b70;--line:#dcdcd8;--ok:#168052;--tie:#8a8a8f;--bg:#faf9f7;--pick:#1f6feb}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin-top:6px}
.bar>i{display:block;height:100%;background:var(--ok);width:0;transition:width .2s}
main{max-width:1180px;margin:0 auto;padding:14px 16px 40px}
.refbox{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.refbox h2{margin:0 0 8px;font-size:13px;font-weight:600;color:var(--mut);letter-spacing:.02em;text-transform:uppercase}
.refs{display:flex;gap:10px;flex-wrap:wrap}
.refs img{height:190px;width:auto;max-width:100%;border:1px solid var(--line);border-radius:5px;display:block}
.prompt{margin:10px 0 0;font-size:16px}
.ask{margin:18px 0 10px;font-size:19px;font-weight:650;text-align:center}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:860px){.two{grid-template-columns:1fr}.refs img{height:140px}}
.card{background:#fff;border:2px solid var(--line);border-radius:8px;padding:8px;cursor:pointer}
.card.pick{border-color:var(--pick);box-shadow:0 0 0 3px rgba(31,111,235,.15)}
.card img{width:100%;display:block;border-radius:4px}
.tag{display:inline-block;font:13px ui-monospace,monospace;color:var(--mut);border:1px solid var(--line);
     border-radius:4px;padding:1px 7px;margin-bottom:6px}
.btns{display:flex;gap:10px;justify-content:center;margin-top:16px;flex-wrap:wrap}
button{font:inherit;padding:9px 16px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}
button:hover{border-color:#999}
button.sel[data-v=left],button.sel[data-v=right]{background:var(--pick);border-color:var(--pick);color:#fff}
button.sel[data-v=tie]{background:var(--tie);border-color:var(--tie);color:#fff}
.nav{display:flex;gap:10px;align-items:center;margin-top:22px;border-top:1px solid var(--line);padding-top:12px}
.key{font:12px ui-monospace,monospace;color:var(--mut);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
.muted{color:var(--mut);font-size:13px}
.done{color:var(--ok);font-weight:600}
"""

JS = """
const D = DATA, IMG = IMGS, KEY = "ctig_pairs_" + DATA_ID;
let i = 0, ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ans = {}; }
let shownAt = Date.now();

const save = () => { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} };
const el = (id) => document.getElementById(id);
const nDone = () => D.filter(it => ans[it.uid] && ans[it.uid].choice).length;

function render() {
  const it = D[i], a = ans[it.uid] || {};
  el("refs").innerHTML = it.refs.map(k => '<img src="' + IMG[k] + '" alt="">').join("");
  el("prompt").textContent = it.prompt_vi || "(thiếu câu prompt tiếng Việt)";
  el("imgL").src = IMG[it.li];
  el("imgR").src = IMG[it.ri];
  el("cardL").className = "card" + (a.choice === "left" ? " pick" : "");
  el("cardR").className = "card" + (a.choice === "right" ? " pick" : "");
  el("pos").textContent = (i + 1) + " / " + D.length;
  el("done").textContent = nDone() + " cặp đã chọn";
  el("bar").style.width = (100 * nDone() / D.length) + "%";
  document.querySelectorAll("#btns button").forEach(b =>
    b.classList.toggle("sel", a.choice === b.dataset.v));
  window.scrollTo({top: 0});
  shownAt = Date.now();
}

function pick(v) {
  const it = D[i];
  const a = ans[it.uid] || (ans[it.uid] = {});
  a.choice = v;
  a.ms = (a.ms || 0) + (Date.now() - shownAt);   // cộng dồn: quay lại sửa thì thời gian vẫn tính đủ
  save(); render();
  if (i < D.length - 1) setTimeout(() => { i++; render(); }, 200);
}

function go(d) { const j = i + d; if (j >= 0 && j < D.length) { i = j; render(); } }

document.addEventListener("keydown", e => {
  if (e.key === "ArrowRight") { go(1); return; }
  if (e.key === "ArrowLeft") { go(-1); return; }
  const m = {"1": "left", "2": "right", "3": "tie"}[e.key];
  if (m) { e.preventDefault(); pick(m); }
});

function exportJson() {
  const out = [];
  for (const it of D) {
    const a = ans[it.uid];
    if (!a || !a.choice) continue;
    out.push({pair_id: it.pair_id, prompt_id: it.prompt_id, rep: it.rep,
              left_arm: it.left_arm, right_arm: it.right_arm, choice: a.choice,
              chosen_arm: a.choice === "left" ? it.left_arm : (a.choice === "right" ? it.right_arm : null),
              is_repeat: it.is_repeat, annotator: ANNOT, ms: a.ms || null});
  }
  if (out.length < D.length &&
      !confirm("Mới chọn " + out.length + "/" + D.length + " cặp. Vẫn xuất file?")) return;
  const txt = "[\\n" + out.map(r => JSON.stringify(r)).join(",\\n") + "\\n]\\n";
  const u = URL.createObjectURL(new Blob([txt], {type: "application/json"}));
  const l = document.createElement("a");
  l.href = u; l.download = "pairs_" + ANNOT + ".json"; l.click(); URL.revokeObjectURL(u);
}

el("cardL").onclick = () => pick("left");
el("cardR").onclick = () => pick("right");
document.querySelectorAll("#btns button").forEach(b => { b.onclick = () => pick(b.dataset.v); });
render();
"""

PAGE_HEAD = (
    "<!doctype html><html lang='vi'><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>CTIG · so cặp ảnh</title><style>%s</style>"
    "<header><b>So cặp ảnh</b> <span class='muted'>phím <span class='key'>1</span> ảnh trái · "
    "<span class='key'>2</span> ảnh phải · <span class='key'>3</span> không phân biệt được · "
    "<span class='key'>&larr;</span><span class='key'>&rarr;</span> chuyển cặp</span>"
    "<div class='bar'><i id='bar'></i></div></header>"
)

PAGE_BODY = (
    "<main>"
    "<div class='refbox'><h2>Ảnh thật (tham chiếu)</h2><div class='refs' id='refs'></div>"
    "<p class='prompt' id='prompt'></p></div>"
    "<p class='ask'>Ảnh nào giống văn hoá Việt Nam hơn, so với ảnh thật ở trên?</p>"
    "<div class='two'>"
    "<div class='card' id='cardL'><span class='tag'>1</span><img id='imgL' alt=''></div>"
    "<div class='card' id='cardR'><span class='tag'>2</span><img id='imgR' alt=''></div>"
    "</div>"
    "<div class='btns' id='btns'>"
    "<button data-v='left'>[1] ảnh trái</button>"
    "<button data-v='right'>[2] ảnh phải</button>"
    "<button data-v='tie'>[3] không phân biệt được</button></div>"
    "<div class='nav'><button onclick='go(-1)'>&larr; trước</button>"
    "<button onclick='go(1)'>sau &rarr;</button>"
    "<span class='muted' id='pos'></span><span class='done' id='done'></span>"
    "<button onclick='exportJson()' style='margin-left:auto'>Xuất JSON</button></div>"
    "</main>"
)


def build(runs, out, pairs, annotator, seed, max_pairs, n_refs, ref_dirs, n_eval, ids, log=print) -> str:
    recs = load_arms(runs, ids)
    if not recs:
        raise SystemExit(
            "không thấy `arms.json` nào trong " + ", ".join(runs) + ".\n"
            "Trang này chỉ đọc được run của thiết kế BỐN NHÁNH (scripts/run_arms.py ghi <run>/<pid>_r<rep>/arms.json).\n"
            "Run của thiết kế cũ (multigen.json / loop_v2.json) không có đủ bốn nhánh cùng seed nên không so cặp được."
        )
    log(f"{len(recs)} bản ghi arms.json")

    seq, thieu = build_items(recs, pairs, seed, max_pairs)
    if not seq:
        raise SystemExit("không dựng được cặp nào: mọi bản ghi đều thiếu ảnh ở ít nhất một nhánh")

    vi = prompt_vi_map()
    imgs: list[str] = []            # kho base64 dùng chung
    idx: dict[str, int] = {}        # đường dẫn -> vị trí trong kho
    refs_cache: dict[str, list[int] | None] = {}

    def nhung(path: str, side: int, q: int) -> int | None:
        """Nhúng một lần, dùng lại nhiều nơi. Ảnh của nhánh M xuất hiện ở cả ba phép so, ảnh thật xuất hiện ở
        mọi cặp cùng prompt — nhúng lặp thì file phồng lên gấp ba mà không thêm thông tin gì."""
        if path in idx:
            return idx[path]
        b64 = thumb_b64(path, side, quality=q)
        if not b64:
            return None
        idx[path] = len(imgs)
        imgs.append(b64)
        return idx[path]

    def refs_of(pid: str) -> list[int] | None:
        """Ảnh thật CẤT RIÊNG của prompt, lấy N ảnh đầu. Cùng một prompt luôn hiện CÙNG bộ ảnh thật ở cả ba
        phép so — nếu mỗi cặp một bộ khác nhau thì chênh lệch giữa M–B và M–S có thể chỉ do mốc neo khác."""
        if pid in refs_cache:
            return refs_cache[pid]
        try:
            _, eval_refs = ref_split(ref_dirs, pid, max(n_eval, n_refs))
        except Exception as exc:  # noqa: BLE001
            log(f"  {pid}: đọc ảnh thật lỗi ({type(exc).__name__}: {exc})")
            eval_refs = []
        got = [k for k in (nhung(p, REF_SIDE, REF_Q) for p in eval_refs[:n_refs]) if k is not None]
        refs_cache[pid] = got if len(got) >= MIN_REFS else None
        return refs_cache[pid]

    data, bo = [], []
    for it in seq:
        rf = refs_of(it["prompt_id"])
        if rf is None:
            bo.append((it["pair_id"], f"dưới {MIN_REFS} ảnh thật cất riêng"))
            continue
        li, ri = nhung(it["left_path"], GEN_SIDE, GEN_Q), nhung(it["right_path"], GEN_SIDE, GEN_Q)
        if li is None or ri is None:
            bo.append((it["pair_id"], "không đọc được ảnh sinh"))
            continue
        # `uid` là khoá của câu trả lời trong localStorage: phải khác nhau giữa bản gốc và bản lặp, nếu không
        # hai lần xem sẽ ghi đè nhau và mất luôn số đo tự nhất quán.
        data.append({"uid": it["pair_id"] + ("#lap" if it["is_repeat"] else ""),
                     "pair_id": it["pair_id"], "prompt_id": it["prompt_id"], "rep": it["rep"],
                     "left_arm": it["left_arm"], "right_arm": it["right_arm"],
                     "li": li, "ri": ri, "refs": rf, "is_repeat": it["is_repeat"],
                     "prompt_vi": vi.get(it["prompt_id"], "")})
    if not data:
        raise SystemExit("không cặp nào đủ điều kiện (thiếu ảnh thật hoặc ảnh sinh) — kiểm lại --ref-dir")

    data_id = f"{annotator}_{seed}_{len(data)}_{'+'.join(a + b for a, b in pairs)}"
    page = (
        (PAGE_HEAD % CSS) + PAGE_BODY
        + "<script>const DATA=" + json.dumps(data, ensure_ascii=False)
        + ";const IMGS=" + json.dumps(imgs)
        + ";const DATA_ID=" + json.dumps(data_id)
        + ";const ANNOT=" + json.dumps(annotator) + ";" + JS + "</script></html>"
    )
    Path(out).write_text(page, encoding="utf-8")

    n_lap = sum(1 for d in data if d["is_repeat"])
    theo_loai: dict[str, int] = {}
    for d in data:
        if d["is_repeat"]:
            continue                      # đếm cặp GỐC theo từng phép so; cặp lặp đã tính riêng ở trên
        k = d["pair_id"].rsplit("_", 1)[-1]
        theo_loai[k] = theo_loai.get(k, 0) + 1
    log(f"{out} · {len(data)} màn hình ({len(data) - n_lap} cặp + {n_lap} cặp lặp) · "
        + " · ".join(f"{k} {v}" for k, v in sorted(theo_loai.items()))
        + f" · {len(imgs)} ảnh nhúng · {os.path.getsize(out) // 1024} KB")
    for pid, rep, kind, ly_do in thieu[:6]:
        log(f"   bỏ {pid}_r{rep} {kind}: {ly_do}")
    if len(thieu) > 6:
        log(f"   ... và {len(thieu) - 6} cặp nữa thiếu ảnh")
    for pair_id, ly_do in bo[:6]:
        log(f"   bỏ {pair_id}: {ly_do}")
    if len(bo) > 6:
        log(f"   ... và {len(bo) - 6} cặp nữa bị bỏ")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="dựng trang gán nhãn so cặp cho bốn nhánh B/T/S/M")
    ap.add_argument("runs", nargs="+", help="một hoặc nhiều thư mục run có <pid>_r<rep>/arms.json")
    ap.add_argument("-o", "--out", default="pairs.html")
    ap.add_argument("--pairs", default="M-B,M-T,M-S", help="các phép so, cách nhau bằng dấu phẩy")
    ap.add_argument("--annotator", default="a1")
    ap.add_argument("--seed", type=int, default=0, help="quyết định xáo trái/phải, xáo thứ tự, chọn cặp lặp")
    ap.add_argument("--max-pairs", type=int, default=None, help="trần số cặp GỐC (cặp lặp cộng thêm ngoài trần)")
    ap.add_argument("--refs", type=int, default=3, help="số ảnh thật hiện làm mốc neo (2–3)")
    ap.add_argument("--ref-dir", default=None, help="kho ảnh thật, nhiều thư mục cách nhau bằng dấu phẩy")
    ap.add_argument("--config", default=None, help="lấy retrieval.ref_dir từ file cấu hình này")
    ap.add_argument("--n-eval", type=int, default=5, help="số ảnh cất riêng mà ref_split tách ra mỗi prompt")
    ap.add_argument("--ids", default=None, help="giới hạn prompt, cách nhau bằng dấu phẩy")
    a = ap.parse_args(argv)

    pairs = []
    for tok in a.pairs.split(","):
        tok = tok.strip()
        if not tok:
            continue
        x, _, y = tok.partition("-")
        if not (x and y):
            raise SystemExit(f"--pairs sai định dạng ở '{tok}', phải là dạng M-B")
        pairs.append((x.strip(), y.strip()))
    if not pairs:
        raise SystemExit("--pairs rỗng")
    if a.refs < MIN_REFS:
        raise SystemExit(f"--refs phải >= {MIN_REFS}: dưới hai ảnh thật thì mốc neo quá yếu")

    build(a.runs, a.out, pairs, a.annotator, a.seed, a.max_pairs, a.refs,
          resolve_ref_dirs(a.ref_dir, a.config, a.runs), a.n_eval,
          {i.strip() for i in a.ids.split(",")} if a.ids else None)


if __name__ == "__main__":
    main()
