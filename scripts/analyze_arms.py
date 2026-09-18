"""Đọc kết quả thí nghiệm BỐN NHÁNH (B / T / S / M) và trả lời ba khẳng định của bài.

    python scripts/analyze_arms.py /workspace/runs/arms \
        --config configs/vast_arms.yaml \
        --labels data/labels/pairs_huy.json \
        -o results/arms_analysis.json

Bốn nhánh, mỗi nhánh MỘT ảnh cho cùng prompt và CÙNG SEED (xem `scripts/run_arms.py`):

    B  không agent                     ·  T  có contract nhưng KHÔNG nhìn ảnh
    S  một VLM tự viết mệnh đề sửa     ·  M  Observer -> Critic -> Refiner -> Critic duyệt

Ba khẳng định, mỗi khẳng định là một phép so:

    M > B   phản hồi có ích
    M > T   NHÌN ẢNH có ích   (không có T thì M>B có thể chỉ vì prompt dài thêm vài chữ)
    M > S   PHÂN VAI có ích   (không có S thì sơ đồ ba agent chỉ là trang trí)

Luật kết luận được cài cứng trong script, ở hàm `ket_luan_ba_khang_dinh`: nếu M chỉ thắng B mà KHÔNG thắng S
thì chỉ được viết "nhìn ảnh có ích", TUYỆT ĐỐI không được viết "phân vai có ích". Cài cứng vì đây đúng là chỗ
dễ tự lừa mình nhất khi ngồi nhìn bảng số lúc 2 giờ sáng.

Hai điều bắt buộc về thống kê, và vì sao:

  1. **Không dùng điểm của Mistral làm thước kết luận.** Vòng sửa của nhánh M tối ưu thẳng vào bộ chấm ấy,
     nên báo cáo chính điểm ấy là hệ thống tự chấm mình. Nếu tìm thấy điểm đó trong dữ liệu thì script chỉ in
     ra ở phần CHẨN ĐOÁN và ghi rõ là chẩn đoán.
  2. **Mọi khoảng tin cậy bootstrap lấy mẫu lại THEO CỤM PROMPT, không theo cặp.** Hai lần lặp (`rep`) của
     cùng một prompt dùng chung câu prompt, chung contract, chung ảnh tham chiếu — chúng không độc lập.
     Bootstrap theo cặp coi 2 lần lặp × 25 prompt là 50 quan sát độc lập và cho khoảng tin cậy hẹp hơn sự
     thật khoảng căn(2) lần. Cỡ mẫu thật là SỐ PROMPT, không phải số cặp; script luôn in số cụm prompt thật.

Bốn phần in ra:
  1. nhãn người (bỏ qua nếu không có tệp nhãn) — thước KẾT LUẬN
  2. thước tự động DINOv2-SIM và CLIP — thước bổ trợ, dùng lại `dino_embed` của `scripts/eval_independent.py`
  3. chẩn đoán vòng sửa — tỉ lệ no-op, lỗi Critic tìm được, ảnh M có trùng byte với B không
  4. bảng markdown gọn để dán vào bài
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARMS = ("B", "T", "S", "M")
MO_TA_NHANH = {
    "B": "không agent",
    "T": "contract, KHÔNG nhìn ảnh",
    "S": "một VLM tự viết",
    "M": "Observer→Critic→Refiner→duyệt",
}
#: Ba phép so làm nên bài báo, kèm điều mỗi phép so chứng minh được.
DOI_CHUNG = [("M", "B", "phản hồi có ích"),
             ("M", "T", "NHÌN ẢNH có ích"),
             ("M", "S", "PHÂN VAI có ích")]
#: Dưới ngần này cụm prompt thì mọi khoảng tin cậy chỉ để tham khảo, không đủ để kết luận.
IT_CUM = 10
#: Người gán chọn lại giống lần đầu dưới ngưỡng này thì chính nhãn người cũng không tin được.
NGUONG_TU_NHAT_QUAN = 0.7


# =====================================================================  công cụ thống kê
def phan_vi(xs: list[float], q: float) -> float:
    """Phân vị nội suy tuyến tính. Tự viết để script chạy được bằng stdlib, không cần numpy."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def boot_theo_cum(cum: dict[str, list[float]], n_boot: int = 5000, seed: int = 0,
                  alpha: float = 0.10) -> tuple[float, float, float, int]:
    """Bootstrap trung bình, LẤY MẪU LẠI THEO CỤM PROMPT. Trả (trung bình, cận dưới, cận trên, số cụm).

    `cum` là {prompt_id: [giá trị của từng quan sát thuộc prompt đó]}. Mỗi vòng bootstrap rút lại ĐỦ SỐ CỤM
    prompt có hoàn lại rồi gộp toàn bộ quan sát của các cụm rút được; đơn vị lấy mẫu là prompt chứ không phải
    quan sát. Đây là điểm khác biệt duy nhất nhưng quyết định: rút theo quan sát sẽ giả vờ rằng hai lần lặp
    của cùng một prompt là hai bằng chứng độc lập, trong khi chúng dùng chung câu prompt và chung contract.
    """
    keys = [k for k, v in cum.items() if v]
    if not keys:
        return float("nan"), float("nan"), float("nan"), 0
    tat_ca = [v for k in keys for v in cum[k]]
    tb = statistics.mean(tat_ca)
    rng = random.Random(seed)
    mau = []
    for _ in range(n_boot):
        pool: list[float] = []
        for _ in range(len(keys)):
            pool += cum[keys[rng.randrange(len(keys))]]
        if pool:
            mau.append(sum(pool) / len(pool))
    return tb, phan_vi(mau, alpha / 2), phan_vi(mau, 1 - alpha / 2), len(keys)


def fmt_ktc(tb: float, lo: float, hi: float) -> str:
    if any(math.isnan(x) for x in (tb, lo, hi)):
        return "không đủ dữ liệu"
    return f"{tb:.3f} [KTC 90%: {lo:.3f} – {hi:.3f}]"


# =====================================================================  đọc arms.json
def doc_arms(run_dirs: list[str], log=print) -> list[dict]:
    """Gom mọi `<run>/<prompt_id>_r<rep>/arms.json`. Khoá duy nhất là (prompt_id, rep).

    Báo lỗi RÕ RÀNG khi không tìm thấy gì: thư mục `runs_backup/` thuộc thiết kế CŨ (ba nhánh A/B/C, tệp
    `loop_v2.json` / `bestofn.json`) nên chắc chắn không có `arms.json`, và người chạy cần biết ngay đó là
    lý do chứ không phải script hỏng.
    """
    recs: dict[tuple[str, int], dict] = {}
    for rd in run_dirs:
        p = Path(rd)
        if not p.is_dir():
            log(f"  CẢNH BÁO: {rd} không phải thư mục -> bỏ")
            continue
        got = sorted(p.glob("*/arms.json"))
        log(f"  {rd}: {len(got)} tệp arms.json")
        for f in got:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                log(f"  CẢNH BÁO: đọc hỏng {f} ({type(exc).__name__}) -> bỏ")
                continue
            pid, rep = str(d.get("prompt_id") or f.parent.name), int(d.get("rep") or 0)
            d["_file"], d["prompt_id"], d["rep"] = str(f), pid, rep
            if (pid, rep) in recs:
                log(f"  CẢNH BÁO: {pid} rep {rep} có ở nhiều run, lấy bản sau: {f}")
            recs[(pid, rep)] = d
    if not recs:
        raise SystemExit(
            "KHÔNG tìm thấy tệp arms.json nào trong: " + ", ".join(run_dirs)
            + "\nScript này chỉ đọc được thiết kế BỐN NHÁNH do scripts/run_arms.py sinh ra, tức là"
              "\n    <run>/<prompt_id>_r<rep>/arms.json"
              "\nCác run cũ (runs_backup/pilotA3, pilotB3, pilotC, bestof4 ...) thuộc thiết kế BA NHÁNH A/B/C"
              "\nvới loop_v2.json / bestofn.json và KHÔNG dùng được ở đây — hãy chạy lại scripts/run_arms.py,"
              "\nhoặc dùng scripts/eval_independent.py cho các run cũ.")
    return [recs[k] for k in sorted(recs)]


def md5(path: str | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# =====================================================================  PHẦN 1 — nhãn người
#: `choice` do người gán bấm. Chấp nhận nhiều cách viết vì tệp nhãn do script khác sinh ra và có thể đổi.
TRAI = {"left", "l", "a", "trai", "ben_trai", "1"}
PHAI = {"right", "r", "b", "phai", "ben_phai", "2"}
HOA = {"tie", "hoa", "bang", "ngang", "draw", "equal", "same", "0"}
BO = {"skip", "bo_qua", "khong_chac", "khong_thay", "unsure", "", "none"}


def ben_thang(row: dict) -> str | None:
    """Trả tên nhánh thắng, "HOA", hoặc None nếu dòng nhãn phải bỏ (không chắc / không hiểu được).

    Ưu tiên `chosen_arm` vì nó đã khử được chuyện trái/phải bị trộn ngẫu nhiên lúc trình bày; chỉ khi thiếu
    mới suy từ `choice`. Không đoán bừa: giá trị lạ thì bỏ dòng và đếm riêng, chứ không gán đại cho một bên.
    """
    l, r = str(row.get("left_arm") or ""), str(row.get("right_arm") or "")
    ca = row.get("chosen_arm")
    if ca is not None:
        ca = str(ca).strip()
        if ca in (l, r):
            return ca
        if ca.lower() in HOA:
            return "HOA"
        if ca.lower() in BO:
            return None
    ch = str(row.get("choice") or "").strip().lower()
    if ch in TRAI:
        return l
    if ch in PHAI:
        return r
    if ch in HOA:
        return "HOA"
    return None


def khoa_cap(row: dict) -> tuple:
    """Khoá nhận dạng MỘT cặp ảnh đã hỏi, bất kể lần này nó được bày trái hay phải."""
    l, r = str(row.get("left_arm") or ""), str(row.get("right_arm") or "")
    return (str(row.get("annotator") or "?"), str(row.get("prompt_id") or "?"),
            int(row.get("rep") or 0), tuple(sorted((l, r))))


def tu_nhat_quan(rows: list[dict], log=print) -> dict:
    """Người gán chọn lại giống lần đầu bao nhiêu phần trăm, tính từ các cặp `is_repeat`.

    Vì sao phải có: đây là TRẦN của mọi con số khác trong phần 1. Nếu chính người gán chỉ nhất quán với mình
    60% thì không phép thử nào phân biệt nổi hai nhánh chênh nhau vài phần trăm, và biên thắng quan sát được
    phần lớn là nhiễu của người chứ không phải hiệu ứng của hệ thống.
    """
    goc: dict[tuple, str] = {}
    for r in rows:
        if not r.get("is_repeat"):
            kq = ben_thang(r)
            if kq is not None:
                goc.setdefault(khoa_cap(r), kq)
    khop, tong, mo_coi = 0, 0, 0
    for r in rows:
        if not r.get("is_repeat"):
            continue
        kq = ben_thang(r)
        if kq is None:
            continue
        g = goc.get(khoa_cap(r))
        if g is None:
            mo_coi += 1      # cặp lặp mà không tìm được lần gán đầu -> không tính, không đoán
            continue
        tong += 1
        khop += int(g == kq)
    return {"n": tong, "khop": khop, "mo_coi": mo_coi,
            "ti_le": (khop / tong) if tong else None}


def phan_1_nhan_nguoi(labels: list[dict], n_boot: int, seed: int, log=print) -> dict:
    log("\n" + "=" * 78)
    log("PHẦN 1 — NHÃN NGƯỜI (thước KẾT LUẬN)")
    log("=" * 78)

    ket: dict = {"n_dong": len(labels), "doi_chung": {}}
    bo_dong = sum(1 for r in labels if ben_thang(r) is None)
    if bo_dong:
        log(f"  bỏ {bo_dong}/{len(labels)} dòng vì nhãn không rõ (không chắc / không thấy / giá trị lạ)")

    # Các cặp `is_repeat` chỉ dùng để đo tự nhất quán, KHÔNG tính vào tỉ lệ thắng: tính cả hai lần cho
    # cùng một cặp ảnh là đếm một bằng chứng hai lần và làm khoảng tin cậy hẹp đi một cách giả tạo.
    chinh = [r for r in labels if not r.get("is_repeat")]
    log(f"  {len(chinh)} dòng chính + {len(labels) - len(chinh)} dòng lặp (chỉ dùng đo tự nhất quán)")

    for a, b, y_nghia in DOI_CHUNG:
        cum: dict[str, list[float]] = defaultdict(list)   # prompt_id -> [1.0 nếu M thắng, 0.0 nếu thua]
        thang = thua = hoa = 0
        for r in chinh:
            arms = {str(r.get("left_arm") or ""), str(r.get("right_arm") or "")}
            if arms != {a, b}:
                continue
            kq = ben_thang(r)
            if kq is None:
                continue
            pid = str(r.get("prompt_id") or "?")
            if kq == "HOA":
                hoa += 1
                continue                       # loại hoà khỏi tỉ lệ thắng, đúng như đã nêu trong bài
            if kq == a:
                thang += 1
                cum[pid].append(1.0)
            else:
                thua += 1
                cum[pid].append(0.0)

        tb, lo, hi, n_cum = boot_theo_cum(cum, n_boot, seed)
        n_qd = thang + thua
        vuot = (not math.isnan(lo)) and lo > 0.5
        d = {"thang": thang, "thua": thua, "hoa": hoa, "n_quyet_dinh": n_qd,
             "n_cum_prompt": n_cum, "ti_le_thang": tb if n_qd else None,
             "ktc90": [lo, hi] if n_qd else None, "can_duoi_vuot_0_5": bool(vuot), "y_nghia": y_nghia}
        ket["doi_chung"][f"{a}_{b}"] = d

        log(f"\n  {a} vs {b}  ({y_nghia})")
        if not n_qd:
            log("    KHÔNG có cặp nào được gán -> không kết luận được gì về khẳng định này")
            continue
        log(f"    {a} thắng {thang} · thua {thua} · hoà {hoa}")
        log(f"    tỉ lệ thắng (bỏ hoà, n = {n_qd} cặp trên {n_cum} cụm prompt): {fmt_ktc(tb, lo, hi)}")
        log(f"    -> cận dưới {'VƯỢT' if vuot else 'KHÔNG vượt'} 0,5"
            f"  ==>  {'có bằng chứng: ' + y_nghia if vuot else 'CHƯA chứng minh được: ' + y_nghia}")
        if n_cum < IT_CUM:
            log(f"    CẢNH BÁO CỠ MẪU: chỉ {n_cum} cụm prompt thật sự. Khoảng tin cậy bootstrap theo cụm với"
                f" dưới {IT_CUM} cụm rất rộng và không ổn định — đọc như tham khảo, đừng kết luận.")

    tq = tu_nhat_quan(labels, log)
    ket["tu_nhat_quan"] = tq
    log("\n  Độ tự nhất quán của người gán (từ các cặp is_repeat):")
    if not tq["n"]:
        log("    KHÔNG có cặp lặp nào ghép được -> không đo được. Nên thêm ~10% cặp lặp vào phiên gán sau,"
            "\n    vì không có nó thì không biết biên thắng ở trên là hiệu ứng thật hay nhiễu của người gán.")
    else:
        log(f"    {tq['khop']}/{tq['n']} = {tq['ti_le']:.3f}"
            + (f"  ({tq['mo_coi']} cặp lặp không tìm được lần gán đầu, đã bỏ)" if tq["mo_coi"] else ""))
        if tq["ti_le"] < NGUONG_TU_NHAT_QUAN:
            log(f"    CẢNH BÁO: dưới {NGUONG_TU_NHAT_QUAN:.1f}. Chính người gán còn không lặp lại được phán"
                "\n    đoán của mình, nên mọi tỉ lệ thắng ở trên đều bị nhiễu của người nén về phía 0,5."
                "\n    Phải sửa hướng dẫn gán hoặc làm rõ tiêu chí trước khi báo cáo con số nào.")

    ket["ket_luan"] = ket_luan_ba_khang_dinh(ket["doi_chung"], log)
    return ket


def ket_luan_ba_khang_dinh(dc: dict, log=print) -> dict:
    """Luật đọc bảng, cài cứng để không tự nới tay lúc viết bài.

    Điểm quan trọng nhất: M thắng B mà KHÔNG thắng S thì chỉ chứng minh được "có phản hồi thì tốt hơn không
    có", chứ hoàn toàn chưa chứng minh được việc TÁCH VAI ba agent đóng góp gì. Lúc đó sơ đồ ba agent trong
    bài vẫn chỉ là trang trí, và phải viết đúng như vậy.
    """
    def ok(k):
        return bool(dc.get(k, {}).get("can_duoi_vuot_0_5"))

    mb, mt, ms = ok("M_B"), ok("M_T"), ok("M_S")
    cau = []
    if mb:
        cau.append("Có bằng chứng phản hồi có ích (M > B).")
    else:
        cau.append("CHƯA có bằng chứng phản hồi có ích (M không thắng chắc B).")
    cau.append("Có bằng chứng nhìn ảnh có ích (M > T)." if mt
               else "CHƯA có bằng chứng nhìn ảnh có ích (M không thắng chắc T) — biên thắng so với B có thể"
                    " chỉ do prompt được nối thêm chữ.")
    if ms:
        cau.append("Có bằng chứng phân vai có ích (M > S).")
    else:
        cau.append("CHƯA có bằng chứng phân vai có ích (M không thắng chắc S). KHÔNG được viết rằng sơ đồ ba"
                   " agent đóng góp; ở mức bằng chứng này nó vẫn là trang trí.")
    if mb and not ms:
        cau.append("Cụ thể: M thắng B nhưng không thắng S, nên phần đóng góp chỉ được phát biểu tới mức"
                   " 'có phản hồi thị giác thì tốt hơn', KHÔNG được phát biểu 'phân vai thì tốt hơn'.")
    log("\n  KẾT LUẬN ĐƯỢC PHÉP VIẾT:")
    for c in cau:
        log("    · " + c)
    return {"M>B": mb, "M>T": mt, "M>S": ms, "cau": cau}


# =====================================================================  PHẦN 2 — thước tự động
def phan_2_thuoc_tu_dong(recs: list[dict], cfg, n_eval: int, top: int, n_boot: int, seed: int,
                         log=print) -> dict:
    """DINOv2-SIM với ảnh thật CẤT RIÊNG, và CLIP với câu prompt GỐC. So từng cặp trên cùng (prompt, rep).

    Dùng lại nguyên `dino_embed` và `ref_split` của `scripts/eval_independent.py` thay vì viết lại: hai thước
    phải giống hệt nhau về mọi chi tiết (model DINOv2 nào, tách ảnh thật theo băm nội dung ra sao, lấy trung
    bình top-k nào) thì bảng của hai script mới so được với nhau. Viết lại là cách chắc chắn nhất để hai con
    số lệch nhau mà không ai biết vì sao.
    """
    log("\n" + "=" * 78)
    log("PHẦN 2 — THƯỚC TỰ ĐỘNG (bổ trợ, không thay được nhãn người)")
    log("=" * 78)

    from ctig.evaluation import ref_split                      # noqa: PLC0415
    from ctig.pipeline import load_prompts                     # noqa: PLC0415
    from ctig.stages.perception import CLIPProbe               # noqa: PLC0415

    from scripts.eval_independent import dino_embed            # noqa: PLC0415

    prompts = {p.id: p for p in load_prompts(cfg.prompts_path)}
    dev = cfg.multigen.device
    try:
        clip = CLIPProbe(cfg.perception.clip_model, cfg.perception.device)
    except Exception as exc:  # noqa: BLE001
        log(f"  CẢNH BÁO: không nạp được CLIP ({type(exc).__name__}) -> bỏ cột CLIP")
        clip = None

    # ref_split đọc đĩa và băm nội dung từng ảnh; hai lần lặp của cùng prompt cho cùng tập nên nhớ lại.
    cache_ref: dict[str, list[str]] = {}
    rows, bo_qua = [], []
    for rec in recs:
        pid, rep = rec["prompt_id"], rec["rep"]
        imgs = {k: v for k, v in (rec.get("images") or {}).items() if v}
        thieu = [k for k in ARMS if not imgs.get(k) or not Path(imgs[k]).exists()]
        if thieu:
            bo_qua.append((pid, rep, "thiếu ảnh nhánh " + ",".join(thieu)))
            continue
        if pid not in cache_ref:
            cache_ref[pid] = ref_split(cfg.retrieval.ref_dir, pid, n_eval)[1]
        eval_refs = cache_ref[pid]
        if len(eval_refs) < 2:
            bo_qua.append((pid, rep, f"chỉ {len(eval_refs)} ảnh thật cất riêng"))
            continue
        try:
            ref_vec = dino_embed(eval_refs, dev)
            gen_vec = dino_embed([imgs[k] for k in ARMS], dev)
        except Exception as exc:  # noqa: BLE001
            bo_qua.append((pid, rep, f"DINOv2 lỗi {type(exc).__name__}"))
            continue
        r = {"prompt_id": pid, "rep": rep, "n_eval_refs": len(eval_refs)}
        for i, k in enumerate(ARMS):
            sims = sorted((gen_vec[i] @ ref_vec.T).tolist(), reverse=True)
            r[f"sim_{k}"] = round(sum(sims[:top]) / min(top, len(sims)), 4)
            if clip is not None and pid in prompts:
                try:
                    # CÂU PROMPT GỐC (text_en), không phải câu Culture-TRIP đã nở ra: ý định người dùng nằm ở
                    # câu gốc, chấm nhánh B bằng chính câu nó được nở ra là thiên vị nó.
                    r[f"clip_{k}"] = round(float(clip.similarity(imgs[k], [prompts[pid].text_en])[0]), 4)
                except Exception as exc:  # noqa: BLE001
                    log(f"    CLIP lỗi {type(exc).__name__} ở {pid} r{rep} nhánh {k}")
        rows.append(r)
        log(f"  {pid} r{rep}  " + "  ".join(f"{k} {r[f'sim_{k}']:.3f}" for k in ARMS))

    if not rows:
        log("  KHÔNG chấm được (prompt, rep) nào.")
        for pid, rep, ly_do in bo_qua[:10]:
            log(f"    bỏ {pid} r{rep}: {ly_do}")
        return {"rows": [], "bo_qua": bo_qua}

    log(f"\n  {len(rows)} (prompt, rep) chấm được · {len(bo_qua)} bỏ qua"
        f" · {len({r['prompt_id'] for r in rows})} cụm prompt")
    for pid, rep, ly_do in bo_qua[:8]:
        log(f"    bỏ {pid} r{rep}: {ly_do}")

    tong_ket = {}
    log(f"\n  {'nhánh':6s} {'SIM ảnh thật':>18s} {'CLIP prompt gốc':>17s}")
    for k in ARMS:
        s = [r[f"sim_{k}"] for r in rows]
        c = [r[f"clip_{k}"] for r in rows if f"clip_{k}" in r]
        tong_ket[k] = {"sim": statistics.mean(s), "sim_sd": statistics.pstdev(s),
                       "clip": statistics.mean(c) if c else None, "n_clip": len(c)}
        log(f"  {k:6s} {tong_ket[k]['sim']:10.4f} ±{tong_ket[k]['sim_sd']:.3f} "
            + (f"{tong_ket[k]['clip']:17.4f}" if c else f"{'-':>17s}"))

    # So TỪNG CẶP trên cùng (prompt, rep): prompt khó dễ chênh nhau nhiều hơn hiệu ứng cần đo rất nhiều,
    # nên so trung bình toàn cục sẽ chìm nghỉm trong phương sai giữa các prompt.
    cap = {}
    for ten, thuoc in (("SIM", "sim"), ("CLIP", "clip")):
        co = [r for r in rows if f"{thuoc}_M" in r]
        if not co:
            continue
        log(f"\n  So từng cặp trên cùng (prompt, rep) — {ten}, n = {len(co)}:")
        for a, b, y_nghia in DOI_CHUNG:
            cum: dict[str, list[float]] = defaultdict(list)
            for r in co:
                cum[r["prompt_id"]].append(r[f"{thuoc}_{a}"] - r[f"{thuoc}_{b}"])
            d = [v for vs in cum.values() for v in vs]
            tb, lo, hi, n_cum = boot_theo_cum(cum, n_boot, seed)
            thang = sum(1 for v in d if v > 0)
            bang = sum(1 for v in d if v == 0)
            sd = statistics.pstdev(d) or 1e-9
            cap[f"{thuoc}_{a}_{b}"] = {"hieu_tb": tb, "ktc90": [lo, hi], "n": len(d), "n_cum_prompt": n_cum,
                                       "thang": thang, "bang_dung_0": bang, "d_cohen": tb / sd}
            log(f"    {a} − {b} ({y_nghia}): {fmt_ktc(tb, lo, hi)}"
                f" · {a} hơn ở {thang}/{len(d)} cặp"
                + (f" ({bang} cặp bằng ĐÚNG 0 — ảnh trùng nhau)" if bang else "")
                + f" · d = {tb / sd:+.2f}")
        if len({r["prompt_id"] for r in co}) < IT_CUM:
            log(f"    CẢNH BÁO CỠ MẪU: chỉ {len({r['prompt_id'] for r in co})} cụm prompt.")

    log("\n  LƯU Ý: SIM và CLIP đều là thước BỔ TRỢ. Chưa có bằng chứng SIM bám theo phán đoán của người về"
        "\n  tính đúng văn hoá, nên chúng không thay được phần 1 — phải ghi vào phần Hạn chế của bài.")
    return {"rows": rows, "bo_qua": bo_qua, "tong_ket": tong_ket, "tung_cap": cap}


# =====================================================================  PHẦN 3 — chẩn đoán vòng sửa
def tim_diem_mistral(rec: dict) -> dict:
    """Quét xem trong arms.json có điểm của bộ chấm Mistral lọt vào không.

    Có thì CHỈ in ở phần chẩn đoán. Vòng sửa của nhánh M tối ưu thẳng vào bộ chấm ấy (Critic và bước duyệt
    đều là cùng một MLLM), nên dùng nó làm thước kết luận là hệ thống tự chấm mình — đúng cái lỗi mà
    `eval_independent.py` được viết ra để tránh.
    """
    ra = {}
    for k in ("eval", "scores", "score", "overall", "judge", "mistral"):
        if k in rec:
            ra[k] = rec[k]
    for arm, tr in (rec.get("traces") or {}).items():
        if isinstance(tr, dict) and any(x in tr for x in ("overall", "score", "eval")):
            ra[f"traces.{arm}"] = {x: tr[x] for x in ("overall", "score", "eval") if x in tr}
    return ra


def phan_3_chan_doan(recs: list[dict], log=print) -> dict:
    log("\n" + "=" * 78)
    log("PHẦN 3 — CHẨN ĐOÁN VÒNG SỬA")
    log("=" * 78)
    n = len(recs)

    # --- no-op: con số quan trọng nhất của cả phần này
    noop = [r for r in recs if r.get("noop_M")]
    ti_le_noop = len(noop) / n
    log(f"\n  no-op của M: {len(noop)}/{n} = {ti_le_noop:.1%}")
    ly_do = Counter(str(r.get("ly_do_noop_M") or "(không ghi)") for r in noop)
    for ld, c in ly_do.most_common():
        log(f"    {c:3d}  {ld}")
    if ti_le_noop >= 0.5:
        log(f"\n    BÁO ĐỘNG: M no-op ở {ti_le_noop:.0%} số (prompt, rep). No-op nghĩa là M trả về ĐÚNG prompt"
            "\n    gốc với ĐÚNG seed, tức ảnh M trùng hệt ảnh B. Khi phần lớn prompt như vậy thì M ≈ B theo"
            "\n    nghĩa đen, mọi so sánh M vs B mất hết ý nghĩa (phần lớn cặp là hoà cưỡng bức), và biên"
            "\n    thắng nếu có chỉ đến từ thiểu số prompt M thật sự can thiệp. PHẢI báo con số này trong bài"
            "\n    và nên báo cáo thêm tỉ lệ thắng TÍNH RIÊNG trên tập prompt M có can thiệp.")
    elif ti_le_noop >= 0.25:
        log(f"\n    LƯU Ý: {ti_le_noop:.0%} no-op là đáng kể; nhớ báo cáo kèm mọi tỉ lệ thắng.")

    # --- Critic tìm được bao nhiêu lỗi, và viện dẫn contract_id nào
    so_loi, cids, khong_chay = [], Counter(), 0
    for r in recs:
        m2 = ((r.get("traces") or {}).get("M") or {}).get("m2")
        if not isinstance(m2, dict):
            khong_chay += 1          # Observer hỏng nên Critic chưa bao giờ chạy -> không tính vào trung bình
            continue
        v = m2.get("violations") or []
        so_loi.append(len(v))
        for x in v:
            cid = str((x or {}).get("contract_id") or "").strip()
            if cid:
                cids[cid] += 1
    log(f"\n  Critic tìm được trung bình {statistics.mean(so_loi) if so_loi else float('nan'):.2f} lỗi/ảnh"
        f"  (trên {len(so_loi)} lượt Critic thật sự chạy"
        + (f", {khong_chay} lượt Critic không chạy vì Observer hỏng)" if khong_chay else ")"))
    if cids:
        log("  contract_id bị viện dẫn nhiều nhất:")
        for cid, c in cids.most_common(8):
            log(f"    {c:3d}  {cid}")
        top_cid, top_c = cids.most_common(1)[0]
        if top_c / max(sum(cids.values()), 1) > 0.5:
            log(f"    LƯU Ý: một mình '{top_cid}' chiếm {top_c / sum(cids.values()):.0%} số lần viện dẫn —"
                "\n    nhiều khả năng Critic đang bám một mục dễ thấy của contract chứ không soi thật.")
    else:
        log("  KHÔNG có contract_id nào được viện dẫn — Critic chưa bao giờ báo được lỗi hợp lệ.")

    # --- ảnh M có trùng BYTE với ảnh B không: phép kiểm tính nhất quán rẻ nhất của cả pipeline
    trung, khac, thieu = 0, 0, 0
    sai_noop, sai_khong_noop = [], []
    for r in recs:
        im = r.get("images") or {}
        hm, hb = md5(im.get("M")), md5(im.get("B"))
        if hm is None or hb is None:
            thieu += 1
            continue
        giong = hm == hb
        trung += int(giong)
        khac += int(not giong)
        if r.get("noop_M") and not giong:
            sai_noop.append(f"{r['prompt_id']} r{r['rep']}")
        if (not r.get("noop_M")) and giong:
            sai_khong_noop.append(f"{r['prompt_id']} r{r['rep']}")
    n_so = trung + khac
    log(f"\n  Ảnh M trùng BYTE với ảnh B (md5): {trung}/{n_so}"
        + (f" = {trung / n_so:.1%}" if n_so else "")
        + (f"  ({thieu} cặp thiếu tệp ảnh, không so được)" if thieu else ""))
    log(f"    (đối chiếu: {len(noop)} lượt ghi no-op — no-op thì BẮT BUỘC trùng 100%, vì cùng prompt cùng seed)")
    if sai_noop:
        log(f"    LỖI: {len(sai_noop)} lượt ghi no-op mà ảnh M KHÁC ảnh B: {', '.join(sai_noop[:6])}"
            "\n    Đây là bug thật, không phải nhiễu: cùng prompt + cùng seed phải cho ảnh giống từng byte."
            "\n    Nghi can theo thứ tự: negative_terms vẫn bị nối vào dù no-op; seed hoặc scheduler không"
            "\n    được đặt lại giữa hai lần sinh; hoặc phép sinh không tất định (attention slicing, xformers).")
    if sai_khong_noop:
        log(f"    LỖI: {len(sai_khong_noop)} lượt KHÔNG no-op mà ảnh M vẫn trùng hệt B:"
            f" {', '.join(sai_khong_noop[:6])}"
            "\n    Nghĩa là mệnh đề sửa đã được duyệt nhưng KHÔNG hề tới được bộ sinh — kiểm lại đường đi của"
            "\n    `repair_clause` trong run_arms.py trước khi tin bất kỳ con số nào của nhánh M.")
    if n_so and not sai_noop and not sai_khong_noop:
        log("    khớp hoàn toàn với cờ no-op — đường đi của mệnh đề sửa nhất quán.")

    # --- độ dài mệnh đề sửa
    log("\n  Độ dài mệnh đề sửa (số từ, chỉ tính lượt CÓ mệnh đề):")
    do_dai = {}
    for arm in ("T", "S", "M"):
        ws = [len(str(((r.get("proposals") or {}).get(arm) or {}).get("clause") or "").split())
              for r in recs]
        co = [w for w in ws if w > 0]
        do_dai[arm] = {"tb": statistics.mean(co) if co else None, "n_co": len(co), "n": len(ws),
                       "max": max(co) if co else 0}
        log(f"    {arm}: " + (f"{do_dai[arm]['tb']:.1f} từ · dài nhất {max(co)} từ" if co
                              else "không có mệnh đề nào")
            + f" · có mệnh đề ở {len(co)}/{len(ws)} lượt")
    log("    (trần cứng là 25 từ trong ctig/agents/vietrepair.py; sát trần nghĩa là mệnh đề đang bị cắt)")

    # --- điểm Mistral: chỉ chẩn đoán
    diem = [tim_diem_mistral(r) for r in recs]
    co_diem = [d for d in diem if d]
    if co_diem:
        log(f"\n  CHẨN ĐOÁN (KHÔNG PHẢI THƯỚC KẾT LUẬN): tìm thấy điểm bộ chấm Mistral ở {len(co_diem)}/{n}"
            " bản ghi.\n    Vòng sửa của M tối ưu thẳng vào bộ chấm này nên báo cáo nó là hệ thống tự chấm"
            " mình. Chỉ được dùng để\n    xem vòng sửa có 'tin' là mình đã sửa được không, tuyệt đối không"
            " đưa vào bảng kết quả.")

    return {"n": n, "noop": {"n": len(noop), "ti_le": ti_le_noop, "ly_do": dict(ly_do)},
            "critic": {"loi_tb": statistics.mean(so_loi) if so_loi else None, "n_luot": len(so_loi),
                       "n_khong_chay": khong_chay, "contract_id": dict(cids.most_common())},
            "md5_M_trung_B": {"trung": trung, "khac": khac, "thieu": thieu,
                              "ti_le": (trung / n_so) if n_so else None,
                              "bug_noop_ma_khac": sai_noop, "bug_khong_noop_ma_trung": sai_khong_noop},
            "do_dai_menh_de": do_dai,
            "co_diem_mistral": bool(co_diem)}


# =====================================================================  PHẦN 4 — bảng markdown
def phan_4_bang(ket1: dict | None, ket2: dict | None, ket3: dict, log=print) -> str:
    """Bảng markdown gọn để dán thẳng vào bài. In ra stdout và trả về chuỗi để ghi vào tệp -o."""
    d: list[str] = []
    d.append("### Bảng 1 — ba khẳng định, theo nhãn người\n")
    if ket1 and ket1.get("doi_chung"):
        d.append("| phép so | ý nghĩa | M thắng–thua–hoà | tỉ lệ thắng (bỏ hoà) | KTC 90% theo cụm prompt | kết luận |")
        d.append("|---|---|---|---|---|---|")
        for a, b, y in DOI_CHUNG:
            x = ket1["doi_chung"].get(f"{a}_{b}")
            if not x or not x["n_quyet_dinh"]:
                d.append(f"| {a} vs {b} | {y} | – | – | – | không có nhãn |")
                continue
            lo, hi = x["ktc90"]
            d.append(f"| {a} vs {b} | {y} | {x['thang']}–{x['thua']}–{x['hoa']} | {x['ti_le_thang']:.3f} "
                     f"| {lo:.3f} – {hi:.3f} | "
                     + ("**có bằng chứng**" if x["can_duoi_vuot_0_5"] else "chưa đủ bằng chứng") + " |")
        tq = ket1.get("tu_nhat_quan") or {}
        d.append("")
        d.append(f"Độ tự nhất quán của người gán: "
                 + (f"**{tq['ti_le']:.3f}** ({tq['khop']}/{tq['n']} cặp lặp)"
                    + ("  ⚠ dưới 0,7" if tq["ti_le"] < NGUONG_TU_NHAT_QUAN else "")
                    if tq.get("ti_le") is not None else "không đo được (không có cặp lặp)"))
    else:
        d.append("_Chưa có nhãn người._ Không có bảng này thì bài KHÔNG kết luận được về ba khẳng định:"
                 " hai thước tự động ở Bảng 2 chưa được chứng minh là bám theo phán đoán của người về tính"
                 " đúng văn hoá.")

    d.append("\n### Bảng 2 — thước tự động (bổ trợ)\n")
    if ket2 and ket2.get("tong_ket"):
        tk = ket2["tong_ket"]
        d.append("| nhánh | mô tả | DINOv2-SIM ↑ | CLIP prompt gốc ↑ |")
        d.append("|---|---|---|---|")
        for k in ARMS:
            c = tk[k]["clip"]
            d.append(f"| **{k}** | {MO_TA_NHANH[k]} | {tk[k]['sim']:.4f} ± {tk[k]['sim_sd']:.3f} | "
                     + (f"{c:.4f} |" if c is not None else "– |"))
        cap = ket2.get("tung_cap") or {}
        if cap:
            d.append("")
            d.append("| hiệu số từng cặp (SIM) | trung bình | KTC 90% theo cụm prompt | M hơn ở | ảnh trùng nhau |")
            d.append("|---|---|---|---|---|")
            for a, b, _ in DOI_CHUNG:
                x = cap.get(f"sim_{a}_{b}")
                if not x:
                    continue
                lo, hi = x["ktc90"]
                # Cột cuối là số cặp có hiệu ĐÚNG BẰNG 0, tức hai nhánh ra cùng một ảnh (M no-op thì ảnh M
                # trùng byte ảnh B). Không in cột này thì người đọc tưởng hiệu số nhỏ là "hiệu ứng yếu",
                # trong khi thật ra phần lớn cặp đơn giản là KHÔNG CÓ can thiệp nào để mà đo.
                d.append(f"| {a} − {b} | {x['hieu_tb']:+.4f} | {lo:+.4f} – {hi:+.4f} |"
                         f" {x['thang']}/{x['n']} cặp | {x['bang_dung_0']}/{x['n']} |")
        n_cum = len({r["prompt_id"] for r in ket2.get("rows") or []})
        if n_cum:
            d.append(f"\n_n = {len(ket2['rows'])} cặp (prompt, rep) trên {n_cum} cụm prompt."
                     + ("  ⚠ cỡ mẫu nhỏ." if n_cum < IT_CUM else "") + "_")
    else:
        d.append("_Chưa chấm được thước tự động._")

    k3 = ket3
    d.append("\n### Bảng 3 — chẩn đoán vòng sửa\n")
    d.append("| chỉ số | giá trị |")
    d.append("|---|---|")
    d.append(f"| tỉ lệ M no-op | **{k3['noop']['ti_le']:.1%}** ({k3['noop']['n']}/{k3['n']}) |")
    mt = k3["md5_M_trung_B"]
    d.append(f"| ảnh M trùng byte ảnh B | "
             + (f"{mt['ti_le']:.1%} ({mt['trung']}/{mt['trung'] + mt['khac']})" if mt["ti_le"] is not None
                else "không so được") + " |")
    cr = k3["critic"]
    d.append(f"| lỗi Critic tìm được / ảnh | "
             + (f"{cr['loi_tb']:.2f}" if cr["loi_tb"] is not None else "–") + " |")
    top = list(cr["contract_id"].items())[:3]
    d.append("| contract_id bị viện dẫn nhiều nhất | "
             + (", ".join(f"`{c}` ×{n}" for c, n in top) if top else "–") + " |")
    for arm in ("T", "S", "M"):
        x = k3["do_dai_menh_de"][arm]
        d.append(f"| độ dài mệnh đề sửa {arm} | "
                 + (f"{x['tb']:.1f} từ ({x['n_co']}/{x['n']} lượt có mệnh đề)" if x["tb"] is not None
                    else "không có mệnh đề nào") + " |")
    if mt["bug_noop_ma_khac"] or mt["bug_khong_noop_ma_trung"]:
        d.append(f"\n⚠ **Bất nhất:** {len(mt['bug_noop_ma_khac'])} lượt no-op mà ảnh khác,"
                 f" {len(mt['bug_khong_noop_ma_trung'])} lượt không no-op mà ảnh trùng. Sửa bug trước khi báo số.")
    if k3["noop"]["ti_le"] >= 0.5:
        d.append(f"\n⚠ **M no-op ở {k3['noop']['ti_le']:.0%} số lượt**, nghĩa là M trả về đúng ảnh của B ở phần"
                 " lớn prompt. Mọi so sánh M vs B phải đọc kèm con số này.")
    d.append("\n_Điểm của bộ chấm Mistral KHÔNG được dùng ở bất kỳ bảng nào bên trên: vòng sửa của M tối ưu"
             " thẳng vào nó._")

    txt = "\n".join(d)
    log("\n" + "=" * 78)
    log("PHẦN 4 — BẢNG GỌN ĐỂ DÁN VÀO BÀI")
    log("=" * 78 + "\n")
    log(txt)
    return txt


# =====================================================================  main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phân tích thí nghiệm bốn nhánh B/T/S/M của CTIG.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+", help="thư mục run chứa <prompt_id>_r<rep>/arms.json")
    ap.add_argument("--labels", default=None,
                    help="JSON nhãn người do scripts/label_pairs.py sinh ra; thiếu thì bỏ phần 1")
    ap.add_argument("--config", default=None,
                    help="YAML cấu hình; thiếu thì bỏ phần 2 (phần 1/3/4 vẫn chạy, không cần GPU)")
    ap.add_argument("-o", "--out", default=None, help="ghi toàn bộ kết quả ra JSON")
    ap.add_argument("--set", action="append", default=[], help="ghi đè cấu hình, vd --set multigen.device=cpu")
    ap.add_argument("--ids", default=None, help="chỉ phân tích các prompt này, cách nhau bằng dấu phẩy")
    ap.add_argument("--n-eval", type=int, default=5, help="số ảnh thật cất riêng mỗi prompt (phần 2)")
    ap.add_argument("--top", type=int, default=2, help="trung bình top-k ảnh thật giống nhất (phần 2)")
    ap.add_argument("--n-boot", type=int, default=5000, help="số vòng bootstrap")
    ap.add_argument("--seed", type=int, default=0, help="seed bootstrap, để chạy lại ra đúng số cũ")
    ap.add_argument("--skip-auto", action="store_true", help="bỏ phần 2 dù có --config (chạy máy không GPU)")
    ap.add_argument("--md", default=None, help="ghi riêng bảng markdown của phần 4 ra tệp này")
    a = ap.parse_args(argv)

    log = lambda *x, **kw: print(*x, flush=True, **kw)  # noqa: E731

    log("Đọc arms.json:")
    recs = doc_arms(a.run_dir, log)
    if a.ids:
        want = {i.strip() for i in a.ids.split(",") if i.strip()}
        recs = [r for r in recs if r["prompt_id"] in want]
        if not recs:
            raise SystemExit("--ids không khớp prompt nào trong các run đã cho")
    n_cum = len({r["prompt_id"] for r in recs})
    log(f"  -> {len(recs)} bản ghi (prompt, rep) trên {n_cum} cụm prompt")
    if n_cum < IT_CUM:
        log(f"  CẢNH BÁO CỠ MẪU: chỉ {n_cum} cụm prompt. Đơn vị thống kê là PROMPT (hai lần lặp của cùng một"
            f"\n  prompt không độc lập), nên cỡ mẫu thật là {n_cum} chứ không phải {len(recs)}.")

    # ---- phần 1
    ket1 = None
    if a.labels:
        p = Path(a.labels)
        if not p.exists():
            log(f"\nKHÔNG thấy tệp nhãn {p} -> bỏ phần 1, chạy tiếp phần còn lại.")
        else:
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                labels = raw if isinstance(raw, list) else (raw.get("rows") or raw.get("pairs") or [])
            except Exception as exc:  # noqa: BLE001
                log(f"\nĐỌC HỎNG tệp nhãn {p} ({type(exc).__name__}: {exc}) -> bỏ phần 1.")
                labels = None
            if labels:
                if a.ids:
                    labels = [r for r in labels if str(r.get("prompt_id")) in want]
                ket1 = phan_1_nhan_nguoi(labels, a.n_boot, a.seed, log)
            elif labels is not None:
                log(f"\nTệp nhãn {p} rỗng -> bỏ phần 1.")
    else:
        log("\n(không truyền --labels -> bỏ PHẦN 1. Không có nhãn người thì bài chưa kết luận được về ba"
            "\nkhẳng định, vì hai thước tự động chưa được chứng minh là bám theo phán đoán của người.)")

    # ---- phần 2
    ket2 = None
    if a.config and not a.skip_auto:
        try:
            from ctig.config import Config, set_dotted            # noqa: PLC0415
            ov: dict = {}
            for kv in a.set:
                k, _, v = kv.partition("=")
                set_dotted(ov, k, v)
            cfg = Config.load(a.config, ov)
            ket2 = phan_2_thuoc_tu_dong(recs, cfg, a.n_eval, a.top, a.n_boot, a.seed, log)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            import traceback
            log(f"\nPHẦN 2 HỎNG ({type(exc).__name__}: {exc}) -> bỏ qua, phần 3/4 vẫn chạy.")
            traceback.print_exc()
    else:
        log("\n(bỏ PHẦN 2: " + ("--skip-auto" if a.skip_auto else "không truyền --config") + ")")

    # ---- phần 3 + 4
    ket3 = phan_3_chan_doan(recs, log)
    md = phan_4_bang(ket1, ket2, ket3, log)

    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md + "\n", encoding="utf-8")
        log(f"\n-> {a.md}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {"run_dir": a.run_dir, "n_ban_ghi": len(recs), "n_cum_prompt": n_cum,
             "nhan_nguoi": ket1, "thuoc_tu_dong": ket2, "chan_doan": ket3, "markdown": md},
            ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"-> {a.out}")


if __name__ == "__main__":
    main()
