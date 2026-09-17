"""Bốn phép kiểm rẻ chạy TRƯỚC lô chính, để không đốt một đêm máy cho một bảng không đọc được.

    python scripts/preflight.py --config configs/vast_arms.yaml --labels data/labels/labels_huy.json \
        --runs /workspace/runs/pilotC [--set llm.backend=mistral_vl]

Giới hạn phải nhớ: mấy phép này **chỉ phát hiện được hệ thống hỏng, không chứng minh được hệ thống tốt**.
Đạt hết nghĩa là "chưa thấy hỏng, đáng chạy tiếp", không phải "chắc thắng". Mẫu chỉ 12 ảnh, 4 cặp, từ MỘT
prompt duy nhất (S002 không có ảnh nào người bảo đúng, S003 không có ảnh nào người bảo sai).

  P1 LẶP LẠI     chấm cùng một ảnh 3 lần. Lệch > 1,0 điểm thì mọi thiết kế phía trên đều vô nghĩa.
  P2 ĐỐI CHỨNG   ghép hai ảnh THẬT cùng nền văn hoá, hỏi cái nào Việt hơn. Phải ra ~0,50. Ra 0,8 nghĩa là
                 bộ chấm đọc độ nét hoặc vết nén chứ không đọc nội dung — và khi đó phép thử A/B cũng vô nghĩa.
  P3 ÉP CHỌN     identity_p sau khi xáo vị trí đáp án còn bằng 1,00 không. Còn thì nó bão hoà thật, bỏ được.
  P4 CHẤM LẠI    chấm lại đúng 12 ảnh đã có nhãn người, bằng mã đã sửa. Đây là phép quan trọng nhất.
                 SỐ HOÀ là chỉ báo nhạy nhất: hoà chính là thứ khiến hệ thống giữ ảnh sai ở 2/3 prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.agents import copilot  # noqa: E402
from ctig.config import Config, set_dotted  # noqa: E402
from ctig.pipeline import load_prompts  # noqa: E402
from ctig.session import Session  # noqa: E402

MOC = {"hoa": 0, "lech_lap": 1.0, "doi_chung": (0.35, 0.65)}


def load_loop(runs: list[str]) -> dict[str, dict]:
    out = {}
    for run in runs:
        for lf in sorted(Path(run).glob("*/loop_v2.json")):
            out[lf.parent.name] = json.loads(lf.read_text())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)

    ov: dict = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        set_dotted(ov, k, v)
    cfg = Config.load(a.config, ov)
    runs = [x.strip() for x in a.runs.split(",") if x.strip()]
    loops = load_loop(runs)
    if not loops:
        raise SystemExit(f"không thấy loop_v2.json nào trong {runs}")
    prompts = {p.id: p for p in load_prompts(cfg.prompts_path)}
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    quiet = lambda *x: None  # noqa: E731

    pid0 = sorted(loops)[0]
    s = Session(cfg, prompts[pid0], run_dir=Path(cfg.runs_dir) / "preflight", log=quiet)
    agent = s.agent
    ket = {}

    # ---------------------------------------------------------------- P1 lặp lại
    log("\n=== P1 · CHẤM CÙNG MỘT ẢNH BA LẦN ===")
    p1 = []
    for pid in sorted(loops)[:2]:
        d = loops[pid]
        img = (d.get("kept") or [None])[0]
        if not img or not Path(img).exists():
            continue
        sc = [copilot.evaluate(agent, img, d["report"], d.get("refs") or [], log=quiet) for _ in range(3)]
        cul = [e.axes.get("culture", float("nan")) for e in sc]
        tot = [e.overall for e in sc]
        lech = max(tot) - min(tot)
        p1.append(lech)
        log(f"  {pid}  văn hoá {['%.1f' % c for c in cul]}  tổng {['%.2f' % t for t in tot]}  lệch {lech:.2f}")
    ket["p1_lech_max"] = max(p1) if p1 else None

    # ---------------------------------------------------------------- P2 đối chứng rỗng
    log("\n=== P2 · ĐỐI CHỨNG: HAI ẢNH THẬT CÙNG NỀN VĂN HOÁ ===")
    p2 = []
    for pid in sorted(loops):
        sp = Session(cfg, prompts[pid], run_dir=Path(cfg.runs_dir) / "preflight", log=quiet)
        sp._agent = agent
        reals = sp.prompt_refs(include_candidates=True)
        if len(reals) < 2:
            continue
        ent = (loops[pid].get("report") or {}).get("entity_en") or pid
        q = (f"One of these two images shows a more authentic {ent}.\n"
             "A. the first image\nB. the second image\nAnswer with a single letter, A or B.")
        fn = getattr(getattr(agent, "llm", None), "choice_prob", None)
        if fn is None:
            log("  backend không hỗ trợ choice_prob -> bỏ qua"); break
        r1, r2 = reals[0], reals[1]
        pa = [float(x) for x in fn(q, [r1, r2], ("A", "B"))]
        pb = [float(x) for x in fn(q, [r2, r1], ("A", "B"))]
        p_r1 = (pa[0] + pb[1]) / 2
        p2.append(p_r1)
        log(f"  {pid}  P(chọn ảnh thật thứ nhất) = {p_r1:.2f}   (thứ tự 1: {pa[0]:.2f} · thứ tự 2: {pb[1]:.2f})")
    ket["p2_trung_binh"] = sum(p2) / len(p2) if p2 else None
    ket["p2_lech_khoi_05"] = max(abs(x - 0.5) for x in p2) if p2 else None

    # ---------------------------------------------------------------- P3 + P4 chấm lại 12 ảnh
    log("\n=== P3+P4 · CHẤM LẠI ẢNH ĐÃ CÓ NHÃN NGƯỜI ===")
    rows = json.loads(Path(a.labels).read_text(encoding="utf-8"))
    human = {}
    for r in rows:
        h = human.setdefault(r["image"], {"pid": r["prompt_id"], "overall": None})
        if r.get("overall"):
            h["overall"] = r["overall"]

    moi, ids = {}, []
    log(f"  {'prompt':7s} {'người':>8s} {'văn hoá':>8s} {'tổng':>6s} {'ident':>6s}  ảnh")
    for img, h in sorted(human.items(), key=lambda kv: (kv[1]["pid"], kv[0])):
        pid = h["pid"]
        d = loops.get(pid)
        if not d or not Path(img).exists():
            log(f"  {pid:7s} thiếu ảnh hoặc hồ sơ -> bỏ"); continue
        ev = copilot.evaluate(agent, img, d["report"], d.get("refs") or [], log=quiet)
        moi[img] = ev
        ids.append(ev.identity_p)
        log(f"  {pid:7s} {str(h['overall'] or '-'):>8s} "
            f"{(('%.1f' % ev.axes['culture']) if 'culture' in ev.axes else '  -  '):>8s} "
            f"{ev.overall:6.2f} {ev.identity_p:6.2f}  {Path(img).parent.parent.name}")
    ket["p3_identity_min"] = min(ids) if ids else None
    ket["p3_identity_max"] = max(ids) if ids else None

    def auc(key):
        by_p = defaultdict(list)
        for img, ev in moi.items():
            o = human[img]["overall"]
            v = ev.axes.get("culture") if key == "culture" else ev.overall
            if o in ("dung", "sai") and v is not None:
                by_p[human[img]["pid"]].append((o == "dung", v))
        w = t = l = 0
        for items in by_p.values():
            for _, sg in [x for x in items if x[0]]:
                for _, sb in [x for x in items if not x[0]]:
                    w += sg > sb; t += sg == sb; l += sg < sb
        n = w + t + l
        return (None, 0, 0, 0) if not n else ((w + 0.5 * t) / n, w, t, l)

    for key, ten in (("culture", "trục văn hoá"), ("overall", "điểm tổng")):
        v, w, t, l = auc(key)
        ket[f"p4_auc_{key}"], ket[f"p4_hoa_{key}"] = v, t
        log(f"\n  AUC {ten:14s} {('%.2f' % v) if v is not None else '-'}   "
            f"(thắng {w} · HOÀ {t} · thua {l})")

    # ảnh hệ thống sẽ giữ theo luật hiện tại, so với nhãn người
    sai = 0
    for pid, d in loops.items():
        cand = [(i, moi[i]) for i in (d.get("kept") or []) if i in moi]
        if not cand:
            continue
        keep = max(cand, key=lambda x: x[1].overall)[0]
        if human.get(keep, {}).get("overall") == "sai":
            sai += 1
    ket["p4_giu_anh_sai"] = sai
    log(f"  Hệ thống sẽ giữ ảnh người bảo SAI ở {sai}/{len(loops)} prompt")

    # ---------------------------------------------------------------- kết luận
    log("\n" + "=" * 74)
    log(f"{'phép':28s} {'kết quả':>12s}  {'mốc':>14s}  đạt?")
    hang = [
        ("P1 lệch khi chấm lặp", ket["p1_lech_max"], f"< {MOC['lech_lap']}",
         ket["p1_lech_max"] is not None and ket["p1_lech_max"] < MOC["lech_lap"]),
        ("P2 lệch khỏi 0,50", ket["p2_lech_khoi_05"], "< 0,15",
         ket["p2_lech_khoi_05"] is not None and ket["p2_lech_khoi_05"] < 0.15),
        ("P4 số cặp HOÀ (văn hoá)", ket["p4_hoa_culture"], "= 0", ket["p4_hoa_culture"] == 0),
        ("P4 giữ ảnh người bảo sai", ket["p4_giu_anh_sai"], "<= 1", ket["p4_giu_anh_sai"] <= 1),
    ]
    for ten, val, moc, ok in hang:
        sval = f"{val:.2f}" if isinstance(val, float) else str(val)
        log(f"{ten:28s} {sval:>12s}  {moc:>14s}  {'ĐẠT' if ok else 'KHÔNG'}")
    log(f"\nidentity_p sau khi xáo vị trí: {ket['p3_identity_min']:.2f} … {ket['p3_identity_max']:.2f}"
        + ("  -> vẫn bão hoà, bỏ khỏi công thức được"
           if (ket["p3_identity_max"] or 0) - (ket["p3_identity_min"] or 0) < 0.05 else
           "  -> CÓ phân biệt, giữ lại"))
    n_dat = sum(1 for *_, ok in hang if ok)
    log(f"\n{n_dat}/4 phép đạt. " + ("Chưa thấy hỏng — đáng chạy lô chính." if n_dat == 4 else
        "CHƯA nên chạy lô 24 prompt qua đêm; sửa chỗ trượt trước."))
    log("(Mẫu chỉ 12 ảnh, 4 cặp, từ MỘT prompt. Đạt hết KHÔNG có nghĩa là hệ thống tốt, chỉ là chưa thấy hỏng.)")

    if a.out:
        Path(a.out).write_text(json.dumps(ket, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"-> {a.out}")


if __name__ == "__main__":
    main()
