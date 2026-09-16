"""Sinh prompt tinh chỉnh bằng Culture-TRIP (NAACL 2025) cho bộ prompt văn hoá Việt Nam.

    python scripts/culture_trip_prompts.py --repo /workspace/baselines/Culture-TRIP \
        --prompts data/prompts_simple.json --out data/culture_trip [--ids S001,S002] [--limit 10]

Mã Culture-TRIP KHÔNG được chép vào repo này (repo của họ không có file LICENSE ⇒ mặc định giữ toàn quyền).
Script nạp theo đường dẫn lúc chạy qua `--repo`, chỉ lưu lại ĐẦU RA.

Hai chỗ lệch so với bài gốc, bắt buộc khai báo khi viết bài:

1. **LLM**: bài gốc dùng `llama3:70b`; bản lượng tử hoá ~40 GB không vừa đĩa máy thuê nên chạy `llama3:8b`
   (`--model`). Nhánh không-loop và nhánh có-loop dùng chung câu prompt này nên hiệu số giữa hai nhánh không
   bị ảnh hưởng; chỉ con số tuyệt đối của Culture-TRIP bị hạ thấp so với bài gốc.
2. **Prompt nhiều thực thể**: giao diện của họ nhận MỘT culture noun. Với câu phức ta **nối chuỗi** — đầu ra
   lượt trước làm BASE PROMPT cho lượt sau. Đây là mở rộng của ta, không phải của bài gốc; kết quả câu đơn và
   câu phức phải báo cáo tách riêng.

`GoogleSearchAPIWrapper` của họ cần GOOGLE_API_KEY + GOOGLE_CSE_ID. Không có thì script thay bằng Serper
(SERPER_API_KEY), không có nữa thì chỉ dùng Wikipedia. Dùng cách nào cũng được ghi vào trường `search_backend`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def install_search_shim(log=print) -> str:
    """Thay GoogleSearchAPIWrapper bằng Serper khi thiếu khoá Google. Trả về tên backend đã dùng."""
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        return "google_cse"
    import langchain_community.utilities as U

    key = os.getenv("SERPER_API_KEY")

    class _Shim:
        def run(self, q: str) -> str:
            if not key:
                return ""
            try:
                import requests

                r = requests.post("https://google.serper.dev/search",
                                  json={"q": q, "num": 5, "gl": "vn", "hl": "en"},
                                  headers={"X-API-KEY": key, "Content-Type": "application/json"}, timeout=30)
                if r.status_code != 200:
                    return ""
                return " ".join(x.get("snippet", "") for x in r.json().get("organic", [])[:5])
            except Exception:  # noqa: BLE001
                return ""

    U.GoogleSearchAPIWrapper = _Shim
    name = "serper" if key else "wikipedia_only"
    log(f"[shim] không có khoá Google CSE -> tìm kiếm web dùng {name}")
    return name


def load_culture_trip(repo: str, model: str, log=print):
    """Nạp hàm culture_trip() từ checkout của họ; ép LLM sang `model`."""
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)                       # utils/custom_wiki.py và .env đọc theo thư mục hiện hành
    try:
        backend = install_search_shim(log)
        import iterative_refinement.iterative_refinement as IR

        if getattr(IR.llm, "model", None) != model:
            from langchain_community.chat_models import ChatOllama

            IR.llm = ChatOllama(model=model)
            from langchain_core.output_parsers import StrOutputParser
            from iterative_refinement.prompt_templates import refine_prompt, scoring_prompt, feedback_prompt

            IR.refine_llm = refine_prompt | IR.llm | StrOutputParser()
            IR.scoring_llm = scoring_prompt | IR.llm | StrOutputParser()
            IR.feedback_llm = feedback_prompt | IR.llm | StrOutputParser()
            log(f"[culture-trip] LLM ép về {model} (bài gốc dùng llama3:70b)")
        from iterative_refinement.graph_workflow import culture_trip

        return culture_trip, backend, repo
    finally:
        os.chdir(cwd)


def refine_one(culture_trip, repo: str, nouns: list[str], prompt_en: str, threshold: int, log=print):
    """Nối chuỗi qua từng culture noun; trả (câu cuối, [bước]) ."""
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        cur, steps = prompt_en, []
        for noun in nouns:
            t0 = time.time()
            try:
                out = culture_trip(noun, cur, threshold, False)
                out = " ".join(str(out).split())
            except Exception as exc:  # noqa: BLE001
                log(f"    [{noun}] LỖI {type(exc).__name__}: {str(exc)[:90]} -> giữ câu trước")
                steps.append({"culture_noun": noun, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                              "seconds": round(time.time() - t0, 1), "out": cur})
                continue
            steps.append({"culture_noun": noun, "in": cur, "out": out, "words": len(out.split()),
                          "seconds": round(time.time() - t0, 1)})
            log(f"    [{noun}] {len(cur.split())} -> {len(out.split())} từ, {time.time() - t0:.0f}s")
            if out:
                cur = out
        return cur, steps
    finally:
        os.chdir(cwd)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="thư mục checkout Culture-TRIP (không chép vào repo này)")
    ap.add_argument("--prompts", action="append", default=None,
                    help="tệp prompt; lặp lại được. Mặc định cả simple lẫn complex")
    ap.add_argument("--out", default="data/culture_trip")
    ap.add_argument("--model", default="llama3:8b")
    ap.add_argument("--threshold", type=int, default=40, help="ngưỡng điểm dừng, mặc định 40 như bài gốc")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="làm lại cả prompt đã có tệp")
    a = ap.parse_args(argv)
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    files = a.prompts or ["data/prompts_simple.json", "data/prompts_complex.json"]
    ids = {i.strip() for i in a.ids.split(",")} if a.ids else None
    recs = []
    for f in files:
        p = f if Path(f).is_absolute() else ROOT / f
        recs += [r for r in json.loads(Path(p).read_text(encoding="utf-8")) if not ids or r["id"] in ids]
    if a.limit:
        recs = recs[: a.limit]
    out_dir = ROOT / a.out if not Path(a.out).is_absolute() else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    culture_trip, backend, repo = load_culture_trip(a.repo, a.model, log)
    log(f"{len(recs)} prompt · model {a.model} · ngưỡng {a.threshold} · tìm kiếm {backend} · ra {out_dir}")

    done = skip = fail = 0
    t_all = time.time()
    for r in recs:
        dst = out_dir / f"{r['id']}.json"
        sig = hashlib.sha1(f"{r['text_en']}|{a.model}|{a.threshold}".encode()).hexdigest()[:12]
        if dst.exists() and not a.force:
            try:
                if json.loads(dst.read_text(encoding="utf-8")).get("sig") == sig:
                    skip += 1
                    continue
            except Exception:  # noqa: BLE001
                pass
        nouns = [n for n in (r.get("entities") or []) if n]
        if not nouns:
            log(f"[{r['id']}] không có culture noun -> bỏ qua")
            fail += 1
            continue
        log(f"[{r['id']}] {len(nouns)} thực thể: {', '.join(nouns)}")
        t0 = time.time()
        refined, steps = refine_one(culture_trip, repo, nouns, r["text_en"], a.threshold, log)
        ok = bool(refined) and refined != r["text_en"]
        dst.write_text(json.dumps({
            "prompt_id": r["id"], "sig": sig, "model": a.model, "threshold": a.threshold,
            "search_backend": backend, "chained": len(nouns) > 1,
            "prompt_en": r["text_en"], "culture_nouns": nouns,
            "refined_prompt": refined, "words_in": len(r["text_en"].split()), "words_out": len(refined.split()),
            "per_step": steps, "seconds": round(time.time() - t0, 1),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
        if not ok:
            fail += 1
            log(f"[{r['id']}] CẢNH BÁO: câu không đổi so với gốc")
    log(f"xong: {done} mới, {skip} bỏ qua (đã có), {fail} có vấn đề · {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
