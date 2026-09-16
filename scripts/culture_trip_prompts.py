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
import re
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


#: Các mốc mà LLM nhỏ hay chèn vào; mọi thứ từ mốc này trở đi không còn là prompt nữa.
_TAIL = ("### refine feedback", "### feedback", "### score", "### evaluation", "### note",
         "refine feedback:", "**clarity", "**visual detail", "**background", "**purpose",
         "**comparable object", "**total")
_HEAD = ("### refined prompt:", "### refined prompt", "refined prompt:", "answer:")
_PREFIX = ("here is the refined prompt based on the feedback:", "here is the refined prompt:",
           "here's the refined prompt:", "sure, here is the refined prompt:")
#: Câu tự thuật ở cuối: "I added more sensory descriptions…", "This refined prompt improves…". Không phải prompt.
_META = ("i added", "i also added", "i have added", "i included", "i made", "i changed", "i rewrote",
         "i expanded", "i incorporated", "i provided", "this refined prompt", "the refined prompt",
         "note:", "in this version", "these changes", "this should improve", "by adding")


def clean_refined(text: str) -> tuple[str, bool]:
    """Bóc phần prompt thật ra khỏi lời rào và phần feedback/điểm mà LLM nhỏ nhả kèm. (câu sạch, có phải cắt không).

    llama3:8b không giữ đúng khuôn của bài gốc: nó trả về "Here is the refined prompt… ### Refined Prompt: …
    ### Refine Feedback: … **Clarity (10/10)**…" trong CÙNG một chuỗi, 469 từ cho câu vào 13 từ. Dùng nguyên
    chuỗi đó làm nhánh baseline là dựng bù nhìn. Đây là sửa PHẦN ĐỌC KẾT QUẢ, không đụng vào phương pháp của
    họ; với 70B như bài gốc thì khả năng cao không cần bước này. Phải khai báo là có hậu xử lý.
    """
    t = " ".join((text or "").split())
    if not t:
        return "", False
    low = t.lower()
    cut = False
    for h in _HEAD:                       # lấy phần SAU nhãn "### Refined Prompt:"
        i = low.find(h)
        if i >= 0:
            t = t[i + len(h):].strip()
            low = t.lower()
            cut = True
            break
    else:
        for pre in _PREFIX:               # không có nhãn thì bóc lời rào đầu câu
            if low.startswith(pre):
                t = t[len(pre):].strip()
                low = t.lower()
                cut = True
                break
    ends = [low.find(m) for m in _TAIL if low.find(m) > 0]
    if ends:
        t = t[: min(ends)].strip()
        cut = True
    t = " ".join(t.split()).strip(" :-*#")
    # Bỏ các CÂU CUỐI mà model tự thuật việc mình vừa sửa; 8B hay viết thêm dù không có nhãn ### nào.
    # Cắt theo câu chứ không theo chuỗi con, để không xén nhầm giữa một câu mô tả.
    parts = re.split(r"(?<=[.!?])\s+", t)
    while parts and any(parts[-1].lower().lstrip().startswith(m) for m in _META):
        parts.pop()
        cut = True
    t = " ".join(parts).strip()
    return t, cut


def clip_tokens(text: str, tok=None) -> int:
    """Số token CLIP thật nếu có transformers, không thì ước lượng theo số từ."""
    if tok is not None:
        try:
            return len(tok(text)["input_ids"])
        except Exception:  # noqa: BLE001
            pass
    return int(len(text.split()) * 1.35)


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
            clean, was_cut = clean_refined(out)
            steps.append({"culture_noun": noun, "in": cur, "out_raw": out, "out": clean,
                          "words_raw": len(out.split()), "words": len(clean.split()), "post_processed": was_cut,
                          "seconds": round(time.time() - t0, 1)})
            log(f"    [{noun}] {len(cur.split())} -> {len(out.split())} từ thô"
                + (f", bóc còn {len(clean.split())} từ" if was_cut else "") + f", {time.time() - t0:.0f}s")
            if clean:
                cur = clean
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

    done = skip = fail = n_over = 0
    t_all = time.time()
    tok = None
    try:                                  # đếm token CLIP thật nếu có; không thì ước lượng theo số từ
        from transformers import CLIPTokenizerFast

        tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    except Exception:  # noqa: BLE001
        log("[cảnh báo] không nạp được tokenizer CLIP -> số token chỉ là ước lượng")
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
        n_tok = clip_tokens(refined, tok)
        dst.write_text(json.dumps({
            "prompt_id": r["id"], "sig": sig, "model": a.model, "threshold": a.threshold,
            "search_backend": backend, "chained": len(nouns) > 1,
            "post_processed": any(s_.get("post_processed") for s_ in steps),
            "prompt_en": r["text_en"], "culture_nouns": nouns,
            "refined_prompt": refined, "words_in": len(r["text_en"].split()), "words_out": len(refined.split()),
            "clip_tokens": n_tok, "over_77_tokens": n_tok > 77,
            "per_step": steps, "seconds": round(time.time() - t0, 1),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        if n_tok > 77:
            n_over += 1
        done += 1
        if not ok:
            fail += 1
            log(f"[{r['id']}] CẢNH BÁO: câu không đổi so với gốc")
    log(f"xong: {done} mới, {skip} bỏ qua (đã có), {fail} có vấn đề · {time.time() - t_all:.0f}s")
    if n_over:
        log(f"[cảnh báo] {n_over}/{done} prompt vượt 77 token của CLIP -> bộ sinh sẽ CẮT phần cuối. "
            "Đây là giới hạn sẵn có của cách bung prompt, phải báo cáo.")


if __name__ == "__main__":
    main()
