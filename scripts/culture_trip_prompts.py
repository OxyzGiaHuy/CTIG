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


#: Mốc bắt đầu phần KHÔNG còn là prompt (feedback, điểm, giải thích). Dò không phân biệt hoa thường.
_TAIL = ("### refine feedback", "### feedback", "### score", "### evaluation", "### note",
         "refine feedback:", "score:", "scores:", "total_score", "{'clarity'", '{"clarity"',
         "**clarity", "**visual detail", "**background", "**purpose", "**comparable object", "**total",
         "the refined prompt aims", "the revised prompt aims", "the refined prompt provides",
         "this refined prompt", "the changes i made", "changes made:")
#: Lời rào mở đầu; bắt bằng regex vì 8B viết mỗi lần một kiểu.
_INTRO = re.compile(
    r"^\s*(?:###\s*)?(?:[^.\"']{0,80}?\b(?:refined|revised)\s+prompt\b[^.:]{0,60}:"
    r"|based on the feedback[^.:]{0,60}:"
    r"|here(?:'s| is)[^.:]{0,60}:"
    r"|sure[,!][^.:]{0,60}:"
    r"|answer\s*:)\s*", re.I)
#: Câu tự thuật ở cuối: "I added more sensory descriptions…". Không phải prompt.
_META = ("i added", "i also added", "i have added", "i included", "i made", "i changed", "i rewrote",
         "i expanded", "i incorporated", "i provided", "i refined", "this refined prompt",
         "the refined prompt", "the revised prompt", "note:", "in this version", "these changes",
         "this should improve", "by adding", "describe the unique")


def clean_refined(text: str) -> tuple[str, bool]:
    """Bóc phần prompt thật ra khỏi lời rào, phần feedback và bảng điểm mà LLM nhỏ nhả kèm.

    Trả (câu sạch, có phải đã cắt không). llama3:8b không giữ khuôn của bài gốc; đã gặp bốn kiểu trên dữ liệu
    thật: nhãn `### Refined Prompt:`, lời rào `Here is the refined prompt…`, prompt nằm trong NGOẶC KÉP rồi
    `SCORE: {...}`, và phần `The refined prompt aims to: 1. 2. 3.` ở cuối. Đây là sửa PHẦN ĐỌC KẾT QUẢ, không
    đụng phương pháp của họ; với 70B như bài gốc nhiều khả năng không cần bước này. Cờ `post_processed` được
    ghi vào tệp đầu ra để khai báo.

    An toàn: nếu bóc xong còn dưới 8 từ thì coi như dò sai và trả lại chuỗi ban đầu.
    """
    raw = " ".join((text or "").split())
    if not raw:
        return "", False
    t, cut = raw, False
    for _ in range(3):                    # lời rào chồng nhau: "Here is the refined prompt: ### Refined Prompt: …"
        m = _INTRO.match(t)
        if not m or m.end() == 0:
            break
        t, cut = t[m.end():].strip(), True
    # prompt nằm trong ngoặc kép -> lấy phần bên trong
    for q in ('"', "\u201c", "'"):
        if t.startswith(q):
            close = t.find('"' if q == '"' else ("\u201d" if q == "\u201c" else "'"), 1)
            if close > 30:
                t, cut = t[1:close].strip(), True
            break
    low = t.lower()
    ends = [low.find(mk) for mk in _TAIL if low.find(mk) > 0]
    if ends:
        t, cut = t[: min(ends)].strip(), True
    parts = re.split(r"(?<=[.!?])\s+", t)
    while parts and any(parts[-1].lower().lstrip().startswith(mk) for mk in _META):
        parts.pop()
        cut = True
    t = re.sub(r"\*{1,3}", " ", " ".join(parts))
    t = " ".join(t.split()).strip(' :-#"\u201c\u201d')
    if len(t.split()) < 8:
        return raw, False                 # dò sai -> giữ nguyên, để còn thấy mà sửa
    return t, cut


_STOP = set("a an the of in on at with and or for to is are was were by from as its their this "
            "that into over under it his her they them".split())
_JUNK = ("refined prompt", "base prompt", "culture noun", "feedback", "clarity", "score",
         "here is", "here's", "based on the", "i refined", "i added", "### ")

EXTRACT_SYS = (
    "You extract the final image-generation prompt from a noisy text produced by a prompt-refinement system.\n"
    "Output ONLY the prompt itself: one paragraph of plain descriptive English about what the picture shows.\n"
    "Remove every heading, label, quote mark, score, criterion, and any sentence that talks about the prompt, "
    "the refinement, the feedback, or the instructions. Do not add anything new. Do not explain. "
    "If the text contains several versions, take the most complete descriptive one."
)


def grounded(out: str, raw: str) -> float:
    """Tỉ lệ từ nội dung của chuỗi bóc ra có mặt trong chuỗi thô. Thấp = LLM bịa thêm."""
    import re as _re

    w = [x for x in _re.findall(r"[a-zà-ỹ]+", out.lower()) if x not in _STOP and len(x) > 3]
    if not w:
        return 0.0
    src = set(_re.findall(r"[a-zà-ỹ]+", raw.lower()))
    return sum(x in src for x in w) / len(w)


def looks_clean(t: str) -> bool:
    low = " " + t.lower()
    return bool(t) and len(t.split()) >= 8 and not any(j in low for j in _JUNK)


def compose(prompt_en: str, refined: str) -> str:
    """Câu cuối = CÂU GỐC + phần mô tả văn hoá mà Culture-TRIP thêm vào.

    Vì sao ghép mọi lúc chứ không chỉ khi mất cảnh: template của chính họ ghi "the scene depicted in the BASE
    PROMPT must remain unchanged", nhưng llama3:8b có lúc viết lại thành bài từ điển về thực thể và bỏ sạch
    cảnh (S001 giữ 0% từ khoá, mất cả "young woman" lẫn "school gate"). Ghép có điều kiện thì mỗi prompt hành
    xử một kiểu, khó mô tả trong bài; ghép đều thì chỉ cần một câu để giải thích, và **hiệu số B−A trở thành
    đúng phần chi tiết văn hoá mà họ thêm vào**, không lẫn chuyện họ sửa hay bỏ cảnh.

    Ghép được vì SDXL và RealVis bật compel `truncate_long_prompts=False` nên prompt dài được nối embedding,
    KHÔNG bị cắt. Riêng SD 3.5 Medium (không T5) vẫn cắt ở 77 token, phải chú thích khi báo cáo hàng đó.
    Câu gốc đặt TRƯỚC để nếu có bị cắt thì cảnh vẫn còn.
    """
    base, add = " ".join((prompt_en or "").split()), " ".join((refined or "").split())
    if not add:
        return base
    if not base or base.lower().rstrip(".") in add.lower():
        return add                        # họ đã giữ nguyên câu gốc -> không lặp lại
    return f"{base.rstrip('.')}. {add}"


def scene_keep(prompt_en: str, refined: str) -> float:
    """Tỉ lệ từ khoá của câu GỐC còn trong câu tinh chỉnh. 1.0 = giữ trọn cảnh."""
    import re as _re

    src = {w for w in _re.findall(r"[a-z]+", prompt_en.lower()) if w not in _STOP and len(w) > 3}
    if not src:
        return 1.0
    got = set(_re.findall(r"[a-z]+", refined.lower()))
    return len(src & got) / len(src)


def extract_prompt_llm(llm, raw: str, log=print) -> str:
    """Nhờ chính LLM bóc câu prompt ra khỏi chuỗi nhiễu. Trả "" nếu không dùng được.

    Regex không đuổi kịp: llama3:8b bọc kết quả mỗi lần một kiểu ("### Refined Prompt:", ngoặc kép rồi
    "SCORE: {...}", "Based on the provided INFORMATION, I refined the BASE PROMPT to…", "The refined prompt
    aims to: 1. 2. 3."). Sau bốn vòng vá regex vẫn còn 6/10 tệp dính rác, nên chuyển sang bóc bằng LLM và giữ
    regex làm lớp trước. Kết quả chỉ được nhận nếu SẠCH và NGẮN HƠN chuỗi vào.
    """
    try:
        r = llm.invoke(EXTRACT_SYS + "\n\nTEXT:\n" + raw + "\n\nPROMPT ONLY:")
        out = " ".join(str(getattr(r, "content", r)).split()).strip(' "\u201c\u201d')
    except Exception as exc:  # noqa: BLE001
        log(f"    [bóc bằng LLM] lỗi {type(exc).__name__}")
        return ""
    if not looks_clean(out) or len(out.split()) > len(raw.split()):
        return ""
    # Chốt chặn quan trọng nhất: chuỗi bóc ra phải BÁM vào chuỗi gốc. S015 có bản thô chỉ chứa feedback, không
    # có prompt nào ("The refined prompt meets the requirements… REFINE FEEDBACK: … ANSWER:"), và LLM đã BỊA ra
    # "A majestic tiger emerges from a misty forest". Yêu cầu phần lớn từ nội dung phải xuất hiện trong bản thô.
    if grounded(out, raw) < 0.6:
        log(f"    [bóc bằng LLM] kết quả không bám bản thô ({grounded(out, raw):.0%}) -> bỏ")
        return ""
    return out


def clip_tokens(text: str, tok=None) -> int:
    """Số token CLIP thật nếu có transformers, không thì ước lượng theo số từ."""
    if tok is not None:
        try:
            return len(tok(text)["input_ids"])
        except Exception:  # noqa: BLE001
            pass
    return int(len(text.split()) * 1.35)


#: Wikimedia siết mạnh User-Agent mặc định của thư viện `wikipedia` ("wikipedia (https://github.com/goldsmith/…)").
#: Sau vài chục truy vấn API trả HTTP 429 kèm text/plain, thư viện gọi r.json() nên nổ JSONDecodeError ở char 0 và
#: Culture-TRIP hỏng TOÀN BỘ (10/10 prompt rơi về câu gốc). Khai tên dự án + giãn nhịp thì hết.
WIKI_UA = "CTIG-research/0.1 (https://github.com/OxyzGiaHuy/CTIG)"


def configure_wikipedia(min_wait_s: float = 1.0, log=print) -> None:
    from datetime import timedelta

    import wikipedia

    wikipedia.set_user_agent(WIKI_UA)
    wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=min_wait_s))
    log(f"[wikipedia] User-Agent riêng + giãn nhịp {min_wait_s}s/truy vấn (tránh HTTP 429)")


_SCORE_KEYS = ("Clarity", "Visual_detail", "Background", "Purpose", "Comparable_object")


def patch_scoring(IR, log=print) -> None:
    """Làm nút chấm điểm của Culture-TRIP chịu được JSON hỏng mà LLM nhỏ hay trả về.

    Mã gốc `scoring()` làm `json.loads` trên cụm `{...}` đầu tiên. llama3:8b trả `{'Clarity': 9.5, …}` dùng
    NHÁY ĐƠN nên `json.loads` vỡ, và 7/10 prompt rơi về câu gốc — tức nhánh Culture-TRIP thành y hệt nhánh
    không có gì, thí nghiệm mất nghĩa. Bản vá chỉ đổi CÁCH ĐỌC số, không đổi câu hỏi chấm điểm, không đổi
    ngưỡng, không đổi luồng. Phải vá TRƯỚC khi import graph_workflow vì nó `from … import scoring`.
    """
    import ast
    import json as _json
    import re as _re

    def scoring(state):
        resp = IR.scoring_llm.invoke({"culture_noun": state["culture_noun"],
                                      "refined_prompt": state["refined_prompt"]})
        score = None
        m = _re.search(r"\{.*?\}", resp, _re.DOTALL)
        if m:
            for parse in (_json.loads, ast.literal_eval):
                try:
                    got = parse(m.group(0))
                    if isinstance(got, dict):
                        score = got
                        break
                except Exception:  # noqa: BLE001
                    continue
        if not isinstance(score, dict):          # lùi tiếp: nhặt từng cặp "tên: số" trong văn bản
            score = {}
            for k in _SCORE_KEYS:
                mm = _re.search(k.replace("_", "[ _]") + r"\D{0,14}?(\d+(?:\.\d+)?)", resp, _re.I)
                if mm:
                    score[k] = float(mm.group(1))
        vals = {}
        for k in _SCORE_KEYS:
            try:
                vals[k] = float(score.get(k, 5))
            except (TypeError, ValueError):
                vals[k] = 5.0
        try:
            vals["Total_score"] = float(score["Total_score"])
        except (KeyError, TypeError, ValueError):
            vals["Total_score"] = sum(vals[k] for k in _SCORE_KEYS)
        sh = state["score_history"]
        for k in sh:
            if k in vals:
                sh[k].append(vals[k])
        return IR.GraphState(score=vals, score_history=sh)

    IR.scoring = scoring
    log("[culture-trip] vá nút chấm điểm: chấp nhận JSON nháy đơn và văn bản tự do (8B hay trả sai khuôn)")


def load_culture_trip(repo: str, model: str, log=print):
    """Nạp hàm culture_trip() từ checkout của họ; ép LLM sang `model`."""
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    cwd = os.getcwd()
    os.chdir(repo)                       # utils/custom_wiki.py và .env đọc theo thư mục hiện hành
    try:
        backend = install_search_shim(log)
        configure_wikipedia(log=log)
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
        patch_scoring(IR, log)           # phải vá TRƯỚC import dưới đây (graph_workflow `from … import scoring`)
        from iterative_refinement.graph_workflow import culture_trip

        return culture_trip, backend, repo
    finally:
        os.chdir(cwd)


def refine_one(culture_trip, repo: str, nouns: list[str], prompt_en: str, threshold: int, log=print, llm=None):
    """Nối chuỗi qua từng culture noun; trả (câu cuối, [bước]) ."""
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        cur, steps = prompt_en, []
        for noun in nouns:
            t0 = time.time()
            try:
                out = ""
                for attempt in range(3):          # 429 vẫn có thể lọt qua -> lùi dần rồi thử lại
                    try:
                        out = " ".join(str(culture_trip(noun, cur, threshold, False)).split())
                        break
                    except Exception as e2:  # noqa: BLE001
                        if attempt == 2 or "JSONDecode" not in type(e2).__name__:
                            raise
                        wait = 10 * (attempt + 1)
                        log(f"    [{noun}] có thể bị siết truy vấn, chờ {wait}s rồi thử lại")
                        time.sleep(wait)
            except Exception as exc:  # noqa: BLE001
                log(f"    [{noun}] LỖI {type(exc).__name__}: {str(exc)[:90]} -> giữ câu trước")
                steps.append({"culture_noun": noun, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                              "seconds": round(time.time() - t0, 1), "out": cur})
                continue
            clean, was_cut = clean_refined(out)
            how = "regex" if was_cut else "nguyên văn"
            if not looks_clean(clean):        # regex không đuổi kịp -> nhờ chính LLM bóc
                alt = extract_prompt_llm(llm, out, log) if llm is not None else ""
                if alt:
                    clean, was_cut, how = alt, True, "llm"
                else:
                    how = "CÒN RÁC"
                    if grounded(clean, out) < 0.6 or len(clean.split()) < 8:
                        clean, how = "", "BẢN THÔ KHÔNG CÓ PROMPT"
                        log(f"    [{noun}] bản thô chỉ có feedback, không có prompt -> giữ câu trước")

            steps.append({"culture_noun": noun, "in": cur, "out_raw": out, "out": clean,
                          "words_raw": len(out.split()), "words": len(clean.split()), "post_processed": was_cut,
                          "cleaned_by": how, "seconds": round(time.time() - t0, 1)})
            log(f"    [{noun}] {len(cur.split())} -> {len(out.split())} từ thô"
                + (f", bóc còn {len(clean.split())} từ [{how}]" if was_cut else f" [{how}]")
                + f", {time.time() - t0:.0f}s")
            steps[-1]["scene_keep_raw"] = round(scene_keep(prompt_en, clean), 2)
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
    import iterative_refinement.iterative_refinement as _IR   # dùng lại đúng LLM đó để bóc prompt

    llm = _IR.llm
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
        refined, steps = refine_one(culture_trip, repo, nouns, r["text_en"], a.threshold, log, llm)
        refined = compose(r["text_en"], refined)
        ok = bool(refined) and refined != r["text_en"]
        n_tok = clip_tokens(refined, tok)
        dst.write_text(json.dumps({
            "prompt_id": r["id"], "sig": sig, "model": a.model, "threshold": a.threshold,
            "search_backend": backend, "chained": len(nouns) > 1,
            "post_processed": any(s_.get("post_processed") for s_ in steps),
            "composed": True,
            "scene_keep_raw": min([s_.get("scene_keep_raw", 1.0) for s_ in steps] or [1.0]),
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
        log(f"[lưu ý] {n_over}/{done} prompt vượt 77 token CLIP. SDXL/RealVis bật compel nên NỐI embedding, "
            "không cắt; riêng SD 3.5 Medium (không T5) sẽ cắt phần cuối -> chú thích khi báo cáo hàng đó.")


if __name__ == "__main__":
    main()
