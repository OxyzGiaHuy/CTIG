# CTIG — Cultural Text-to-Image Generation (Vietnam)

> **Abstract (EN).** Off-the-shelf text-to-image models fail on under-represented cultures not by producing nonsense but by substituting *visually similar artefacts from better-represented cultures*: an áo dài becomes a kimono, bánh chưng becomes zongzi, Tết becomes a Chinese lantern street. CTIG is a retrieval-grounded, agentic pipeline for the Vietnamese domain: an analysis agent expands the prompt into cultural entities, a retriever collects verifiable visual evidence (curated KB + Wikipedia + Commons reference images), the evidence is distilled into a *cultural contract* (must-have / must-not attributes), SDXL generates candidates conditioned on that contract (negative prompts, IP-Adapter reference, optional LoRA), and a VLM + CLIP review loop detects cultural substitution and revises the generation spec. Runs end-to-end on a single Kaggle T4 with no API keys.

Pipeline v1 chạy được **toàn bộ các khối** trên Kaggle GPU và trả về **ảnh thật** cho mỗi prompt. Không cần API key.

```
Input ─► Analysis Agent ─► Search ─► Summary/Filter/Rank ─► Gen (SDXL ×N) ─► Tri giác (VLM+CLIP) ─► Review ─► Eval
prompt   keywords +          kb        CulturalSpec            candidates       Perception             revise    report.html
n=50     keywords mới        wiki      must_have/must_not      IP-Adapter                              ↺ ≤2      user_study.csv
                             image     confusables             LoRA (tuỳ chọn)
```

---

## Chạy nhanh

**Trên Kaggle** (khuyến nghị, xem [docs/KAGGLE.md](docs/KAGGLE.md)): mở `notebooks/kaggle_run.ipynb`, bật GPU T4 và Internet, chạy lần lượt.

**Local không GPU** (để đọc code và test luồng):

```bash
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python tests/test_offline.py
.venv/bin/python -m ctig.cli batch --config configs/offline.yaml --limit 5
```

**Local có GPU ≥ 16 GB:**

```bash
pip install -e ".[gpu]"
python -m ctig.cli run p050 --config configs/kaggle_t4.yaml --set runs_dir=runs
python -m ctig.cli batch --config configs/kaggle_t4.yaml --set runs_dir=runs --limit 10
```

**Cache:** stage 1–3 (analysis, search, spec) được cache theo hash của prompt và cấu hình liên quan. Chạy lại cùng prompt thì bỏ qua ba stage đó, chỉ sinh ảnh và review. `--refresh` để chạy lại, `--no-cache` để tắt. Bằng chứng rút được cache riêng theo thực thể trong `runs/_cache/evidence/<entity_id>.json`, có thể mở ra đọc và duyệt.

Mỗi lần chạy tạo `runs/<run_id>/` gồm `report.html` (mọi prompt, mọi vòng, ảnh và phán quyết), `user_study.csv`, `summary.json`, và với mỗi prompt một thư mục có log JSON từng stage cùng ảnh từng vòng.

---

## Các khối và hiện thực v1

| Khối trong sơ đồ | File | v1 dùng gì |
|---|---|---|
| Input n=50 | `data/prompts_vi.jsonl` | 50 prompt tiếng Việt, có nhãn vàng chỉ dùng để đo recall |
| Analysis Agent → Keywords → Keywords mới | `ctig/stages/analysis.py`, `ctig/llm/prompt_agent.py` | Qwen2.5-VL-3B: tách surface / expanded, viết prompt tiếng Anh; KB alias bù thực thể nêu tên |
| Search: Image / API / text wiki | `ctig/stages/retrieval.py`, `ctig/stages/extraction.py` | Gọi API lúc chạy: Wikipedia VI (tìm bài, lấy toàn văn), Commons (ảnh tải về, CLIP kiểm), Serper tuỳ chọn. **VLM rút must_have / must_not / confusable_with từ văn bản, mỗi thuộc tính kèm trích đoạn gốc.** Thực thể chưa có trong KB được dựng bằng chứng ngay lúc chạy. KB tay chỉ là bằng chứng bổ sung. |
| 1 Summary 2 Filter 3 Rank | `ctig/stages/spec.py`, `ctig/llm/rule_agent.py` | Luật deterministic: gộp, lọc vùng miền, xếp hạng; LLM chỉ dịch thuộc tính sang tiếng Anh |
| Gen: D D LoRA | `ctig/stages/generation.py` | SDXL base + VAE fp16-fix, **N ứng viên** chọn bằng CLIP, negative prompt, IP-Adapter với ảnh tham chiếu, LoRA nếu có |
| *(chưa có trong draft)* Tri giác | `ctig/stages/perception.py` | VLM liệt kê thứ nhìn thấy (hỏi mở) + CLIP probe P(mục tiêu) vs P(confusable) |
| Agent Loop Review, Debate, Reasoning | `ctig/stages/review.py`, `ctig/llm/shared.py` | VLM phê bình theo must_have/must_not; **CLIP làm ý kiến thứ hai**; hoà giải và lập bản sửa bằng luật; ≤ 2 vòng sửa |
| Prompt / Kết quả / Evidence; User study; LLM | `ctig/stages/evaluation.py` | VLM judge 3 trục, CLIP fidelity, retrieval recall, CSV user study, báo cáo HTML |

**Khối tri giác là mũi tên sơ đồ draft chưa có.** Bộ sinh thật chỉ trả pixel; phải có bước biến pixel thành thứ đọc được thì review mới làm việc. Xem [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Debate trong v1** không phải hai LLM tranh luận, mà là **VLM đối chiếu CLIP**: hai tín hiệu độc lập về cùng câu hỏi "ảnh này giống áo dài hay kimono". Bất đồng được ghi vào `adjudication.disagreements` và hiện trong báo cáo.

---

## Cấu hình

Mọi thứ trong `configs/*.yaml`, ghi đè bằng `--set key.sub=value`.

| File | Dùng khi |
|---|---|
| `offline.yaml` | Không GPU, không mạng. Agent luật, bộ sinh stub. Chỉ để test luồng. |
| `kaggle_t4.yaml` | Kaggle 1×T4. VLM + SDXL chung GPU, SDXL bật cpu_offload. ~2–3 phút/prompt/vòng. |
| `kaggle_t4x2.yaml` | Kaggle 2×T4. VLM GPU 0, SDXL GPU 1, ảnh 1024. Nhanh gấp đôi. |
| `kaggle_claude.yaml` | Dùng Claude API làm agent + tri giác (cần `ANTHROPIC_API_KEY`). |

Ba đòn can thiệp của vòng review, ánh xạ sang SDXL:

| Lỗi phát hiện | Thao tác | Trong SDXL |
|---|---|---|
| Vẽ ra thực thể văn hoá khác | thêm negative, đẩy thực thể lên đầu prompt, tăng guidance, dùng ảnh tham chiếu | `negative_prompt`, thứ tự token, `guidance_scale`, IP-Adapter |
| Bỏ sót thực thể | thêm tên và thuộc tính vào prompt, ảnh tham chiếu | prompt, IP-Adapter |
| Prior quá thấp (< 0.20) | gắn LoRA văn hoá | `load_lora_weights` — chỉ khi `t2i.lora_path` được đặt |

**LoRA:** repo không kèm LoRA văn hoá Việt. Đặt `t2i.lora_path` tới một LoRA SDXL (local hoặc HF) nếu bạn có. Không có thì đòn này bị bỏ qua và ghi rõ trong log.

---

## Kết quả

Chưa có số thật. Số duy nhất tồn tại là từ bộ sinh mô phỏng (config `offline.yaml`) và **không nói gì về SDXL**. Lần chạy Kaggle đầu tiên trên 50 prompt sẽ cho: tỉ lệ đạt, CLIP fidelity ở vòng 0 so với vòng cuối (đo tác dụng của vòng review), retrieval recall, và `user_study.csv` để người chấm.

Đọc `report.html` trước khi tin bất kỳ số nào. Nó cho thấy từng vòng đã sửa gì và VLM với CLIP bất đồng ở đâu.

---

## Cấu trúc

```
ctig/
  schema.py          kiểu dữ liệu mọi stage — đọc trước
  config.py          YAML -> Config
  kb.py              knowledge base, khớp thuật ngữ theo biên từ, so khớp thuộc tính
  pipeline.py        nối các stage, ghi log từng stage, ghi tổng hợp sau mỗi prompt
  cli.py
  llm/
    base.py          LLMBackend (chat/complete_json) và Agent (theo nhiệm vụ)
    qwen_vl.py       Qwen2.5-VL local
    anthropic_backend.py
    prompt_agent.py  agent dùng LLM: phân tích, dịch thuộc tính, phê bình, judge
    rule_agent.py    agent luật: offline, baseline
    shared.py        hoà giải VLM+CLIP và lập bản sửa — deterministic
    rules.py         từ điển vùng miền, bối cảnh, luật mở rộng
  stages/
    analysis.py  retrieval.py  spec.py  generation.py  perception.py  review.py  evaluation.py
data/prompts_vi.jsonl     50 prompt
data/kb/entities.json     38 thực thể: must_have / must_not / confusable_with / prior_strength
configs/                  offline, kaggle_t4, kaggle_t4x2, kaggle_claude
notebooks/kaggle_run.ipynb
tests/test_offline.py
docs/ARCHITECTURE.md  docs/KAGGLE.md
```

## Bằng chứng đến từ đâu

Hai lớp, được gộp ở stage 3 và phân biệt bằng `provenance`:

| Lớp | Nguồn | Khi nào |
|---|---|---|
| `extracted` | VLM đọc văn bản Wikipedia / web truy hồi lúc chạy, rút thuộc tính thị giác, **chỉ giữ thuộc tính có trích đoạn gốc** (`attr_sources`) | mọi thực thể, kể cả thực thể chưa có trong KB |
| `kb@...` | file `data/kb/entities.json` do người viết code soạn tay | chỉ 38 thực thể có sẵn |

Mở `runs/_cache/evidence/<entity_id>.json` để xem máy rút ra gì và dựa vào câu nào. So với KB tay là cách rẻ nhất để đánh giá chất lượng bước rút: máy có tìm ra "cổ đứng cao, xẻ tà từ hông" từ bài Wikipedia không, hay chỉ ra được lịch sử.

Web search API: đặt `SERPER_API_KEY` (Kaggle Secrets) và `retrieval.web_api: serper` để thêm snippets Google và Google Images khi Wikipedia không đủ. Không có key thì chỉ dùng Wikipedia + Commons, không cần key nào.

## Hạn chế của v1

* **KB tay chưa được kiểm định** và `prior_strength` là ước lượng. Bằng chứng rút lúc chạy giảm phụ thuộc vào KB nhưng chất lượng phụ thuộc VLM 3B đọc tiếng Việt; hãy đọc vài file trong `runs/_cache/evidence/` trước khi tin.
* **Wikipedia cho ngữ cảnh nhiều hơn thuộc tính thị giác.** Bước rút có thể trả về ít must_have cho thực thể mà bài viết thiên về lịch sử. Serper giúp phần này.
* **Judge không độc lập với reviewer**: cùng một VLM. CLIP là tín hiệu độc lập duy nhất.
* **Qwen2.5-VL-3B nhận dạng văn hoá Việt còn yếu**, đặc biệt các thực thể prior thấp (đàn bầu, nón quai thao). Khi VLM không nhận ra thứ nó đang nhìn, vòng review không thể sửa đúng. Config `kaggle_claude.yaml` là cách kiểm xem giới hạn nằm ở VLM hay ở kiến trúc.
* **Không có LoRA văn hoá Việt sẵn.** Huấn luyện một LoRA (20–50 ảnh/khái niệm, có giấy phép) là công việc thu thập dữ liệu và có thể là đóng góp chính của đề tài.
* **Ảnh Commons** được kiểm bằng CLIP nhưng CLIP cũng có thiên lệch giống T2I; ngưỡng 0.55 là tuỳ chọn ban đầu.
* SDXL không có prompt weighting; "tăng conditioning" hiện thực bằng thứ tự token, lặp lại và guidance. Cân nhắc `compel` cho v2.

## Giấy phép

MIT. Ảnh Commons tải về có giấy phép riêng của từng ảnh, kiểm trước khi dùng cho huấn luyện hay xuất bản.
