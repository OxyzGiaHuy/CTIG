# Kiến trúc CTIG v1

## Luồng dữ liệu

```
Prompt
  │  stage 1  agent.analyze
  ▼
AnalysisResult   keywords[surface|expanded], candidate_entity_ids, region_hint, prompt_en
  │  stage 2  retriever.search        (Wikipedia tìm bài + toàn văn; Commons; Serper tuỳ chọn; ảnh tải về, CLIP kiểm)
  │  stage 2b extraction.run           (VLM: văn bản -> must_have/must_not/confusable, mỗi thuộc tính kèm trích đoạn;
  │                                     cache theo entity_id; thực thể ad-hoc được nạp thuộc tính vào KB bộ nhớ)
  ▼
SearchResult     EvidenceItem[] với must_have, must_not, confusable_with, attr_sources, local_path, clip_match
  │  stage 3  agent.build_spec        (luật: gộp, lọc vùng, xếp hạng; LLM: dịch thuộc tính EN)
  ▼
CulturalSpec     SpecEntity[]: required_attrs(_en), forbidden_attrs(_en), confusables, weight, reference_image
  │
  │  ┌──────────────────── vòng lặp (≤ max_iters + 1 lần sinh) ────────────────────┐
  │  │ stage 4  generator.generate(GenSpec)  ->  GenOutput{candidates[N]}           │
  │  │          select_candidate: CLIP fidelity cao nhất                            │
  │  │ tri giác perceiver.perceive        ->  Perception{elements, caption, clip_probs, checklist}
  │  │          checklist: VLM trả lời câu ĐÓNG cho từng thực thể                    │
  │  │ stage 5  shared.checklist_critique  ->  Critique (LUẬT từ checklist)          │
  │  │          shared.adjudicate          ->  Adjudication (VLM ⊕ CLIP, bất đồng)   │
  │  │          shared.plan_revision       ->  RevisionPlan                          │
  │  │          apply_plan                 ->  GenSpec mới                            │
  │  └───────────────────────────────────────────────────────────────────────────────┘
  ▼
ReviewOutcome
  │  stage 6  evaluation: judge (VLM), CLIP fidelity, recall, CSV, HTML
  ▼
EvalRecord, RunSummary
```

## Cache

* **Stage 1–3 theo prompt**: key = sha1(prompt, llm backend/model, retrieval backend/extract/web_api/wiki_chars,
  max_spec_entities, min_entity_score, KB version). Lưu `runs/_cache/stages/<key>/{analysis,search,spec}.json`.
  Gen và review luôn chạy. `cache.refresh` bỏ qua, `cache.enabled=false` tắt hẳn.
* **Bằng chứng theo thực thể**: `runs/_cache/evidence/<entity_id>.json`, không phụ thuộc prompt. Cùng một thực thể
  ở nhiều prompt chỉ rút một lần.
* **Ảnh tham chiếu**: `runs/_cache/ref_images/<sha1(url)>.jpg`.

## v1.2: Session, multigen và ba lớp cache

```
Session(cfg, prompt)
  .analysis()  .compare()  .retrieve()  .spec()  .genspec()  .multigen(models)  .review()
  mỗi method: bộ nhớ (cùng hash đầu vào) -> đĩa (runs/<run>/<pid>/step_*.json) -> chạy mới; trả (value, source)
  .invalidate(step) xoá bước đó và các bước sau; force=True ép chạy lại một bước
```

| Lớp cache | Khoá | Nơi lưu |
|---|---|---|
| gọi LLM/VLM (`llm/cache.py`) | sha1(backend, model, system, user, sha1 ảnh) | `runs/_cache/llm/` |
| web (`stages/websearch.py::WebClient`) | sha1(loại, truy vấn, vùng, n) | `runs/_cache/web/`, ảnh `ref_images/` |
| artefact bước (`session.py`) | sha1(đầu vào bước + hash bước trước + config liên quan) | `runs/<run>/<pid>/step_*.json` |
| ảnh multigen (`stages/multigen.py`) | (model_key, hash GenSpec, seed) | `runs/<run>/<pid>/<model>/`, `multigen.json` |

Multigen: `adapt_spec` (kích cỡ/bước/guidance/negative theo model, kẹp `max_side`) → nạp một model (`models/loader.py`) →
sinh N ứng viên (`DiffusersGenerator`) → `score_run` (CLIP identity qua `review.select_candidate`, BLIP-2 ITM qua
`ITMJudge.itm`, CLIP sim qua `CLIPProbe.similarity`) → `unload`. Model lỗi (OOM, gated, LoRA) thành `ModelRun.error`,
grid vẫn vẽ. `review.enabled=false` → `review.generate_only`: một vòng, điểm = CLIP fidelity, cùng hình dạng ReviewOutcome.

## Ba bất biến

1. **Reviewer chỉ nhìn `Perception`.** Với SDXL không có sự thật nội bộ. Với stub, `GenOutput.oracle`
   tồn tại nhưng không được truyền vào `critique`. Test `test_offline.py` ghim điều này.
2. **Phần cần nhất quán là luật, không phải LLM.** Gộp/lọc/xếp hạng, hoà giải, lập bản sửa nằm trong
   `rule_agent.py` và `shared.py`. Cùng bộ lỗi luôn cho cùng hành động, nên hai lần chạy so được với nhau.
   LLM chỉ ở chỗ cần hiểu ngôn ngữ hoặc ảnh: phân tích prompt, dịch thuộc tính, mô tả ảnh, phê bình, judge.
3. **Spec rỗng không phải đạt.** Prompt mà stage 3 không giữ thực thể nào bị đánh dấu `verifiable=False`
   và loại khỏi mọi trung bình.

## Checklist thay cho findings tự do (v1.1)

Với mỗi thực thể, VLM nhận ảnh và ba câu hỏi đóng (`perception.py::_checklist`):

1. `identity`: `target` | `absent` | `unsure` | `confusable:<tên>` (tên lấy từ confusable_with)
2. `attrs`: với từng `required_attrs`, `yes` | `no` | `unsure`
3. `forbidden`: với từng `forbidden_attrs`, `yes` | `no` | `unsure`

`shared.checklist_critique` biến đáp án thành findings và điểm: confusable → critical; absent → major;
`no` → thiếu (major nếu quá nửa, không thì minor); forbidden `yes` → critical; `unsure` tính nửa.
Điểm thực thể = danh_tính × (1 − 0,45 × tỉ lệ thiếu) − 0,5 × số vi phạm, có trọng số.

Lý do: VLM 3B viết findings tự do và tự chấm điểm kém (lần chạy đầu: điểm 0,6 cố định, lý giải lặp), nhưng
chọn đáp án cho sẵn thì ổn. Đường cũ giữ ở `PromptAgent._critique_freeform`, chỉ dùng khi không có checklist.

## Judge độc lập (v1.1)

`evaluation.ITMJudge`: BLIP-2 ITM (`Salesforce/blip2-itm-vit-g`) chấm ba nhóm câu: `clip_label` của thực thể
(identity), `"<name_en> with <attr_en>"` cho từng must_have (completeness), câu confusable (purity → 1 − max).
Không dùng LLM, khác họ model với reviewer Qwen. Không tải được thì lùi về `CLIPJudge`. `judge.backend: vlm`
để quay về judge bằng agent nếu muốn so.

## Khối tri giác và vì sao có hai tín hiệu

Sơ đồ draft đi thẳng từ Gen sang Review. Với bộ sinh thật, giữa hai khối đó phải có bước "nhìn":

* **VLM** trả lời câu hỏi mở "thấy gì" và liệt kê thuộc tính. Đọc được chi tiết (cổ áo, số dây đàn) nhưng
  dễ bị dẫn dắt và có prior lệch giống T2I.
* **CLIP** so ảnh với tên mục tiêu và tên từng confusable, trả xác suất. Không hiểu chi tiết nhưng không bị
  prompt dẫn, và rẻ.

`shared.adjudicate` hợp hai tín hiệu: điểm = (1 − w)·VLM + w·CLIP; nếu CLIP nghiêng về confusable hơn mục
tiêu quá `drift_margin` thì thêm một finding critical độc lập. Bất đồng giữa hai bên được ghi lại. Đây là
"debate" của v1.

## Công thức của bộ sinh stub

Chỉ dùng cho test offline. Mô phỏng cách T2I thật thất bại về văn hoá:

```
strength = prior + 0.55·conditioning + 0.40·[LoRA phủ] + 0.25·[có ảnh tham chiếu] + 0.10·[tên trong prompt]
         − 0.25·[confusable chính KHÔNG bị chặn trong negative] + nhiễu(seed)
≥ 0.55 đúng · 0.30–0.55 lệch sang confusable · < 0.30 không vẽ
```

`prior_strength` trong KB là ước lượng chủ quan. Lần chạy Kaggle đầu tiên cho bạn con số thật để thay vào.

## Ánh xạ bản sửa sang SDXL

| RevisionPlan | GenSpec | SDXL |
|---|---|---|
| add_negative | negative_terms += (dedupe) | `negative_prompt` |
| add_positive | prompt_terms += (dedupe; chỉ tiếng Anh) | prompt |
| boost[e] | conditioning[e] += ; thực thể ≥ 0.3 được đẩy lên đầu **một lần duy nhất** (emphasis); guidance += 0.5·max | thứ tự token, `guidance_scale` |
| fast_iters | vòng sửa dùng LCM-LoRA `fast_steps` bước, guidance `fast_guidance`; khi đạt → `to_final_render` đủ bước và kiểm lại | `LCMScheduler`, `set_adapters` |
| use_reference_image | ip_adapter_image = ảnh Commons đã kiểm | `ip_adapter_image`, `set_ip_adapter_scale` |
| attach_lora | lora = generator.lora_id | `set_adapters(["culture"], [scale])` |
| guidance_delta | guidance += | `guidance_scale` |

## Điểm cần cải thiện cho v2

* Prompt weighting thật (compel) thay cho lặp token.
* Judge bằng model khác reviewer.
* Hai reviewer persona thật (văn hoá / thị giác) khi có ngân sách gọi model.
* KB do chuyên gia kiểm định; `prior_strength` đo từ vòng 0 của SDXL.
* LoRA văn hoá Việt huấn luyện trên dữ liệu có giấy phép.
