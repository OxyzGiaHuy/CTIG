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
  │  │ tri giác perceiver.perceive        ->  Perception{elements, caption, clip_probs}
  │  │ stage 5  agent.critique(Perception) ->  Critique (VLM)                        │
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

## Ba bất biến

1. **Reviewer chỉ nhìn `Perception`.** Với SDXL không có sự thật nội bộ. Với stub, `GenOutput.oracle`
   tồn tại nhưng không được truyền vào `critique`. Test `test_offline.py` ghim điều này.
2. **Phần cần nhất quán là luật, không phải LLM.** Gộp/lọc/xếp hạng, hoà giải, lập bản sửa nằm trong
   `rule_agent.py` và `shared.py`. Cùng bộ lỗi luôn cho cùng hành động, nên hai lần chạy so được với nhau.
   LLM chỉ ở chỗ cần hiểu ngôn ngữ hoặc ảnh: phân tích prompt, dịch thuộc tính, mô tả ảnh, phê bình, judge.
3. **Spec rỗng không phải đạt.** Prompt mà stage 3 không giữ thực thể nào bị đánh dấu `verifiable=False`
   và loại khỏi mọi trung bình.

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
| add_negative | negative_prompt += | `negative_prompt` |
| add_positive | prompt += | prompt |
| boost[e] | conditioning[e] += ; thực thể ≥ 0.3 lên đầu prompt và lặp; guidance += 0.5·max | thứ tự token, `guidance_scale` |
| use_reference_image | ip_adapter_image = ảnh Commons đã kiểm | `ip_adapter_image`, `set_ip_adapter_scale` |
| attach_lora | lora = generator.lora_id | `set_adapters(["culture"], [scale])` |
| guidance_delta | guidance += | `guidance_scale` |

## Điểm cần cải thiện cho v2

* Prompt weighting thật (compel) thay cho lặp token.
* Judge bằng model khác reviewer.
* Hai reviewer persona thật (văn hoá / thị giác) khi có ngân sách gọi model.
* KB do chuyên gia kiểm định; `prior_strength` đo từ vòng 0 của SDXL.
* LoRA văn hoá Việt huấn luyện trên dữ liệu có giấy phép.
