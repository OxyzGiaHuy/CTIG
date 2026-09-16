"""
Backend Qwen2.5-VL chạy local trên GPU (transformers). Không cần API key.

Một model đảm nhiệm cả agent (text) và tri giác (ảnh). Trên Kaggle T4 16GB:
Qwen2.5-VL-3B-Instruct fp16 ~7 GB. Nếu thiếu VRAM, đổi sang Qwen/Qwen2-VL-2B-Instruct.
"""

from __future__ import annotations

import time

from .base import JSONChatMixin


class QwenVLBackend(JSONChatMixin):
    name = "qwen_vl"

    def __init__(self, model: str, device: str = "cuda:0", dtype: str = "auto",
                 max_new_tokens: int = 1024, temperature: float = 0.2, json_retries: int = 2):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.json_retries = json_retries
        self.model_id = model
        self.calls = 0
        self.seconds = 0.0

        if dtype == "auto":
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dt = torch.bfloat16 if use_bf16 else torch.float16
        else:
            dt = getattr(torch, dtype)

        self.model = _load_model(model, dt, device)
        # Giới hạn số pixel để mỗi ảnh tốn ít token và VRAM.
        self.processor = AutoProcessor.from_pretrained(
            model, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28
        )

    def chat(self, system: str, user: str, images: list[str] | None = None, max_new_tokens: int | None = None) -> str:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        content = []
        for p in images or []:
            img = Image.open(p).convert("RGB")
            img.thumbnail((896, 896))
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        t0 = time.time()
        gen_kwargs = dict(max_new_tokens=max_new_tokens or self.max_new_tokens)
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            # temperature/top_p/top_k trong generation_config gây cảnh báo khi do_sample=False
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        self.calls += 1
        self.seconds += time.time() - t0
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]


    def ground(self, image: str, labels: list[str]) -> list[dict]:
        """Qwen2.5-VL grounding: trả [{"bbox": (x0,y0,x1,y1) PIXEL ẢNH GỐC, "label": str}].
        Qwen2.5-VL trả toạ độ theo ảnh ĐÃ RESIZE (khác Qwen2-VL chuẩn hoá 0-1000) -> phải quy đổi bằng image_grid_thw * 14,
        nếu không hộp lệch có hệ thống (thường lệch trục Y)."""
        import json as _json
        import re as _re

        from PIL import Image
        from qwen_vl_utils import process_vision_info

        orig = Image.open(image).convert("RGB")
        ow, oh = orig.size
        img = orig.copy()
        img.thumbnail((896, 896))
        cats = ", ".join(labels)
        user = (f'Locate every instance that belongs to the following categories: "{cats}". '
                'Report bounding boxes in JSON like [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]. '
                "If a category is not visible, omit it. Output JSON only.")
        messages = [{"role": "system", "content": "You are a precise visual grounding model."},
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=None, top_p=None, top_k=None)
        self.calls += 1
        self.seconds += time.time() - t0
        raw = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        m = _re.search(r"\[.*\]", raw, _re.S)
        if not m:
            return []
        try:
            rows = _json.loads(m.group(0))
        except ValueError:
            return []
        grid = inputs.get("image_grid_thw")
        in_h = int(grid[0][1]) * 14 if grid is not None else img.height
        in_w = int(grid[0][2]) * 14 if grid is not None else img.width
        res = []
        for r in rows if isinstance(rows, list) else []:
            b = r.get("bbox_2d") or r.get("bbox") if isinstance(r, dict) else None
            if not b or len(b) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in b]
            except (TypeError, ValueError):
                continue
            box = (max(0, int(x1 / in_w * ow)), max(0, int(y1 / in_h * oh)),
                   min(ow, int(x2 / in_w * ow)), min(oh, int(y2 / in_h * oh)))
            if box[2] > box[0] and box[3] > box[1]:
                res.append({"bbox": box, "label": str(r.get("label", ""))})
        return res

    def yes_prob(self, question: str, images: list[str] | None = None) -> float:
        """VQAScore (Lin et al. 2024): P('Yes') so với P('No') ở token đầu câu trả lời, một lượt forward, không sinh.
        Dùng cho câu hỏi có/không về thuộc tính (Reviewer) và cho câu chuẩn 'Does this figure show "<prompt>"?'."""
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        content = []
        for p in images or []:
            img = Image.open(p).convert("RGB")
            img.thumbnail((896, 896))
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages = [{"role": "system", "content": "Answer with a single word: Yes or No."},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.model.device)
        tok = self.processor.tokenizer
        yes_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes", " yes")}
        no_ids = {tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No", " no")}
        t0 = time.time()
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        self.calls += 1
        self.seconds += time.time() - t0
        ly = self.torch.logsumexp(logits[list(yes_ids)], 0)
        ln = self.torch.logsumexp(logits[list(no_ids)], 0)
        return float(self.torch.softmax(self.torch.stack([ly, ln]), 0)[0])

    def choice_prob(self, question: str, images: list[str] | None = None,
                    letters: tuple[str, ...] = ("A", "B", "C")) -> list[float]:
        """Câu hỏi TRẮC NGHIỆM: xác suất chuẩn hoá trên chữ cái đầu câu trả lời, một lượt forward, không sinh.

        v1.9.2. Câu có/không không phân biệt được cấu trúc trang phục: trên S001, "áo có tà xẻ hai bên không" cho
        0,87 với ảnh áo liền quần và 0,90-0,96 với ảnh áo dài thật -> trùm lên nhau hoàn toàn. Bắt model CHỌN giữa
        thuộc tính đúng và một mô tả sai cụ thể thì tách hẳn: 0,05-0,24 so với 0,84-0,96.
        """
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        content = []
        for p in images or []:
            img = Image.open(p).convert("RGB")
            img.thumbnail((896, 896))
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages = [{"role": "system", "content": "Answer with a single letter."},
                    {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt").to(self.model.device)
        tok = self.processor.tokenizer
        t0 = time.time()
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        self.calls += 1
        self.seconds += time.time() - t0
        cols = []
        for letter in letters:
            ids = {tok.encode(w, add_special_tokens=False)[0] for w in (letter, " " + letter, letter.lower())}
            cols.append(self.torch.logsumexp(logits[list(ids)], 0))
        return [float(x) for x in self.torch.softmax(self.torch.stack(cols), 0)]



def _load_model(model: str, dt, device: str):
    from transformers import AutoModelForImageTextToText

    kwargs = dict(torch_dtype=dt, device_map=device)
    try:
        return AutoModelForImageTextToText.from_pretrained(model, attn_implementation="sdpa", **kwargs)
    except (ValueError, TypeError):
        return AutoModelForImageTextToText.from_pretrained(model, **kwargs)
