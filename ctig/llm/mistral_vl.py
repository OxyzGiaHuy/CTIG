"""Backend Mistral-Small-3.1-24B-Instruct chạy local (transformers). Không cần API key.

Vì sao có file này: bộ chấm của vòng sửa hỏi ba loại câu, và Qwen2.5-VL-7B chỉ làm tốt hai.

| câu hỏi | số ảnh | Qwen2.5-VL-7B |
|---|---|---|
| chấm 8 tiểu mục khớp prompt và thẩm mỹ | 1 | được |
| ép chọn thực thể giữa các vật dễ nhầm (theo logits) | 1 | được, tách 0,05-0,24 so với 0,84-0,96 |
| "ảnh sinh khác ảnh thật ở chỗ nào" | 3 | **hỏng** - lặp lại mô tả của ảnh thật, không so sánh |

T2I-Copilot (arXiv 2507.20536) dùng GPT-4o-mini mặc định và **Mistral-Small-3.1-24B-Instruct-2503** cho
nhánh model mở. Đây là nhánh mở đó, để bộ chấm của ta giống họ.

Model 24B bf16 chiếm ~48 GB VRAM. Trên card 80 GB thì chạy cùng SDXL được (SDXL đỉnh 24,7 GB) nhưng
KHÔNG cùng FLUX (đỉnh 51,98 GB). Thiếu chỗ thì đặt `llm.quant: 4bit` (~14 GB, cần bitsandbytes).
"""

from __future__ import annotations

import time

from .base import JSONChatMixin

#: Ảnh to hơn mức này bị thu nhỏ trước khi vào processor. Pixtral cắt ảnh thành ô 16x16 nên
#: ảnh 1024px tốn ~4k token; ba ảnh một lượt sẽ vượt ngữ cảnh nếu không thu.
MAX_SIDE = 768


class MistralVLBackend(JSONChatMixin):
    name = "mistral_vl"

    def __init__(self, model: str, device: str = "cuda:0", dtype: str = "auto",
                 max_new_tokens: int = 1024, temperature: float = 0.2, json_retries: int = 2,
                 quant: str | None = None):
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
            dt = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        else:
            dt = getattr(torch, dtype)

        self.model = _load_model(model, dt, device, quant)
        # fix_mistral_regex: kho của Mistral khai sai mẫu regex tách token; không bật thì transformers cảnh báo
        # "This will lead to incorrect tokenization". Cờ này mới có ở transformers gần đây nên có đường lùi.
        try:
            self.processor = AutoProcessor.from_pretrained(model, fix_mistral_regex=True)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model)

    # --- phần dùng chung: dựng messages theo đúng khuôn chat template của Mistral3 ---
    def _messages(self, system: str, user: str, images: list[str] | None):
        from PIL import Image

        content: list[dict] = []
        for p in images or []:
            img = Image.open(p).convert("RGB")
            img.thumbnail((MAX_SIDE, MAX_SIDE))
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": user})
        return [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]

    def _encode(self, messages):
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        return inputs.to(self.model.device)

    def chat(self, system: str, user: str, images: list[str] | None = None,
             max_new_tokens: int | None = None) -> str:
        inputs = self._encode(self._messages(system, user, images))
        gen = dict(max_new_tokens=max_new_tokens or self.max_new_tokens)
        if self.temperature > 0:
            gen.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.time()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        self.calls += 1
        self.seconds += time.time() - t0
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def choice_prob(self, question: str, images: list[str] | None = None,
                    letters: tuple[str, ...] = ("A", "B", "C")) -> list[float]:
        """Xác suất chuẩn hoá trên chữ cái đầu câu trả lời, một lượt forward, không sinh.

        Giống `QwenVLBackend.choice_prob`: bắt model CHỌN giữa thuộc tính đúng và một mô tả sai cụ thể,
        vì câu có/không cho hai phân bố trùm lên nhau hoàn toàn (xem ghi chú ở bản Qwen).
        """
        inputs = self._encode(self._messages("Answer with a single letter.", question, images))
        tok = self.processor.tokenizer
        t0 = time.time()
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        self.calls += 1
        self.seconds += time.time() - t0
        cols = []
        for letter in letters:
            ids = set()
            for w in (letter, " " + letter, letter.lower()):
                enc = tok.encode(w, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
            cols.append(self.torch.logsumexp(logits[list(ids)], 0))
        return [float(x) for x in self.torch.softmax(self.torch.stack(cols), 0)]


def _load_model(model: str, dt, device: str, quant: str | None):
    from transformers import AutoModelForImageTextToText

    kwargs: dict = dict(dtype=dt)
    if quant in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dt, bnb_4bit_quant_type="nf4")
            if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        kwargs["device_map"] = device          # bitsandbytes cần device_map, không nhận .to() sau đó
        return AutoModelForImageTextToText.from_pretrained(model, **kwargs)
    kwargs["device_map"] = device
    return AutoModelForImageTextToText.from_pretrained(model, **kwargs)
