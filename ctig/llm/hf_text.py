"""Generic Hugging Face text-only backend for Qwen3 and PhoGPT."""

from __future__ import annotations

import time

from .base import JSONChatMixin


class HFTextBackend(JSONChatMixin):
    name = "hf_text"

    def __init__(self, model: str, device: str = "cuda:0", dtype: str = "auto",
                 max_new_tokens: int = 1400, temperature: float = 0.0,
                 json_retries: int = 2, trust_remote_code: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.json_retries = json_retries
        self.calls = 0
        self.seconds = 0.0
        if dtype == "auto":
            if not torch.cuda.is_available() or str(device).startswith("cpu"):
                dt = torch.float32
            else:
                dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dt = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dt, device_map=device, trust_remote_code=trust_remote_code
        )

    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # PhoGPT checkpoints may not expose a transformers chat template.
        return f"### Hướng dẫn:\n{system}\n\n### Câu hỏi:\n{user}\n\n### Trả lời:"

    def chat(self, system: str, user: str, images=None, max_new_tokens: int | None = None) -> str:
        if images:
            raise ValueError("HFTextBackend is text-only")
        prompt = self._prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = {"max_new_tokens": max_new_tokens or self.max_new_tokens}
        if self.temperature > 0:
            kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.9)
        else:
            kwargs.update(do_sample=False)
        t0 = time.time()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        self.calls += 1
        self.seconds += time.time() - t0
        generated = output[:, inputs.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
