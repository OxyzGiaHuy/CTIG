"""Bóc prompt thật ra khỏi phần feedback/điểm mà LLM nhỏ nhả kèm."""
import importlib.util as u
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sp = u.spec_from_file_location("ctp", ROOT / "scripts" / "culture_trip_prompts.py")
m = u.module_from_spec(sp)
sp.loader.exec_module(m)

REAL = ("Here is the refined prompt based on the feedback: ### Refined Prompt: Draw a picture of a young woman "
        "wearing a traditional Vietnamese outfit called \"ao dai Le Mur\". Her ao dai is a long, split tunic worn "
        "over silk trousers. She is standing at a school gate. ### Refine Feedback: The refined prompt provides "
        "more specific details. **Clarity (10/10)**: The prompt is clear. **Total Score (45/50)**: Overall good.")
out, cut = m.clean_refined(REAL)
assert cut
assert out.startswith("Draw a picture of a young woman"), out[:60]
assert out.endswith("school gate."), out[-40:]
assert "Feedback" not in out and "Clarity" not in out and "10/10" not in out, out
print("bóc được: %d từ (thô %d từ)" % (len(out.split()), len(REAL.split())))

# chỉ có lời rào, không có nhãn ###
out2, cut2 = m.clean_refined("Here is the refined prompt: A woman in a white ao dai at a school gate.")
assert cut2 and out2 == "A woman in a white ao dai at a school gate.", out2

# chuỗi đã sạch -> giữ nguyên, không cắt nhầm
clean = "A young woman wearing a white ao dai stands at an iron school gate in Hanoi."
out3, cut3 = m.clean_refined(clean)
assert out3 == clean and not cut3, (out3, cut3)

# chuỗi rỗng
assert m.clean_refined("") == ("", False)
assert m.clean_refined(None) == ("", False)
print("chuỗi đã sạch giữ nguyên; chuỗi rỗng không lỗi")

assert m.clip_tokens(" ".join(["w"] * 100)) > 77
assert m.clip_tokens("A woman in ao dai.") < 77
print("đếm token: câu dài vượt 77, câu ngắn không")
print("ĐẠT")
