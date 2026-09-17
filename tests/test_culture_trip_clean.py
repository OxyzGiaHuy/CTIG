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
# bốn khuôn đã gặp trên dữ liệu thật của llama3:8b
CASES = {
 "ngoặc kép + SCORE": ('Here is the revised REFINED PROMPT that incorporates the suggested changes: "In a traditional '
   'Vietnamese setting, banh chung is a square dish wrapped in dong leaves and tied with bamboo strips. The leaves are '
   'green." SCORE: {\'Clarity\': 9.5, \'Total_score\': 42}', "In a traditional", "green."),
 "aims to 1.2.3": ('Based on the feedback, I refined the prompt as follows: The One Pillar Pagoda, a Buddhist temple '
   'near a pond, is famous for its structure. Early morning mist surrounds it in Hanoi. The refined prompt aims to: '
   '1. Maintain the scene. 2. Provide background.', "The One Pillar", "Hanoi."),
 "lời rào chồng nhau": ('Here is the refined prompt based on the feedback: ### Refined Prompt: Draw a picture of a '
   'young woman wearing an ao dai over silk trousers. ### Refine Feedback: more detail.', "Draw a picture", "trousers."),
}
for name, (raw, head, tail) in CASES.items():
    out, cut = m.clean_refined(raw)
    assert cut and out.startswith(head), (name, out[:60])
    assert out.endswith(tail), (name, out[-40:])
    for junk in ("SCORE", "Clarity", "aims to", "Refined Prompt", "Here is", "Based on"):
        assert junk not in out, (name, junk, out)
    print("%-22s %3d -> %2d từ, sạch" % (name, len(raw.split()), len(out.split())))

# an toàn: dò sai làm mất gần hết chữ -> trả lại chuỗi ban đầu
short = "Here is the refined prompt: ok."
assert m.clean_refined(short) == (short, False)
print("dò sai còn dưới 8 từ: trả lại chuỗi ban đầu")
print("ĐẠT")
