import sys, os, time
sys.path.insert(0, "/workspace/ctig17")
from PIL import Image
from ctig.llm.qwen_vl import QwenVLBackend

B = QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0")
T = B.torch
tok = B.processor.tokenizer


def choice_prob(question, images):
    """P over first answer token for A / B / C, one forward pass."""
    from qwen_vl_utils import process_vision_info
    content = []
    for p in images:
        im = Image.open(p).convert("RGB")
        im.thumbnail((896, 896))
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": question})
    msgs = [{"role": "system", "content": "Answer with a single letter."},
            {"role": "user", "content": content}]
    text = B.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ii, vi = process_vision_info(msgs)
    inp = B.processor(text=[text], images=ii, videos=vi, padding=True, return_tensors="pt").to(B.model.device)
    with T.inference_mode():
        lg = B.model(**inp).logits[0, -1].float()
    out = []
    for letter in ("A", "B", "C"):
        ids = {tok.encode(w, add_special_tokens=False)[0] for w in (letter, " " + letter, letter.lower())}
        out.append(T.logsumexp(lg[list(ids)], 0))
    return [float(x) for x in T.softmax(T.stack(out), 0)]


def ask(have, alt, name, img, flip):
    a, b = (alt, have) if flip else (have, alt)
    q = ("Look at the %s in this photo. Which ONE of these matches what you actually see?\n"
         "A. %s\nB. %s\nC. neither A nor B\n"
         "Answer with only the letter A, B or C." % (name, a[0].upper() + a[1:], b[0].upper() + b[1:]))
    p = choice_prob(q, [img])
    have_p = p[1] if flip else p[0]
    return have_p, p[2]


R = "/workspace/refs_new/reference_images_simple/selected/S001"
RUN = "/workspace/runs/v191/S001"
IMGS = [("JUMPSUIT sdxl(No)", RUN + "/sdxl_base/S001_sdxl_base_c1_hr.png")] + \
       [("REF " + f + "(Yes)", R + "/" + f) for f in sorted(os.listdir(R))]

NAME = "ao dai"
PAIRS = [
    ("long-sleeved tunic split at the hips into front and back panels",
     "a one-piece dress or jumpsuit joined from top to bottom, with no separate tunic panels"),
    ("high stand-up mandarin collar",
     "a round, V-shaped or open collar lying flat on the chest"),
    ("worn over wide-legged long trousers",
     "bare legs, a skirt, or a hem that ends above the ankles with no trousers"),
    ("fitted bodice with flowing loose panels",
     "a loose boxy top with no waist shaping"),
]

print("=== ep chon A/B/C, kiem thien lech vi tri (dung | dao) ===", flush=True)
print("%-46s %s" % ("thuoc tinh", " ".join("%-19s" % n[:19] for n, _ in IMGS)), flush=True)
t0 = time.time()
for have, alt in PAIRS:
    cells = []
    for _, p in IMGS:
        h1, _ = ask(have, alt, NAME, p, False)
        h2, _ = ask(have, alt, NAME, p, True)
        cells.append("%.2f|%.2f" % (h1, h2))
    print("%-46s %s" % (have[:46], " ".join("%-19s" % c for c in cells)), flush=True)
print("(%.0fs, %d luot forward)" % (time.time() - t0, B.calls), flush=True)

print("\n=== so voi Yes/No hien tai ===", flush=True)
for have, _ in PAIRS:
    q = ('Look carefully at the %s in this photo. Statement: "the %s has %s". '
         'Is this statement true for what you see? Answer Yes or No.' % (NAME, NAME, have))
    print("%-46s %s" % (have[:46], " ".join("%-19.2f" % B.yes_prob(q, [p]) for _, p in IMGS)), flush=True)
