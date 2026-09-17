import sys, os
sys.path.insert(0, "/workspace/ctig17")
from PIL import Image
from ctig.llm.qwen_vl import QwenVLBackend

B = QwenVLBackend("Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0")
R = "/workspace/refs_new/reference_images_simple/selected/S001"
NEG = "/workspace/runs/v191/S001/sdxl_base/S001_sdxl_base_c1_hr.png"
IMGS = [("JUMPSUIT(phai No)", NEG)] + [("REF " + f + "(phai Yes)", R + "/" + f) for f in sorted(os.listdir(R))]

NAME = "ao dai"
ATTR = "long-sleeved tunic split at the hips into front and back panels"

Q = {
    "Q1 hien tai (statement dai)":
        'Look carefully at the ' + NAME + ' in this photo. Statement: "the ' + NAME + ' has ' + ATTR +
        '". Is this statement true for what you see? Answer Yes or No.',
    "Q2 nguyen tu (khe ho hong)":
        "Look at the long garment. Does the top part have an open slit on each side starting at the hip, so that a "
        "separate front flap and a separate back flap of fabric hang loose below the waist? Answer Yes or No.",
    "Q3 nguyen tu (hai manh roi)":
        "Below the waist, can you see two separate loose flaps of fabric (one in front, one behind) whose side edges "
        "are open and not sewn together? Answer Yes or No.",
    "Q4 phan de (lien mot khoi)":
        "Is the garment a single continuous one-piece outfit from the neck to the ankles, with no separate over-tunic "
        "and no open side slits? Answer Yes or No.",
    "Q5 nguyen tu (ao rieng quan rieng)":
        "Is the person wearing TWO separate garments: a long tunic on top and trousers underneath, where you can see "
        "the tunic hem ending and the trousers continuing below it? Answer Yes or No.",
}

CROP = {
    "Q2c khe ho hong (anh cat)": Q["Q2 nguyen tu (khe ho hong)"],
    "Q3c hai manh roi (anh cat)": Q["Q3 nguyen tu (hai manh roi)"],
    "Q4c lien mot khoi (anh cat)": Q["Q4 phan de (lien mot khoi)"],
}


def lower(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    out = "/tmp/lower_" + os.path.basename(p) + ".png"
    im.crop((0, int(h * 0.35), w, h)).save(out)
    return out


print("%-30s" % "cau hoi", " ".join("%-20s" % n[:20] for n, _ in IMGS), flush=True)
for name, q in Q.items():
    vals = [B.yes_prob(q, [p]) for _, p in IMGS]
    print("%-30s" % name[:30], " ".join("%-20.2f" % v for v in vals), flush=True)
for name, q in CROP.items():
    vals = [B.yes_prob(q, [lower(p)]) for _, p in IMGS]
    print("%-30s" % name[:30], " ".join("%-20.2f" % v for v in vals), flush=True)

print("\n=== ep chon A/B (sinh chu) ===", flush=True)
mc = ("Look at what the person is wearing. Which description is correct?\n"
      "A. A long tunic with open side slits, its front and back panels hanging loose over SEPARATE trousers.\n"
      "B. A one-piece outfit joined from top to bottom, with no separate tunic panels.\n"
      "Answer with only the letter A or B.")
for n, p in IMGS:
    print("%-22s -> %s" % (n[:22], B.chat("Answer with only one letter.", mc, [p], max_new_tokens=4).strip()[:12]), flush=True)

print("\n=== mo ta tu do nua duoi ===", flush=True)
for n, p in IMGS:
    d = B.chat("You are a careful fashion observer.",
               "In ONE short sentence, describe only the lower half of the outfit: is it one continuous piece, or a "
               "tunic with loose panels over separate trousers?", [lower(p)], max_new_tokens=60)
    print("%-22s -> %s" % (n[:22], d.strip().replace("\n", " ")[:160]), flush=True)
