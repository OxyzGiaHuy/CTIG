"""Chạy: python tests/test_offline.py  (không cần GPU, không cần mạng)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ctig.config import Config
from ctig.kb import attr_covered, contains, tokens
from ctig.llm.json_utils import extract_json
from ctig.pipeline import Pipeline, load_prompts
from ctig.schema import GenOutput

FAILED = []


def check(name, cond, detail=""):
    print(("  ok    " if cond else "  FAIL  ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        FAILED.append(name)


def test_matching():
    check("'hue' không khớp 'Khuê'", not contains("Gác Khuê Văn Các", "hue"))
    check("'pho' không khớp 'phố'", not contains("phố cổ Hà Nội", "pho"))
    check("'phở' khớp", contains("một tô phở bò", "phở"))
    req, forb = "nước dùng trong, không sánh đặc", "nước dùng đỏ sánh cay kiểu mì Tứ Xuyên"
    check("túi-từ thô dương tính giả", attr_covered(forb, [req]))
    check("loại token chung thì hết", not attr_covered(forb, [req], exclude=tokens(req)))


def test_json_extract():
    check("json trong fence", extract_json('bla ```json\n{"a": 1}\n``` bla')["a"] == 1)
    check("json có trailing comma", extract_json('{"a": [1,2,],}')["a"] == [1, 2])
    check("json giữa text", extract_json('Kết quả: {"x": {"y": "z"}} xong')["x"]["y"] == "z")


def test_pipeline_offline():
    cfg = Config.load("configs/offline.yaml", {"runs_dir": "runs/_test", "run_name": "offline-test"})
    pipe = Pipeline(cfg, log=lambda *a, **k: None)
    prompts = [p for p in load_prompts(cfg.prompts_path) if p.id in ("p001", "p009", "p048", "p050")]
    results, s = pipe.run_batch(prompts)
    by = {r.prompt.id: r for r in results}
    check("chạy hết 4 prompt", len(results) == 4)
    check("p048 (KB không phủ) bị đánh dấu không kiểm chứng được", not by["p048"].record.verifiable)
    check("p009 (đàn bầu, prior 0.03) được gắn LoRA",
          any(it.gen_spec.lora for it in by["p009"].outcome.iterations),
          str([it.gen_spec.lora for it in by["p009"].outcome.iterations]))
    check("p050 kết thúc đúng", by["p050"].record.oracle_fidelity == 1.0, str(by["p050"].record.oracle_fidelity))
    check("có ảnh cuối", all(Path(r.record.final_image_path).exists() for r in results))
    check("có report.html", (pipe.run_dir / "report.html").exists())
    check("có user_study.csv", (pipe.run_dir / "user_study.csv").exists())
    check("GenOutput không lộ oracle vào Perception",
          "oracle" not in by["p001"].outcome.iterations[0].perception.__dataclass_fields__)


if __name__ == "__main__":
    for fn in (test_matching, test_json_extract, test_pipeline_offline):
        print(f"\n{fn.__name__}")
        fn()
    print("\n" + ("THẤT BẠI: " + ", ".join(FAILED) if FAILED else "TẤT CẢ ĐỀU ĐẠT"))
    sys.exit(1 if FAILED else 0)
