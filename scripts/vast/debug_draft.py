import json, sys
sys.path.insert(0, "/workspace/ctig17")
from pathlib import Path
from ctig.config import Config
from ctig.pipeline import load_prompts
from ctig.session import Session
from ctig.stages.extraction import quote_in_texts
cfg = Config.load("configs/vast_a100.yaml", {"runs_dir": "/workspace/runs"})
cfg.llm.cache = False
pid = sys.argv[1] if len(sys.argv) > 1 else "S001"; eid = sys.argv[2] if len(sys.argv) > 2 else "ao_dai"
p = next(x for x in load_prompts("data/prompts_simple.json") if x.id == pid)
s = Session(cfg, p, Path("/workspace/runs/v18_smoke"), log=print)
sr, src = s.retrieve(); print("retrieve từ", src)
texts = [it for it in sr.items if it.entity_id == eid and it.kind in ("wiki_text", "web_text") and it.snippet and len(it.snippet) > 80
         and not it.provenance.startswith("kb") and it.provenance not in ("extracted", "kb_auto")]
texts = sorted(texts, key=lambda t: -len(t.snippet))[:6]
print("nguồn:", [(t.title[:40], len(t.snippet)) for t in texts])
ent = s.kb.get(eid)
agent = s.agent
# gọi thẳng LLM với prompt của draft_kb_entry để xem JSON thô
import ctig.llm.prompt_agent as pa
llm = agent.llm
orig = llm.complete_json
def spy(system, user, schema, images=None, max_new_tokens=None):
    d = orig(system, user, schema, images=images, max_new_tokens=max_new_tokens)
    print("=== JSON THÔ ===\n" + json.dumps(d, ensure_ascii=False, indent=1)[:3000])
    raw = [t.snippet for t in texts]
    for k in ("must_have", "must_not"):
        for it in d.get(k, []) or []:
            if isinstance(it, dict):
                print(f"  {k}: {str(it.get('attr_en'))[:50]:50} quote ok={quote_in_texts(str(it.get('quote') or ''), raw)} | {str(it.get('quote'))[:80]}")
    return d
llm.complete_json = spy
out = agent.draft_kb_entry(ent, [{"title": t.title, "url": t.url, "text": t.snippet} for t in texts])
print("=== SAU LỌC ===", json.dumps({k: out[k] for k in ("must_have_en", "must_not_en", "tags_en", "neg_tags_en", "kind", "prior_strength", "dropped_unsourced")}, ensure_ascii=False))
