from ctig.contracts.pipeline import ContractExtractionPipeline
from ctig.contracts.wikipedia import BilingualWikipediaRetriever, WikipediaPassage


class FakeRetriever:
    def retrieve(self, entity):
        return [
            WikipediaPassage(
                "VI0", "vi", "Áo dài", "https://vi.wikipedia.org/wiki/%C3%81o_d%C3%A0i",
                "Áo dài truyền thống thường có cổ đứng cao và hai tà áo dài. "
                "Áo được mặc cùng với quần dài. Qipao là một váy liền thân.", "áo dài",
            ),
            WikipediaPassage(
                "EN0", "en", "Ao dai", "https://en.wikipedia.org/wiki/%C3%81o_d%C3%A0i",
                "The ao dai is worn over trousers. It has long panels and side slits.", "ao dai",
            ),
        ]


class QueueBackend:
    name = "fake"
    model_id = "fake/model"

    def __init__(self, outputs):
        self.outputs = list(outputs)

    def complete_json(self, system, user, schema, **kwargs):
        assert self.outputs
        return self.outputs.pop(0)


def entity_output():
    return {
        "name_vi": "áo dài", "name_en": "Vietnamese ao dai", "entity_type": "object",
        "prompt_specific_vi": ["màu trắng", "cổng trường"],
        "prompt_specific_en": ["white", "school gate"],
        "search_queries_vi": ["áo dài"], "search_queries_en": ["Vietnamese ao dai"],
    }


def extraction_output():
    base = {
        "part_vi": "cổ áo", "part_en": "collar", "visibility": "must_be_visible", "importance": 3,
    }
    return {
        "required": [
            {"id": "high_standing_collar", "label_vi": "cổ đứng cao", "label_en": "high standing collar",
             "description_vi": "áo có cổ đứng cao", "description_en": "the tunic has a high standing collar",
             "visual_evidence_vi": "cổ áo dựng sát quanh cổ", "visual_evidence_en": "a raised collar encircles the neck",
             "requirement_type": "canonical_cue",
             "citations": [{"source_id": "VI0", "quote": "Áo dài truyền thống thường có cổ đứng cao"}], **base},
            {"id": "white_fabric", "label_vi": "áo màu trắng", "label_en": "white fabric",
             "description_vi": "áo dài màu trắng", "description_en": "the ao dai is white",
             "visual_evidence_vi": "thân áo có màu trắng", "visual_evidence_en": "the tunic fabric is white",
             "part_vi": "màu vải", "part_en": "fabric color", "requirement_type": "prompt_specific",
             "visibility": "must_be_visible", "importance": 3,
             "citations": [{"source_id": "PROMPT", "quote": "áo dài trắng"}]},
            {"id": "historic_origin", "label_vi": "nguồn gốc", "label_en": "historic origin",
             "description_vi": "nguồn gốc lịch sử", "description_en": "historic origin",
             "visual_evidence_vi": "nguồn gốc lịch sử của trang phục", "visual_evidence_en": "the historical origin of the dress",
             "requirement_type": "identity",
             "citations": [{"source_id": "VI0", "quote": "Áo dài truyền thống thường có cổ đứng cao"}], **base},
            {"id": "english_only", "label_vi": "tà áo", "label_en": "long panels",
             "description_vi": "hai tà dài", "description_en": "two long panels",
             "visual_evidence_vi": "hai tà dài", "visual_evidence_en": "two long panels",
             "part_vi": "tà áo", "part_en": "panels", "requirement_type": "identity",
             "visibility": "must_be_visible", "importance": 3,
             "citations": [{"source_id": "EN0", "quote": "It has long panels and side slits"}]},
        ],
        "confusables": [
            {"id": "qipao", "name_vi": "sườn xám", "name_en": "qipao", "culture": "Chinese",
             "difference_vi": "váy liền thân", "difference_en": "one-piece dress",
             "citations": [{"source_id": "VI0", "quote": "Qipao là một váy liền thân"}]}
        ],
    }


def observability_output():
    return {"items": [
        {"attribute_id": "high_standing_collar", "verdict": "visible", "rationale_vi": "nhìn thấy cổ áo",
         "rationale_en": "the collar is visible"},
        {"attribute_id": "white_fabric", "verdict": "visible", "rationale_vi": "nhìn thấy màu",
         "rationale_en": "color is visible"},
        {"attribute_id": "historic_origin", "verdict": "visible", "rationale_vi": "sai có chủ ý",
         "rationale_en": "intentionally wrong"},
        {"attribute_id": "english_only", "verdict": "visible", "rationale_vi": "nhìn thấy tà",
         "rationale_en": "panels are visible"},
    ]}


def test_bilingual_grounding_and_observability_guards():
    backend = QueueBackend([entity_output(), extraction_output(), observability_output()])
    pipe = ContractExtractionPipeline(backend, FakeRetriever(), require_vi_evidence=True)
    result = pipe.run("S001", "Một cô gái mặc áo dài trắng đứng trước cổng trường.",
                      "A young woman in a white ao dai standing at a school gate.")
    assert [x["id"] for x in result["required"]] == ["high_standing_collar", "white_fabric"]
    assert result["required"][0]["description_vi"] == "áo có cổ đứng cao"
    assert result["required"][0]["citations"][0]["source_id"] == "VI0"
    reasons = {x["id"]: x["reason"] for x in result["dropped"]}
    assert reasons["historic_origin"] == "not_observable_in_single_image"
    assert reasons["english_only"] == "no_vietnamese_source_evidence"
    assert result["review"]["mode"] == "human_required"
    assert result["review"]["auto_merged"] is False
    assert result["status"] == "pending_human_review"


def test_phogpt_crosscheck_never_auto_merges():
    extractor = QueueBackend([entity_output(), extraction_output(), observability_output()])
    checker = QueueBackend([{
        "overall": "accept",
        "items": [
            {"attribute_id": "high_standing_collar", "verdict": "accept",
             "comment_vi": "có nguồn", "comment_en": "supported"}
        ],
        "missing_visual_cues_vi": [], "missing_visual_cues_en": [],
    }])
    pipe = ContractExtractionPipeline(extractor, FakeRetriever(), verifier=checker, require_vi_evidence=True)
    result = pipe.run("S001", "Một cô gái mặc áo dài trắng đứng trước cổng trường.",
                      "A young woman in a white ao dai standing at a school gate.")
    assert result["review"]["mode"] == "model_cross_check"
    assert result["review"]["auto_merged"] is False
    assert result["status"] == "model_checked_pending_human_approval"


def test_wikipedia_retriever_caches_utf8_vietnamese_response():
    import tempfile
    from pathlib import Path

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"query": {"pages": [{
                "index": 1, "title": "Áo dài",
                "fullurl": "https://vi.wikipedia.org/wiki/%C3%81o_d%C3%A0i",
                "extract": "Áo dài là trang phục Việt Nam. " * 10,
            }]}}

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()

    session = Session()
    with tempfile.TemporaryDirectory() as tmp:
        retriever = BilingualWikipediaRetriever(Path(tmp), session=session)
        first = retriever.search("vi", "áo dài", limit=1)
        second = retriever.search("vi", "áo dài", limit=1)
    assert first == second
    assert first[0].lang == "vi" and first[0].title == "Áo dài"
    assert session.calls == 1
