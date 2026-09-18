import json
from pathlib import Path

from ctig.agents.vietrepair import _cho_can_ta, _contract_text, load_contracts
from scripts.validate_contracts import validate


ROOT = Path(__file__).resolve().parent.parent
CONTRACTS = ROOT / "data" / "contracts_v2.json"
PROMPTS = ROOT / "data" / "prompts_simple.json"


def test_all_50_contracts_validate():
    errors, warnings, counts = validate(CONTRACTS, PROMPTS)
    assert errors == []
    assert warnings == []
    assert counts == {
        "contracts": 50,
        "required": 158,
        "confusables": 115,
        "required_scored": 150,
        "conditional": 6,
        "excluded": 2,
    }


def test_optional_fact_is_not_sent_to_critic_or_observer():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    s010 = contracts["S010"]
    text = _contract_text(s010)
    assert "underwater_control" not in text
    assert "control" not in _cho_can_ta(s010)


def test_conditional_fact_is_labelled_not_mandatory():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    text = _contract_text(contracts["S020"])
    assert "CONDITIONAL" in text
    assert "inner_head_ring" in text
    required_section = text.split("CONDITIONAL", 1)[0]
    assert "inner_head_ring" not in required_section


def test_v2_keeps_legacy_agent_keys():
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    for pid, contract in contracts.items():
        assert pid == contract["prompt_id"]
        assert contract["entity"]
        assert contract["required"]
        assert contract["confusables"]


def test_vietrepair_uses_v2_by_default():
    contracts = load_contracts()
    assert contracts["S001"]["contract_version"] == "2.0"
