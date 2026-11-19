from server.nachgpt_verified_audit_answers import verified_audit_answer
from server.nachgpt_intent_router import route_nachgpt_question, missing_required_categories
from server.nachgpt_contract import validate_fact_packet, NachGPTContractError
import pytest


def test_sent_week_answer_is_grounded_and_shows_missing_tracking():
    fact = {"sent_cases_overview": {
        "ok": True, "period_start": "2026-11-16", "period_end": "2026-11-19",
        "sent_count": 1, "cases": [{
            "case_name": "Eben_Conner_002c8d", "sent_date": "2026-11-19",
            "tracking": "", "current_status": "Enviado",
        }],
    }}
    question = "buscame todos los casos enviados esta semana"
    routed = route_nachgpt_question(question)
    assert routed["required_categories"] == ["sent_cases_overview"]
    assert missing_required_categories(question, fact, role="SUPER_ADMIN") == []
    answer = verified_audit_answer(question, fact)
    assert "Eben_Conner_002c8d" in answer
    assert "NO VERIFICADO" in answer
    assert "2026-11-19" in answer


@pytest.mark.parametrize("role", ["CLIENT", "EMPLOYEE"])
def test_nonprivileged_role_cannot_enumerate_sent_cases(role):
    principal = {"role": role, "user_id": "TEST", "display_name": "Test",
                 "laboratory_id": "LAB001", "capabilities": ["sent_cases_overview"]}
    facts = {"verified": True, "read_only": True,
             "laboratory_id": "LAB001", "target_laboratory_id": "LAB001",
             "scope": "LAB", "sources": ["synthetic"],
             "categories": {"sent_cases_overview": {"ok": True, "sent_count": 2}}}
    with pytest.raises(NachGPTContractError):
        validate_fact_packet(principal, facts)
