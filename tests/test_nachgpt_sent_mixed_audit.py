from server.nachgpt_verified_audit_answers import verified_audit_answer
from server.nachgpt_intent_router import route_nachgpt_question, missing_required_categories

def test_sent_cases_invoice_tracking_answer_explicitly_reports_audit_status():
    facts = {"sent_cases_overview": {"ok": True, "period_start": "2026-11-16",
        "period_end": "2026-11-19", "sent_count": 1,
        "cases": [{"case_name": "TEST_CASE", "sent_date": "2026-11-19",
                   "tracking": "", "current_status": "Enviado",
                   "invoice_audit": {"status": "REVIEW_REQUIRED",
                                     "invoice_ids": ["INV-TEST"]}}]}}
    q = "buscame todos los casos enviados esta semana y si se elaboraron bien los invoices ademas si se agregaron los tracking numbers"
    assert route_nachgpt_question(q)["required_categories"] == ["sent_cases_overview"]
    assert missing_required_categories(q, facts, role="SUPER_ADMIN") == []
    result = verified_audit_answer(
        "buscame todos los casos enviados esta semana y si se elaboraron bien los invoices ademas si se agregaron los tracking numbers",
        facts)
    assert "TEST_CASE" in result and "INV-TEST" in result
    assert "REVIEW_REQUIRED" in result
    assert "rastreo: NO VERIFICADO" in result
