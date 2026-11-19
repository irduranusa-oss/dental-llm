import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from server.nachgpt_verified_audit_answers import verified_audit_answer
from server.nachgpt_intent_router import missing_required_categories, route_nachgpt_question


def test_ceo_report_audit_reports_actual_send_not_generation():
    facts = {"executive_reports": {
        "production": {"active_cases": 5},
        "delivery_audit": {"ok": True, "date": "2026-11-19", "sent_today": True,
            "confirmed_send_count": 1,
            "events": [{"report_type": "daily", "period_start": "2026-11-18",
                        "period_end": "2026-11-18",
                        "sent_at_local": "2026-11-19T06:20:14-07:00",
                        "smtp_accepted": True}]}}}
    out = verified_audit_answer("DIME SI EL REPORTE AL CEO SALIO CORRECTAMENTE HOY", facts, "es")
    assert "6:20" in out and "2026-11-18" in out
    assert "no comprueba apertura" in out
    assert verified_audit_answer("Dime si el reporte al CEO salió hoy",
                                 {"executive_reports": {"production": {}}}, "es").startswith("NO VERIFICADO")


def test_weekly_employee_report_is_exact_and_scoped():
    facts = {"employee_activity": {
        "ok": True, "period_start": "2026-11-16", "period_end": "2026-11-19",
        "action_count": 2, "source": "postgres",
        "actions": [
            {"employee_name": "Worker A", "timestamp": "2026-11-16", "action": "Design",
             "case_name": "TEST_CASE_A", "current_case_status": "DESIGN"},
            {"employee_name": "Worker B", "timestamp": "2026-11-19", "action": "QC",
             "case_name": "TEST_CASE_B", "current_case_status": "QC"},
        ]}}
    result = verified_audit_answer("dame las acciones de los trabajadores esta semana", facts)
    assert "2026-11-16 al 2026-11-19" in result
    assert "Worker A" in result and "Worker B" in result


def test_plural_case_overview_shows_name_status():
    facts = {"case_overview": {"ok": True, "period_start": "2026-11-16",
        "period_end": "2026-11-19", "created_count": 1, "truncated": False,
        "cases": [{"case_name": "TEST_CASE", "patient_name": "Synthetic",
                   "current_status": "FINALIZADO", "created_at": "2026-11-16T18:00:00"}]}}
    answer = verified_audit_answer("descríbeme los casos creados esta semana y dime cómo van", facts)
    assert "TEST_CASE" in answer and "FINALIZADO" in answer and "2026-11-16" in answer
    route = route_nachgpt_question("descríbeme los casos creados esta semana y dime cómo van")
    assert "case_overview" in route["required_categories"]
    assert missing_required_categories("casos creados esta semana", facts) == []


def test_employee_and_client_cannot_request_all_case_overviews():
    from server.nachgpt_contract import validate_fact_packet, NachGPTContractError
    for role in ("CLIENT", "EMPLOYEE"):
        p = {"role": role, "user_id": "TEST", "display_name": "Synthetic",
             "laboratory_id": "LAB001", "capabilities": ["case_overview", "case"]}
        packet = {"verified": True, "read_only": True, "laboratory_id": "LAB001",
                  "target_laboratory_id": "LAB001", "scope": "LAB",
                  "sources": ["synthetic"], "categories": {"case_overview": {"ok": True}}}
        with pytest.raises(NachGPTContractError):
            validate_fact_packet(p, packet)
