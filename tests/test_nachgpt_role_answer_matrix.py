"""End-to-end synthetic role-question tests against the operational endpoint.

No user records, production mutations, or external provider calls. Assert both
answer content and tenant/role refusals, not merely successful HTTP status.
"""
from types import SimpleNamespace
import asyncio
import os

import pytest
from fastapi import HTTPException

from server.nachgpt_contract import validate_fact_packet, NachGPTContractError
from server.nachgpt_intent_router import missing_required_categories
from server.nachgpt_operational_fallback import build_operational_fallback


def principal(role, capabilities, lab="LAB001"):
    return {
        "role": role, "user_id": "SIM-" + role,
        "display_name": "Simulation", "laboratory_id": lab,
        "capabilities": list(capabilities),
    }


def packet(categories, lab="LAB001", *, own=False):
    return {
        "verified": True, "read_only": True,
        "laboratory_id": lab, "target_laboratory_id": lab,
        "scope": "OWN_CASE" if own else "LAB",
        "resource_owner_match": bool(own),
        "sources": ["synthetic_verified_fixture"],
        "categories": categories,
    }


@pytest.mark.parametrize("role,query,caps,facts,answer_fragment", [
    ("CLIENT", "En qué estado está mi caso", ["own_case"],
     {"own_case": {"case_name": "SIM-PATIENT", "status": "IN_PROGRESS"}},
     "IN_PROGRESS"),
    ("CLIENT", "Tracking de entrega de mi caso", ["own_case_tracking", "own_case"],
     {"own_case": {"case_name": "SIM-PATIENT", "status": "IN_PROGRESS"},
      "own_case_tracking": {"tracking": "SIM-TRACK-901"}},
     "SIM-TRACK-901"),
    ("EMPLOYEE", "Cuánto inventario de zirconia tenemos", ["inventory"],
     {"inventory": {"total_materiales": 135, "total_items_stock": 0.5}},
     "0.5"),
    ("SUPERVISOR", "Estado de maquinas fresadoras", ["machines", "production"],
     {"machines": {"available": 4}, "production": {"pending": 6}},
     "available: 4"),
    ("LAB_ADMIN", "Revisar compras pendientes", ["purchases", "finance"],
     {"purchases": {"purchase_count": 2, "pending": 2}},
     "purchase_count: 2"),
    ("CEO", "Cuántas compras hay pendientes", ["purchases", "finance"],
     {"purchases": {"purchase_count": 3, "pending": 1}},
     "pending: 1"),
    ("CEO", "Facturación de este laboratorio", ["finance"],
     {"finance": {"published_total": 990}},
     "published_total: 990"),
    ("SUPER_ADMIN", "Audita el estado de plataforma", ["platform_health", "global_labs"],
     {"platform_health": {"status": "OK"}},
     "status: OK"),
    ("SUPER_ADMIN", "Compras de LAB001", ["global_labs", "purchases", "finance"],
     {"purchases": {"purchase_count": 7}},
     "purchase_count: 7"),
])
def test_role_question_returns_grounded_allowed_answer(role, query, caps, facts, answer_fragment):
    p = principal(role, caps)
    doc = packet(facts, own=(role == "CLIENT"))
    if role == "CLIENT" and "own_case_tracking" in facts:
        assert missing_required_categories(query, facts, role=role) == []
    else:
        assert missing_required_categories(query, facts, role=role) == []
    assert validate_fact_packet(p, doc)["verified"] is True
    answer = build_operational_fallback(query, doc, "es")
    assert answer_fragment in answer
    assert "synthetic_verified_fixture" in answer


@pytest.mark.parametrize("role,query,caps,facts,denied", [
    ("CLIENT", "Dame inventario", ["own_case"], {"inventory": {"total_materiales": 135}}, "forbidden_fact_categories"),
    ("CLIENT", "Factura del laboratorio", ["own_case"], {"finance": {"published_total": 90}}, "forbidden_fact_categories"),
    ("EMPLOYEE", "Compras", ["purchases"], {"purchases": {"purchase_count": 3}}, "forbidden_fact_categories"),
    ("SUPERVISOR", "Compras", ["purchases"], {"purchases": {"purchase_count": 3}}, "forbidden_fact_categories"),
    ("LAB_ADMIN", "Compras", ["purchases"], {"purchases": {"purchase_count": 3}}, "forbidden_fact_categories"),
    ("CEO", "Datos de otro laboratorio", ["purchases", "finance"], {"purchases": {"purchase_count": 9}}, "tenant_mismatch"),
])
def test_denied_questions_do_not_expose_data(role, query, caps, facts, denied):
    p = principal(role, caps)
    doc = packet(facts, lab="LAB002" if role == "CEO" else "LAB001", own=role == "CLIENT")
    with pytest.raises(NachGPTContractError, match=denied):
        validate_fact_packet(p, doc)


def test_client_must_match_case_owner():
    p = principal("CLIENT", ["own_case"])
    doc = packet({"own_case": {"status": "IN_PROGRESS"}}, own=False)
    with pytest.raises(NachGPTContractError, match="client_scope_must_be_own_case"):
        validate_fact_packet(p, doc)


def test_missing_evidence_is_rejected_not_guessed():
    assert missing_required_categories("Revisar compras", {}, role="CEO") == ["purchases"]
    assert missing_required_categories("Mi caso", {}, role="CLIENT") == ["own_case"]
    assert missing_required_categories("Tracking de mi caso", {"own_case": {}}, role="CLIENT") == ["own_case_tracking"]
    assert missing_required_categories("Audita las compras", {"inventory": {}}, role="SUPER_ADMIN") == ["purchases"]


def test_operational_endpoint_role_simulation(monkeypatch):
    # Exercise the real authenticated request handler, with an isolated
    # deterministic provider instead of calling OpenAI or production.
    from server import main as svc

    monkeypatch.setenv("NACHGPT_GATEWAY_TOKEN", "simulated-secret")
    monkeypatch.setattr(svc, "generate_answer", lambda q, lang, **kw: build_operational_fallback(
        q, __import__("json").loads(kw["dentodo_context"].split("VERIFIED_NACHGPT_FACT_PACKET:", 1)[1].strip())["fact_packet"], lang
    ))
    principal_obj = principal("CEO", ["purchases", "finance"])
    evidence = packet({"purchases": {"purchase_count": 2}})
    body = svc.NachGPTChatIn(pregunta="Dime cuántas compras hay", idioma="es",
                            principal=principal_obj, fact_packet=evidence)
    request = SimpleNamespace(headers={"X-NACHGPT-Gateway-Token": "simulated-secret"})
    response = asyncio.run(svc.nachgpt_chat_endpoint(body, request))
    assert response["verified_fact_packet"] is True
    assert "purchase_count: 2" in response["respuesta"]

    request_bad = SimpleNamespace(headers={"X-NACHGPT-Gateway-Token": "wrong"})
    with pytest.raises(HTTPException) as err:
        asyncio.run(svc.nachgpt_chat_endpoint(body, request_bad))
    assert err.value.status_code == 403

    missing = svc.NachGPTChatIn(pregunta="Dime cuántas compras hay", idioma="es",
                               principal=principal_obj, fact_packet=packet({}))
    with pytest.raises(HTTPException) as err:
        asyncio.run(svc.nachgpt_chat_endpoint(missing, request))
    assert err.value.status_code == 409
