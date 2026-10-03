from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.nachgpt_intent_router import missing_required_categories, route_nachgpt_question
from server.nachgpt_contract import (
    NachGPTContractError,
    ROLE_CEO,
    ROLE_CLIENT,
    ROLE_EMPLOYEE,
    ROLE_SUPER_ADMIN,
    ROLE_SUPERVISOR,
    build_nachgpt_fact_context,
    build_role_aware_greeting,
    validate_fact_packet,
)


def principal(role, lab="LAB001", caps=None, name="Ignacio Ramirez"):
    return {
        "user_id": "U1",
        "display_name": name,
        "role": role,
        "laboratory_id": lab,
        "capabilities": caps or [],
    }


class NachGPTContractTests(unittest.TestCase):
    def test_client_greeting_and_own_case_only(self):
        p = principal(ROLE_CLIENT, caps=["own_case", "own_case_files"], name="Carlos Ortiz")
        packet = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB001",
            "scope": "OWN_CASE",
            "resource_owner_match": True,
            "categories": {
                "own_case": {"case_name": "TEST"},
                "own_case_files": {"count": 3},
            },
            "sources": ["canonical_case", "r2_head"],
        }
        ctx = build_nachgpt_fact_context(p, packet)
        self.assertIn('"scope": "OWN_CASE"', ctx)
        self.assertTrue(build_role_aware_greeting(p, "es").startswith("Hola, Carlos. A la orden."))

    def test_doctor_honorific_keeps_first_name(self):
        p = principal(ROLE_CEO, caps=["executive_reports"], name="Dr. Rajan Sheth")
        greeting = build_role_aware_greeting(p, "en")
        self.assertTrue(greeting.startswith("Hello, Dr. Rajan. At your service."))

    def test_client_cannot_receive_other_client_or_lab_facts(self):
        p = principal(ROLE_CLIENT, caps=["own_case"])
        bad_scope = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB001",
            "scope": "LAB_ALL_CASES",
            "resource_owner_match": True,
            "categories": {"own_case": {}},
            "sources": [],
        }
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p, bad_scope)

        bad_tenant = dict(bad_scope, scope="OWN_CASE", laboratory_id="LAB002")
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p, bad_tenant)

    def test_employee_cannot_receive_finance(self):
        p = principal(ROLE_EMPLOYEE, caps=["case", "production"])
        packet = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB001",
            "scope": "ROLE_SCOPE",
            "categories": {"finance": {"total": 100}},
            "sources": ["postgres"],
        }
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p, packet)

    def test_supervisor_stays_inside_own_lab(self):
        p = principal(ROLE_SUPERVISOR, caps=["case", "processor", "r2_evidence"])
        packet = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB002",
            "scope": "LAB",
            "categories": {"processor": {}},
            "sources": ["processor_run"],
        }
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p, packet)

    def test_ceo_can_receive_finance_only_for_own_lab(self):
        p = principal(ROLE_CEO, caps=["finance", "executive_reports"])
        ok = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB001",
            "scope": "LAB",
            "categories": {
                "finance": {"published_total": 500},
                "executive_reports": {"cases": 4},
            },
            "sources": ["postgres", "ceo_report"],
        }
        self.assertTrue(validate_fact_packet(p, ok)["verified"])

        cross = dict(ok, laboratory_id="LAB002")
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p, cross)

    def test_superadmin_global_requires_explicit_capability(self):
        p = principal(ROLE_SUPER_ADMIN, lab="", caps=["global_labs", "platform_health"])
        packet = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB002",
            "target_laboratory_id": "LAB002",
            "scope": "PLATFORM",
            "categories": {"global_labs": {"lab": "LAB002"}},
            "sources": ["platform_catalog"],
        }
        self.assertTrue(validate_fact_packet(p, packet)["verified"])

        p_no_global = principal(ROLE_SUPER_ADMIN, lab="", caps=["platform_health"])
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p_no_global, packet)

        tenant_finance = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB002",
            "target_laboratory_id": "LAB002",
            "scope": "LAB",
            "categories": {"finance": {"published_total": 25}},
            "sources": ["postgres"],
        }
        p_finance_without_global = principal(ROLE_SUPER_ADMIN, lab="", caps=["finance"])
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p_finance_without_global, tenant_finance)

    def test_missing_capabilities_fails_closed(self):
        p = principal(ROLE_EMPLOYEE, caps=[])
        packet = {
            "verified": True,
            "read_only": True,
            "laboratory_id": "LAB001",
            "scope": "ROLE_SCOPE",
            "categories": {"case": {}},
            "sources": ["canonical_case"],
        }
        with self.assertRaises(NachGPTContractError):
            validate_fact_packet(p, packet)

    def test_write_fact_packet_is_rejected_for_every_role(self):
        for role in (ROLE_CLIENT, ROLE_EMPLOYEE, ROLE_SUPERVISOR, ROLE_CEO, ROLE_SUPER_ADMIN):
            lab = "" if role == ROLE_SUPER_ADMIN else "LAB001"
            caps = ["own_case"] if role == ROLE_CLIENT else ["case"]
            if role == ROLE_SUPER_ADMIN:
                caps = ["global_labs"]
            p = principal(role, lab=lab, caps=caps)
            packet = {
                "verified": True,
                "read_only": False,
                "laboratory_id": "LAB001",
                "scope": "OWN_CASE" if role == ROLE_CLIENT else "LAB",
                "resource_owner_match": role == ROLE_CLIENT,
                "categories": {caps[0]: {}},
                "sources": [],
            }
            with self.assertRaises(NachGPTContractError):
                validate_fact_packet(p, packet)


class NachGPTIntentRouterTests(unittest.TestCase):
    def test_processor_audit_requires_processor_and_intake(self):
        routed = route_nachgpt_question("Audita la última corrida del procesador Gmail")
        self.assertTrue(routed["operational"])
        self.assertIn("processor", routed["required_categories"])
        self.assertIn("intake", routed["required_categories"])
        missing = missing_required_categories(
            "Audita la última corrida del procesador Gmail",
            {"processor": {}},
        )
        self.assertEqual(missing, ["intake"])

    def test_r2_case_audit_requires_physical_evidence(self):
        routed = route_nachgpt_question("Verifica si el caso Pepe está físicamente en R2")
        self.assertIn("r2_evidence", routed["required_categories"])
        self.assertIn("case_files", routed["required_categories"])

    def test_hyperdent_audit_needs_case_files_and_production(self):
        routed = route_nachgpt_question("Audita HyperDent del caso Terry")
        self.assertIn("hyperdent", routed["required_categories"])
        self.assertIn("production", routed["required_categories"])
        self.assertIn("case_files", routed["required_categories"])

    def test_finance_question_routes_to_finance(self):
        routed = route_nachgpt_question("Dame la facturación publicada de hoy")
        self.assertIn("finance", routed["required_categories"])


class EndpointWiringTests(unittest.TestCase):
    def test_nachgpt_endpoint_is_gateway_protected(self):
        src = (ROOT / "server" / "main.py").read_text(encoding="utf-8")
        self.assertIn('@app.post("/nachgpt/chat")', src)
        self.assertIn('NACHGPT_GATEWAY_TOKEN', src)
        self.assertIn('X-NACHGPT-Gateway-Token', src)
        self.assertIn('NACHGPT_OPERATIONAL_READ_ONLY', src)
        self.assertIn('build_role_aware_greeting', src)
        self.assertIn('build_nachgpt_fact_context', src)
        self.assertIn('missing_verified_fact_categories', src)
        self.assertIn('route_nachgpt_question', src)
        self.assertIn('if operational_mode:\n        return answer_text', src)


if __name__ == "__main__":
    unittest.main()


class OperationalFallbackTests(unittest.TestCase):
    def test_fallback_uses_verified_categories(self):
        from server.nachgpt_operational_fallback import build_operational_fallback

        packet = {
            "categories": {
                "processor": {"status": "HEALTHY", "processed": 12},
                "intake": {"overall": "HEALTHY"},
            },
            "sources": ["processor_run", "postgres_cases"],
        }
        answer = build_operational_fallback(
            "Audita la ultima corrida del procesador",
            packet,
            "es",
        )
        self.assertIn("Evidencia verificada de NACHGPT", answer)
        self.assertIn("processor:", answer)
        self.assertIn("status: HEALTHY", answer)
        self.assertIn("processed: 12", answer)
        self.assertIn("Fuentes: processor_run, postgres_cases", answer)
        self.assertIn("No se infirieron causas", answer)

    def test_processor_run_fallback_preserves_zero_false_and_formats_summary(self):
        from server.nachgpt_operational_fallback import build_operational_fallback

        packet = {
            "categories": {
                "processor": {"overall": "HEALTHY"},
                "intake": {"overall": "HEALTHY"},
                "processor_run": {
                    "run_id": 37,
                    "status": "SUCCESS",
                    "started_at": "2026-10-02T17:49:47+00:00",
                    "finished_at": "2026-10-02T18:10:51+00:00",
                    "processed": 0,
                    "new_cases": 0,
                    "case_names": [],
                    "technical_sheet_file_count": 0,
                    "technical_sheet_files": [],
                    "created_cases_during_run": [],
                    "updated_cases_during_run": [
                        {
                            "case_name": "subject_c_e8786a",
                            "patient_name": "Raul Camacho",
                            "status": "RECIBIDO",
                            "updated_at": "2026-10-02 18:10:50+00:00",
                            "current_file_count": 0,
                        }
                    ],
                    "files_added_during_run_count": 0,
                    "files_added_during_run": [],
                    "partial": False,
                    "time_budget_hit": False,
                    "skipped_already_processed": 3,
                    "skipped_existing_case": 0,
                    "repaired_cases": [],
                    "error": "",
                    "message": "Procesados 0 correos",
                },
            },
            "sources": ["dental_ai_intake_health_service", "processor_run_audit_service"],
        }
        answer = build_operational_fallback("Audita la última corrida del procesador", packet, "es")
        self.assertIn("Auditoría verificada del procesador", answer)
        self.assertIn("Correos procesados: 0", answer)
        self.assertIn("Casos nuevos reportados por el procesador: 0", answer)
        self.assertIn("Casos creados mapeados en Postgres durante la corrida: 0", answer)
        self.assertIn("Casos actualizados/tocados mapeados durante la corrida: 1", answer)
        self.assertIn("Archivos añadidos durante la corrida: 0", answer)
        self.assertIn("Raul Camacho | estado: RECIBIDO | archivos actuales: 0", answer)
        self.assertIn("Casos omitidos por existir previamente: 0", answer)
        self.assertIn("Corrida parcial: No", answer)
        self.assertIn("Límite de tiempo alcanzado: No", answer)
        self.assertIn("Error registrado: ninguno", answer)

    def test_fallback_without_categories_is_not_verified(self):
        from server.nachgpt_operational_fallback import build_operational_fallback

        answer = build_operational_fallback("Audita", {"categories": {}}, "es")
        self.assertTrue(answer.startswith("NO VERIFICADO:"))

    def test_endpoint_wires_deterministic_provider_failure_fallback(self):
        main_src = (ROOT / "server" / "main.py").read_text(encoding="utf-8")
        self.assertIn("build_operational_fallback", main_src)
        self.assertIn("if answer_text.strip() in _ERROR_MSGS.values()", main_src)
