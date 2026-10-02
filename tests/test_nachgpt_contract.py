from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


class EndpointWiringTests(unittest.TestCase):
    def test_nachgpt_endpoint_is_gateway_protected(self):
        src = (ROOT / "server" / "main.py").read_text(encoding="utf-8")
        self.assertIn('@app.post("/nachgpt/chat")', src)
        self.assertIn('NACHGPT_GATEWAY_TOKEN', src)
        self.assertIn('X-NACHGPT-Gateway-Token', src)
        self.assertIn('NACHGPT_OPERATIONAL_READ_ONLY', src)
        self.assertIn('build_role_aware_greeting', src)
        self.assertIn('build_nachgpt_fact_context', src)


if __name__ == "__main__":
    unittest.main()
