"""NACHGPT external LLM must enforce purchases category and finance capability."""
import pytest

from server.nachgpt_contract import (
    NachGPTContractError, role_ceiling, validate_fact_packet,
)


def packet():
    return {
        "verified": True,
        "read_only": True,
        "laboratory_id": "LAB001",
        "target_laboratory_id": "LAB001",
        "scope": "LAB",
        "categories": {
            "purchases": {"purchase_count": 2, "purchases": [
                {"purchase_id": "PUR-001", "status": "active"}
            ]}
        },
        "sources": ["purchases_store_service"],
    }


def principal(role, caps):
    return {
        "role": role, "user_id": "U1", "display_name": "Authorized",
        "laboratory_id": "LAB001",
        "capabilities": list(caps),
    }


@pytest.mark.parametrize("role", ["CEO", "SUPER_ADMIN", "LAB_ADMIN"])
def test_purchases_requires_explicit_finance_capability(role):
    assert "purchases" in role_ceiling(role)
    required = ["purchases", "finance"] + (["global_labs"] if role == "SUPER_ADMIN" else [])
    verified = validate_fact_packet(principal(role, required), packet())
    assert verified["categories"]["purchases"]["purchase_count"] == 2
    with pytest.raises(NachGPTContractError):
        validate_fact_packet(principal(role, ["purchases"] + (["global_labs"] if role == "SUPER_ADMIN" else [])), packet())


@pytest.mark.parametrize("role", ["EMPLOYEE", "SUPERVISOR", "CLIENT"])
def test_purchases_never_available_to_nonfinancial_role(role):
    with pytest.raises(NachGPTContractError):
        validate_fact_packet(principal(role, ["purchases", "finance"]), packet())


def test_purchases_tenant_boundary_and_read_only():
    p = principal("CEO", ["purchases", "finance"])
    foreign = packet()
    foreign["laboratory_id"] = "LAB002"
    foreign["target_laboratory_id"] = "LAB002"
    with pytest.raises(NachGPTContractError, match="tenant_mismatch"):
        validate_fact_packet(p, foreign)
    mutable = packet()
    mutable["read_only"] = False
    with pytest.raises(NachGPTContractError, match="facts_not_read_only"):
        validate_fact_packet(p, mutable)
