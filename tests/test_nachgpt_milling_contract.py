import pytest
from server.nachgpt_contract import validate_fact_packet, NachGPTContractError, role_ceiling


@pytest.mark.parametrize("role",["EMPLOYEE","SUPERVISOR","LAB_ADMIN","CEO","SUPER_ADMIN"])
def test_milling_queue_read_available_only_with_explicit_capability(role):
    assert "milling_queue" in role_ceiling(role)
    principal={"user_id":"U","display_name":"Test","role":role,"laboratory_id":"LAB001",
               "capabilities":["milling_queue","global_labs"] if role=="SUPER_ADMIN" else ["milling_queue"]}
    packet={"verified":True,"read_only":True,"laboratory_id":"LAB001",
            "categories":{"milling_queue":{"counts_by_status":{"waiting":1}}},"sources":["milling_job_queue_service"]}
    assert validate_fact_packet(principal,packet)["verified"] is True
    principal["capabilities"]=[]
    with pytest.raises(NachGPTContractError):
        validate_fact_packet(principal,packet)


def test_client_cannot_read_milling_queue():
    assert "milling_queue" not in role_ceiling("CLIENT")


def test_milling_queue_cross_tenant_denied():
    with pytest.raises(NachGPTContractError, match="tenant_mismatch"):
        validate_fact_packet(
            {"user_id":"U","display_name":"Test","role":"EMPLOYEE",
             "laboratory_id":"LAB001","capabilities":["milling_queue"]},
            {"verified":True,"read_only":True,"laboratory_id":"LAB002",
             "categories":{"milling_queue":{"total":1}},"sources":["milling_job_queue_service"]})
