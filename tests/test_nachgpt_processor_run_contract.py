from server.nachgpt_contract import validate_fact_packet


def test_superadmin_accepts_verified_processor_run_category():
    principal = {
        "user_id": "0578",
        "display_name": "Ignacio",
        "role": "SUPER_ADMIN",
        "laboratory_id": "",
        "capabilities": ["global_labs", "processor", "processor_run", "intake"],
    }
    packet = {
        "verified": True,
        "read_only": True,
        "laboratory_id": "LAB001",
        "target_laboratory_id": "LAB001",
        "scope": "PLATFORM",
        "resource_owner_match": True,
        "categories": {
            "processor_run": {"status": "SUCCESS", "run_id": 42},
            "processor": {"status": "OK"},
            "intake": {"overall": "OK"},
        },
        "sources": ["processor_run_audit_service"],
    }
    got = validate_fact_packet(principal, packet)
    assert got["categories"]["processor_run"]["run_id"] == 42
