"""Author-inspired style remains non-authoritative and respects verified facts."""
from server.nachgpt_contract import (
    tlantepahtic_author_voice_rules, nachgpt_operational_system_rules,
)
from server.nachgpt_intent_router import (
    route_nachgpt_question,missing_required_categories,
)


def test_owner_persona_is_professional_not_identity_spoofing():
    text=tlantepahtic_author_voice_rules()
    assert "CAD design" in text and "implant capture" in text
    assert "INPUT -> AUDIT -> CORRECTION -> VALIDATION" in text
    assert "never pretend to BE the book author" in text
    assert "not a universal" in text


def test_persona_is_secondary_to_existing_contract_guards():
    system=nachgpt_operational_system_rules()
    assert "READ ONLY" in system
    assert "Never expand permissions" in system
    assert "NO VERIFICADO" in system
    assert "TLANTEPAHTIC_AUTHOR_VOICE" in system
    assert "clinical" in system and "physical" in system


def test_owner_book_question_is_training_not_case_bound_production():
    query="Explica mi flujo de trabajo para Blender en el libro"
    routed=route_nachgpt_question(query)
    assert routed["required_categories"]==["training"]
    assert missing_required_categories(query,{"training":{"author_workflow":{"ok":True}}})==[]
    assert missing_required_categories(query,{"blender":{}})==["training"]


def test_active_specific_case_still_requires_case_evidence():
    query="Explica mi flujo de trabajo para Blender en este caso"
    routed=route_nachgpt_question(query)
    assert "blender" in routed["required_categories"]
    assert "case_files" in routed["required_categories"]


def test_not_every_cad_question_becomes_author_workflow():
    routed=route_nachgpt_question("Que archivos Exocad tiene el caso?")
    assert "training" not in routed["required_categories"]


def test_ordinary_client_case_requests_keep_original_routing():
    routed=route_nachgpt_question("Muéstrame el caso del paciente")
    assert "case" in routed["required_categories"]
    assert "training" not in routed["required_categories"]
