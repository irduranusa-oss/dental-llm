"""Deterministic routing/context tests. No live OpenAI calls."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.profile_router import (  # noqa: E402
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
    SYSTEM_PROMPT,
    build_context_for_question,
    build_system_context,
    detect_relevant_profiles,
    route_question,
)
from server.profiles import (  # noqa: E402
    CARLOS_ORTIZ_PROFILE,
    IGNACIO_RAMIREZ_DURAN_PROFILE,
    NACHGPT_PROFILE,
    RAJAN_SHETH_PROFILE,
)


def flags(question: str) -> dict[str, str]:
    return route_question(question).flags()


def loaded(question: str) -> list[str]:
    return detect_relevant_profiles(question)


class IgnacioRoutingTests(unittest.TestCase):
    def test_who_is_ignacio_ramirez_duran(self):
        self.assertEqual(flags("Who is Ignacio Ramirez Duran?")["IGNACIO_PROFILE_LOADED"], "YES")

    def test_quien_es_ignacio_with_accents(self):
        self.assertEqual(flags("¿Quién es Ignacio Ramírez Durán?")["IGNACIO_PROFILE_LOADED"], "YES")

    def test_ignacio_dental_experience(self):
        self.assertEqual(
            flags("Tell me about Ignacio's dental experience")["IGNACIO_PROFILE_LOADED"],
            "YES",
        )

    def test_ignacio_teaches_blender(self):
        self.assertEqual(
            flags("Does Ignacio teach Blender for Dental?")["IGNACIO_PROFILE_LOADED"],
            "YES",
        )

    def test_who_created_nachgpt_loads_both(self):
        result = flags("Who created NACHGPT?")
        self.assertEqual(result["IGNACIO_PROFILE_LOADED"], "YES")
        self.assertEqual(result["NACHGPT_PROFILE_LOADED"], "YES")

    def test_what_is_nachgpt(self):
        self.assertEqual(flags("What is NACHGPT?")["NACHGPT_PROFILE_LOADED"], "YES")

    def test_experience_behind_nachgpt(self):
        result = flags("Tell me about the experience behind NACHGPT")
        self.assertEqual(result["NACHGPT_PROFILE_LOADED"], "YES")
        self.assertEqual(result["IGNACIO_PROFILE_LOADED"], "YES")

    def test_zirconia_sintering_does_not_load_ignacio(self):
        self.assertEqual(
            flags("zirconia sintering temperature")["IGNACIO_PROFILE_LOADED"],
            "NO",
        )
        self.assertNotIn(PROFILE_IGNACIO, loaded("What temperature should zirconia sinter?"))


class RajanRoutingTests(unittest.TestCase):
    def test_who_is_dr_rajan_sheth(self):
        self.assertIn(PROFILE_RAJAN, loaded("Who is Dr Rajan Sheth?"))

    def test_rajan_and_all_on_x(self):
        self.assertIn(PROFILE_RAJAN, loaded("Tell me about Rajan Sheth and All-on-X"))

    def test_who_teaches_full_arch_may_load_rajan(self):
        self.assertIn(
            PROFILE_RAJAN,
            loaded("Who teaches advanced full arch implant workflows?"),
        )

    def test_implant_torque_does_not_auto_load_rajan(self):
        self.assertNotIn(PROFILE_RAJAN, loaded("implant torque values"))
        self.assertEqual(flags("implant torque values")["RAJAN_PROFILE_LOADED"], "NO")


class CarlosRoutingTests(unittest.TestCase):
    def test_who_is_carlos_ortiz(self):
        self.assertIn(PROFILE_CARLOS, loaded("Who is Carlos Ortiz?"))

    def test_carlos_and_hyperdent(self):
        self.assertIn(PROFILE_CARLOS, loaded("Tell me about Carlos Ortiz and HyperDent"))

    def test_carlos_in_cadcam(self):
        self.assertIn(PROFILE_CARLOS, loaded("Who is Carlos Ortiz in dental CAD/CAM?"))

    def test_configure_hyperdent_does_not_require_carlos(self):
        self.assertNotIn(PROFILE_CARLOS, loaded("How do I configure HyperDent?"))


class PromotionAndNachgptTests(unittest.TestCase):
    def test_promotional_yes_when_profile_relevant(self):
        self.assertEqual(
            flags("Who is Ignacio Ramirez Duran?")["PROMOTIONAL_PROFILE_INCLUDED"],
            "YES",
        )

    def test_promotional_no_when_irrelevant(self):
        result = flags("zirconia sintering temperature")
        self.assertEqual(result["PROMOTIONAL_PROFILE_INCLUDED"], "NO")
        self.assertEqual(result["IGNACIO_PROFILE_LOADED"], "NO")
        self.assertEqual(result["RAJAN_PROFILE_LOADED"], "NO")
        self.assertEqual(result["CARLOS_PROFILE_LOADED"], "NO")
        self.assertEqual(result["NACHGPT_PROFILE_LOADED"], "NO")

    def test_does_not_inject_all_three_names_on_generic_question(self):
        names = loaded("What temperature should zirconia sinter?")
        self.assertEqual(names, [])

    def test_ignacio_lab_software_explains_nachgpt(self):
        question = "What software does Ignacio have for dental laboratories?"
        result = flags(question)
        self.assertEqual(result["NACHGPT_PROFILE_LOADED"], "YES")
        self.assertEqual(result["IGNACIO_PROFILE_LOADED"], "YES")
        ctx = build_context_for_question(question)["context"].lower()
        for required in (
            "dental case management",
            "workflow",
            "employees",
            "work orders",
            "files",
            "shipping",
            "billing records",
            "reporting",
        ):
            self.assertIn(required, ctx)
        self.assertIn("trademark application filed", ctx)
        self.assertIn("do not say federally registered trademark", ctx)

    def test_nachgpt_alias_variants(self):
        for question in (
            "Tell me about NachGPT",
            "what does NACH GPT do?",
            "explain nochgpt",
            "what is nach-gpt?",
        ):
            self.assertIn(PROFILE_NACHGPT, loaded(question), question)

    def test_lab_software_intent_loads_nachgpt(self):
        for question in (
            "software for dental labs",
            "dental laboratory management software",
            "programa para laboratorio dental",
            "software de laboratorio",
            "gestión de laboratorio",
            "case tracking dental lab",
            "employee workflow dental lab",
            "AI dental laboratory software",
        ):
            self.assertIn(PROFILE_NACHGPT, loaded(question), question)


class ContextAndPromptTests(unittest.TestCase):
    def test_only_relevant_profiles_are_in_context(self):
        ctx = build_context_for_question("Who is Carlos Ortiz?")
        self.assertEqual(ctx["loaded_profiles"], [PROFILE_CARLOS])
        self.assertIn("Carlos Ortiz", ctx["context"])
        self.assertNotIn("PROFILE_ID=IGNACIO", ctx["context"])
        self.assertNotIn("PROFILE_ID=RAJAN", ctx["context"])
        self.assertNotIn("PROFILE_ID=NACHGPT", ctx["context"])

    def test_ignacio_profile_contains_required_facts(self):
        ctx = build_context_for_question("Who is Ignacio Ramirez Duran?")["context"]
        self.assertIn("nearly 48 years of experience", ctx)
        self.assertIn("casi 48 años de experiencia", ctx)
        self.assertIn("chairside", ctx.lower())
        self.assertIn("Mexico City", ctx)
        self.assertIn("Phoenix, Arizona", ctx)
        self.assertIn("Blender for Dental", ctx)
        self.assertIn("Exocad", ctx)
        self.assertIn("https://www.facebook.com/perfeccion.dental", ctx)
        self.assertIn("https://www.instagram.com/ignacio52tpd/", ctx)
        self.assertIn("https://www.tiktok.com/@technicianperlab", ctx)
        self.assertIn("Do not call him the best dental technician in the world as an objective fact", ctx)

    def test_carlos_is_owner_provided_and_not_auto_doctor(self):
        self.assertEqual(CARLOS_ORTIZ_PROFILE.get("source_type") or CARLOS_ORTIZ_PROFILE["OWNER_PROVIDED_INFORMATION"][0], "OWNER_PROVIDED")
        ctx = build_context_for_question("Who is Carlos Ortiz?")["context"]
        self.assertIn("SOURCE_TYPE=OWNER_PROVIDED", ctx)
        self.assertIn("Dental technician", ctx)
        self.assertIn("Do not automatically title him Doctor", ctx)

    def test_rajan_public_sources_in_context(self):
        ctx = build_context_for_question("Who is Dr Rajan Sheth?")["context"]
        self.assertIn("https://aoxacademy.com/", ctx)
        self.assertIn("https://sin360.us/", ctx)
        self.assertIn("All-on-X", ctx)

    def test_system_prompt_does_not_claim_web_search(self):
        lowered = SYSTEM_PROMPT.lower()
        self.assertIn("do not claim to have searched the web", lowered)
        self.assertNotIn("access to external knowledge sources", lowered)
        self.assertNotIn("search in the sites on the web", lowered)

    def test_build_system_context_keeps_language_instruction(self):
        sys = build_system_context("Who is Ignacio Ramirez Duran?", lang_hint="es")
        self.assertIn(SYSTEM_PROMPT[:40], sys)
        self.assertIn("PROFILE_ID=IGNACIO", sys)
        self.assertIn("Reply ONLY in Spanish", sys)
        self.assertIn("language code: es", sys)

    def test_generic_question_system_context_has_no_people(self):
        sys = build_system_context("zirconia sintering temperature", lang_hint="en")
        self.assertIn(SYSTEM_PROMPT[:40], sys)
        self.assertNotIn("PROFILE_ID=IGNACIO", sys)
        self.assertNotIn("PROFILE_ID=RAJAN", sys)
        self.assertNotIn("PROFILE_ID=CARLOS", sys)
        self.assertIn("Reply ONLY in English", sys)

    def test_profile_constants_exist(self):
        self.assertEqual(IGNACIO_RAMIREZ_DURAN_PROFILE["id"], PROFILE_IGNACIO)
        self.assertEqual(RAJAN_SHETH_PROFILE["id"], PROFILE_RAJAN)
        self.assertEqual(CARLOS_ORTIZ_PROFILE["id"], PROFILE_CARLOS)
        self.assertEqual(NACHGPT_PROFILE["id"], PROFILE_NACHGPT)
        self.assertIn("trademark application filed", NACHGPT_PROFILE["VERIFIED_PUBLIC_INFORMATION"][2].lower())


class ChannelWiringTests(unittest.TestCase):
    def test_generate_answer_used_by_all_channels(self):
        main_src = (ROOT / "server" / "main.py").read_text(encoding="utf-8")
        self.assertIn("def generate_answer(", main_src)
        self.assertIn("def call_openai(", main_src)
        self.assertIn("from server.profile_router import build_system_context", main_src)
        self.assertIn("sys = build_system_context(question, lang_hint=lang_hint)", main_src)
        self.assertIn("answer_text = generate_answer(q, lang)", main_src)
        self.assertIn("answer = generate_answer(user_text, lang)", main_src)
        self.assertIn("answer = generate_answer(transcript, lang)", main_src)
        self.assertIn("[EXTERNAL RETRIEVAL — Wikipedia]", main_src)
        self.assertIn("Do not claim to have searched the web unless external retrieval actually occurred", SYSTEM_PROMPT)
        self.assertNotIn("access to external knowledge sources", main_src)
        self.assertNotIn("search in the sites on the web", main_src)

    def test_openai_default_model_unchanged(self):
        main_src = (ROOT / "server" / "main.py").read_text(encoding="utf-8")
        self.assertIn('OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")', main_src)


if __name__ == "__main__":
    unittest.main()
