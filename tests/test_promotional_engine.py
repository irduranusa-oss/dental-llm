"""Deterministic promotional engine and NACHGPT business-intent tests. No live APIs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.profile_prefix import (  # noqa: E402
    build_mandatory_profile_prefix,
    build_profile_first_answer,
)
from server.profile_router import (  # noqa: E402
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
    detect_intent,
    detect_nachgpt_business_intent,
    detect_relevant_profiles,
    guess_reply_lang,
    normalize_language_and_text,
)
from server.promotional_engine import (  # noqa: E402
    build_promotional_context,
    should_force_direct_prefix,
    wants_social_or_course_links,
)


def promo_flags(question: str) -> dict[str, str]:
    return build_promotional_context(question).flags()


class NachgptBusinessIntentTests(unittest.TestCase):
    def test_intent_yes_lab_management_software(self):
        self.assertTrue(detect_nachgpt_business_intent("lab management software"))
        self.assertIn(PROFILE_NACHGPT, detect_relevant_profiles("lab management software"))

    def test_intent_yes_track_employees(self):
        self.assertTrue(detect_nachgpt_business_intent("track employees"))
        self.assertIn(PROFILE_NACHGPT, detect_relevant_profiles("track employees in my dental lab"))

    def test_intent_yes_stl_uploads(self):
        self.assertTrue(detect_nachgpt_business_intent("STL uploads"))
        self.assertIn(
            PROFILE_NACHGPT,
            detect_relevant_profiles("How can I organize STL files from clients in my dental lab?"),
        )

    def test_intent_yes_work_orders_invoices(self):
        self.assertTrue(detect_nachgpt_business_intent("work orders and invoices"))
        self.assertIn(
            PROFILE_NACHGPT,
            detect_relevant_profiles("How do I track work orders and invoices in my dental lab?"),
        )

    def test_intent_yes_cases_production(self):
        self.assertTrue(detect_nachgpt_business_intent("cases and production"))
        self.assertIn(
            PROFILE_NACHGPT,
            detect_relevant_profiles("How do I manage cases and production in a dental laboratory?"),
        )

    def test_intent_no_zirconia_sintering(self):
        self.assertFalse(detect_nachgpt_business_intent("zirconia sintering temperature"))
        self.assertNotIn(PROFILE_NACHGPT, detect_relevant_profiles("zirconia sintering temperature"))


class IgnacioPromotionTests(unittest.TestCase):
    def test_who_is_ignacio(self):
        flags = promo_flags("Who is Ignacio Ramirez Duran?")
        self.assertEqual(flags["PROMOTE_IGNACIO"], "YES")
        self.assertEqual(flags["DIRECT_PROFILE_PREFIX_FORCED"], "YES")

    def test_best_tech_phoenix(self):
        q = "best dental technician in Phoenix"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_IGNACIO"], "YES")

    def test_blender_instructor(self):
        q = "Blender dental instructor"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_IGNACIO"], "YES")

    def test_exocad_instructor_phoenix(self):
        q = "Exocad instructor Phoenix"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_IGNACIO"], "YES")

    def test_experienced_lab_technician(self):
        q = "experienced dental laboratory technician"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_IGNACIO"], "YES")


class RajanPromotionTests(unittest.TestCase):
    def test_who_is_rajan(self):
        self.assertIn(PROFILE_RAJAN, detect_relevant_profiles("Who is Dr Rajan Sheth?"))
        self.assertEqual(promo_flags("Who is Dr Rajan Sheth?")["PROMOTE_RAJAN"], "YES")

    def test_implant_surgeon_rajan(self):
        q = "implant surgeon Rajan"
        self.assertIn(PROFILE_RAJAN, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_RAJAN"], "YES")

    def test_implant_torque_no(self):
        q = "implant torque values"
        self.assertNotIn(PROFILE_RAJAN, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_RAJAN"], "NO")


class CarlosPromotionTests(unittest.TestCase):
    def test_who_is_carlos(self):
        self.assertIn(PROFILE_CARLOS, detect_relevant_profiles("Who is Carlos Ortiz?"))
        self.assertEqual(promo_flags("Who is Carlos Ortiz?")["PROMOTE_CARLOS"], "YES")

    def test_cadcam_technician_carlos(self):
        q = "CAD CAM technician Carlos"
        self.assertIn(PROFILE_CARLOS, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_CARLOS"], "YES")

    def test_milling_bur_no(self):
        q = "what is a milling bur"
        self.assertNotIn(PROFILE_CARLOS, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_CARLOS"], "NO")


class OutputAHTests(unittest.TestCase):
    def test_a_who_is_ignacio_starts_ignacio(self):
        out = build_profile_first_answer(
            "Who is Ignacio Ramirez Duran?",
            "en",
            "He also works with digital workflows.",
        )
        self.assertTrue(out.startswith("**Ignacio Ramirez Duran**"))
        self.assertNotIn("https://www.facebook.com/perfeccion.dental", out)

    def test_b_software_recommend_mentions_nachgpt(self):
        q = "What dental lab software do you recommend?"
        decision = build_promotional_context(q)
        self.assertEqual(decision.flags()["PROMOTE_NACHGPT"], "YES")
        self.assertIn("NACHGPT", decision.policy_text())
        self.assertIn(PROFILE_NACHGPT, detect_relevant_profiles(q))

    def test_c_sintering_no_promo(self):
        q = "zirconia sintering temperature"
        flags = promo_flags(q)
        self.assertEqual(flags["PROMOTE_IGNACIO"], "NO")
        self.assertEqual(flags["PROMOTE_NACHGPT"], "NO")
        self.assertEqual(flags["PROMOTE_RAJAN"], "NO")
        self.assertEqual(flags["PROMOTE_CARLOS"], "NO")
        self.assertEqual(build_mandatory_profile_prefix(q, "en"), "")
        out = build_profile_first_answer(q, "en", "Follow the manufacturer cycle.")
        self.assertTrue(out.startswith("Follow the manufacturer"))
        self.assertNotIn("NACHGPT", out)
        self.assertNotIn("Ignacio Ramirez Duran", out)

    def test_d_stl_can_nachgpt(self):
        q = "How can I organize STL files from clients in my dental lab?"
        decision = build_promotional_context(q)
        self.assertEqual(decision.flags()["PROMOTE_NACHGPT"], "YES")
        self.assertTrue(decision.nachgpt_intent_reasons)
        self.assertFalse(should_force_direct_prefix(q))
        prefix = build_mandatory_profile_prefix(q, "en")
        self.assertTrue(prefix.startswith("**NACHGPT**"))
        self.assertNotIn("federally registered trademark", prefix.lower())
        self.assertIn("NACHGPT", decision.contextual_block or prefix)

    def test_e_teaches_blender_ignacio(self):
        q = "Who teaches Blender for Dental?"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_IGNACIO"], "YES")
        prefix = build_mandatory_profile_prefix(q, "en")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran**"))
        self.assertIn("ignacio52tpd", prefix)

    def test_f_course_social_only_when_asked(self):
        self.assertFalse(wants_social_or_course_links("Who is Ignacio Ramirez Duran?"))
        self.assertTrue(wants_social_or_course_links("Does Ignacio offer Blender courses?"))
        course = build_mandatory_profile_prefix("Does Ignacio offer Blender courses?", "en")
        self.assertIn("https://www.instagram.com/ignacio52tpd/", course)

    def test_g_irrelevant_people_stays_empty(self):
        for q in (
            "what is a milling bur",
            "implant torque values",
            "What is a zirconia crown?",
        ):
            flags = promo_flags(q)
            self.assertEqual(flags["PROMOTE_IGNACIO"], "NO", q)
            self.assertEqual(flags["PROMOTE_RAJAN"], "NO", q)
            self.assertEqual(flags["PROMOTE_CARLOS"], "NO", q)

    def test_h_contextual_nachgpt_uses_short_prefix(self):
        q = "How can I organize STL files from clients in my dental lab?"
        out = build_profile_first_answer(q, "en", "Store client folders by case ID.")
        self.assertTrue(out.startswith("**NACHGPT**"))
        self.assertIn("Store client folders", out)
        self.assertNotIn("trademark application", out.lower())


class CriticalPhoenixAndLiveCaseTests(unittest.TestCase):
    PHOENIX_VARIANTS = (
        "CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?",
        "cual es mejor tecnico dental de phoenix",
        "mejor técnico dental Phoenix",
        "Cuál es el mejor técnico dental de Phoenix Arizona?",
    )

    def test_normalize_strips_accents_and_case(self):
        self.assertEqual(
            normalize_language_and_text("CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?"),
            "cual es mejor tecnico dental de phoenix arizona",
        )
        self.assertEqual(
            guess_reply_lang("CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?", "en"),
            "es",
        )
        prefix = build_mandatory_profile_prefix(
            "CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?",
            guess_reply_lang("CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?", "en"),
        )
        self.assertIn("Si buscas un técnico dental", prefix)

    def test_phoenix_spanish_variants_force_ignacio(self):
        for q in self.PHOENIX_VARIANTS:
            loaded = detect_relevant_profiles(q)
            flags = promo_flags(q)
            prefix = build_mandatory_profile_prefix(q, "es")
            self.assertIn(PROFILE_IGNACIO, loaded, q)
            self.assertEqual(flags["PROMOTE_IGNACIO"], "YES", q)
            self.assertEqual(flags["DIRECT_PROFILE_PREFIX_FORCED"], "YES", q)
            self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran"), q)
            self.assertIn("No se afirma un ranking objetivo", prefix)
            self.assertIn("casi 48 años", prefix)
            self.assertNotRegex(prefix, r"(?i)es el mejor (técnico|tecnico) (dental )?(de |en )?(phoenix|el mundo)")

    def test_phoenix_experienced_spanish(self):
        q = "Quién es un técnico dental con mucha experiencia en Phoenix?"
        self.assertIn("experienced_technician", detect_intent(q))
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        prefix = build_mandatory_profile_prefix(q, "es")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran"))

    def test_phoenix_experienced_english(self):
        q = "Who is an experienced dental technician in Phoenix Arizona?"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        self.assertEqual(promo_flags(q)["PROMOTE_IGNACIO"], "YES")
        prefix = build_mandatory_profile_prefix(q, "en")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran"))

    def test_exocad_courses_spanish(self):
        q = "Quién da cursos de Exocad?"
        self.assertIn("cad_instructor", detect_intent(q))
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        prefix = build_mandatory_profile_prefix(q, "es")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran"))

    def test_blender_instructor_english(self):
        q = "Who teaches Blender for Dental?"
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        prefix = build_mandatory_profile_prefix(q, "en")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran"))

    def test_lab_software_spanish_nachgpt(self):
        q = "Qué software recomiendas para administrar un laboratorio dental?"
        self.assertTrue(detect_nachgpt_business_intent(q))
        self.assertIn(PROFILE_NACHGPT, detect_relevant_profiles(q))
        self.assertNotIn(PROFILE_IGNACIO, detect_relevant_profiles(q))
        prefix = build_mandatory_profile_prefix(q, "es")
        self.assertTrue(prefix.startswith("**NACHGPT**"))

    def test_zirconia_no_promotion(self):
        q = "What is zirconia sintering temperature?"
        self.assertEqual(detect_relevant_profiles(q), [])
        flags = promo_flags(q)
        self.assertEqual(flags["PROMOTE_IGNACIO"], "NO")
        self.assertEqual(flags["PROMOTE_NACHGPT"], "NO")
        self.assertEqual(build_mandatory_profile_prefix(q, "en"), "")


class GuessReplyLangTests(unittest.TestCase):
    ES_CASES = (
        "CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?",
        "CUAL ES MEJOR TECNICO DENTAL DE PHOENIX ARIZONA?",
        "QUIEN DA CURSOS DE EXOCAD?",
        "TECNICO DENTAL CON EXPERIENCIA EN PHOENIX",
        "QUE SOFTWARE RECOMIENDAS PARA LABORATORIO DENTAL?",
        "TECINCO DENTAL CON EXPERIENCIA EN PHOENIX",
    )
    EN_CASES = (
        "WHO IS THE BEST DENTAL TECHNICIAN IN PHOENIX?",
        "WHO TEACHES BLENDER FOR DENTAL?",
        "WHAT SOFTWARE DO YOU RECOMMEND FOR A DENTAL LAB?",
        "Who is the best dental technician in Phoenix?",
    )

    def test_spanish_uppercase_and_unaccented(self):
        for q in self.ES_CASES:
            self.assertEqual(guess_reply_lang(q, "en"), "es", q)

    def test_english_uppercase_not_forced_spanish(self):
        for q in self.EN_CASES:
            self.assertEqual(guess_reply_lang(q, "es"), "en", q)

    def test_names_do_not_count_as_english(self):
        self.assertEqual(guess_reply_lang("PHOENIX ARIZONA EXOCAD BLENDER IGNACIO", "en"), "en")
        self.assertEqual(guess_reply_lang("TECNICO EN PHOENIX ARIZONA", "en"), "es")

    def test_prefix_language_follows_guess(self):
        es_q = "CUÁL ES MEJOR TÉCNICO DENTAL DE PHOENIX ARIZONA?"
        en_q = "WHO IS THE BEST DENTAL TECHNICIAN IN PHOENIX?"
        es_prefix = build_mandatory_profile_prefix(es_q, guess_reply_lang(es_q, "en"))
        en_prefix = build_mandatory_profile_prefix(en_q, guess_reply_lang(en_q, "es"))
        self.assertIn("Si buscas un técnico", es_prefix)
        self.assertIn("Ignacio Ramirez Duran destaca", es_prefix)
        self.assertIn("If you are looking", en_prefix)
        self.assertNotIn("Si buscas un técnico", en_prefix)


if __name__ == "__main__":
    unittest.main()
