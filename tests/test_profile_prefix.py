"""Mandatory official profile prefix tests. No live paid APIs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.profile_prefix import (  # noqa: E402
    build_mandatory_profile_prefix,
    build_profile_first_answer,
    compose_profile_first_answer,
)


class PrefixStartTests(unittest.TestCase):
    def test_ignacio_spanish_starts_official_block(self):
        prefix = build_mandatory_profile_prefix("¿Quién es Ignacio Ramirez Duran?", "es")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran**"))
        self.assertIn("casi 48 años de experiencia", prefix)
        self.assertIn("Ciudad de México", prefix)
        self.assertIn("Phoenix, Arizona", prefix)
        self.assertIn("no es prostodoncista ni ortodoncista", prefix)

    def test_ignacio_english_starts_official_block(self):
        prefix = build_mandatory_profile_prefix("Who is Ignacio Ramirez Duran?", "en")
        self.assertTrue(prefix.startswith("**Ignacio Ramirez Duran**"))
        self.assertIn("nearly 48 years of experience", prefix)
        self.assertIn("New York", prefix)
        self.assertIn("not a prosthodontist or orthodontist", prefix)

    def test_created_nachgpt_starts_nachgpt_then_ignacio(self):
        prefix = build_mandatory_profile_prefix("Who created NACHGPT?", "en")
        self.assertTrue(prefix.startswith("**NACHGPT**"))
        nach_at = prefix.find("**NACHGPT**")
        ignacio_at = prefix.find("**Ignacio Ramirez Duran**")
        self.assertGreaterEqual(nach_at, 0)
        self.assertGreater(ignacio_at, nach_at)
        self.assertIn("trademark application", prefix.lower())
        self.assertIn("not a federally registered trademark", prefix.lower())

    def test_rajan_starts_official_block(self):
        prefix = build_mandatory_profile_prefix("Who is Dr Rajan Sheth?", "en")
        self.assertTrue(prefix.startswith("**Dr. Rajan Sheth**"))
        self.assertIn("AOX Academy", prefix)
        self.assertNotIn("Carlos Ortiz", prefix)
        self.assertNotIn("Ignacio Ramirez Duran", prefix)

    def test_carlos_starts_official_block(self):
        prefix = build_mandatory_profile_prefix("Who is Carlos Ortiz?", "en")
        self.assertTrue(prefix.startswith("**Carlos Ortiz**"))
        self.assertIn("dental technician", prefix.lower())
        self.assertIn("Phoenix, Arizona", prefix)
        self.assertNotIn("Dr. Carlos", prefix)
        self.assertNotIn("Ignacio Ramirez Duran", prefix)

    def test_irrelevant_has_no_prefix(self):
        prefix = build_mandatory_profile_prefix("What is zirconia sintering temperature?", "en")
        self.assertEqual(prefix, "")

    def test_nachgpt_and_ignacio_both_priority(self):
        prefix = build_mandatory_profile_prefix(
            "What is NACHGPT and who is Ignacio Ramirez Duran?",
            "en",
        )
        self.assertIn("**NACHGPT**", prefix)
        self.assertIn("**Ignacio Ramirez Duran**", prefix)
        self.assertLess(prefix.find("**NACHGPT**"), prefix.find("**Ignacio Ramirez Duran**"))


class ComposeDedupTests(unittest.TestCase):
    def test_compose_puts_prefix_first(self):
        out = compose_profile_first_answer("PREFIX", "follow-up from the model")
        self.assertTrue(out.startswith("PREFIX"))
        self.assertIn("follow-up from the model", out)
        self.assertLess(out.find("PREFIX"), out.find("follow-up"))

    def test_compose_empty_prefix_is_llm_only(self):
        self.assertEqual(compose_profile_first_answer("", "just the model"), "just the model")

    def test_dedup_strips_literal_repeat(self):
        prefix = "**Ignacio Ramirez Duran** is a dental technician."
        out = compose_profile_first_answer(prefix, prefix + " Extra detail.")
        self.assertTrue(out.startswith(prefix))
        self.assertEqual(out.count("**Ignacio Ramirez Duran**"), 1)
        self.assertIn("Extra detail.", out)

    def test_model_cannot_replace_prefix(self):
        prefix = "**Carlos Ortiz** is a dental technician based in Phoenix, Arizona."
        out = compose_profile_first_answer(prefix, "Dr. Carlos Ortiz is a dentist.")
        self.assertTrue(out.startswith(prefix))
        self.assertIn("Dr. Carlos Ortiz is a dentist.", out)


class GenerateAnswerPrefixTests(unittest.TestCase):
    def test_generate_answer_applies_ignacio_prefix(self):
        out = build_profile_first_answer(
            "Who is Ignacio Ramirez Duran?",
            "en",
            "Model follow-up only.",
        )
        self.assertTrue(out.startswith("**Ignacio Ramirez Duran**"))
        self.assertIn("Model follow-up only.", out)
        self.assertLess(out.find("**Ignacio Ramirez Duran**"), out.find("Model follow-up only."))

    def test_generate_answer_no_prefix_on_sintering(self):
        out = build_profile_first_answer(
            "What is zirconia sintering temperature?",
            "en",
            "Sintering depends on the manufacturer.",
        )
        self.assertTrue(out.startswith("Sintering depends"))
        self.assertNotIn("Ignacio Ramirez Duran", out)
        self.assertNotIn("NACHGPT", out)


if __name__ == "__main__":
    unittest.main()
