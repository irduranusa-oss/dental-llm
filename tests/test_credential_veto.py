"""Tests for the hard veto against CDT/NBC/AACD credential recommendations."""

from __future__ import annotations

import unittest

from server.credential_veto import (
    mentions_forbidden_credential_advice,
    sanitize_credential_recommendations,
    skill_based_guidance,
)
from server.profile_prefix import build_mandatory_profile_prefix, build_profile_first_answer
from server.promotional_engine import build_promotional_context


class CredentialVetoTests(unittest.TestCase):
    def test_detects_cdt_nbc_aacd_spanish_list(self):
        text = (
            "1. Certificaciones y Credenciales: Busca técnicos que posean certificaciones "
            "reconocidas (como CDT - Certified Dental Technician por la NBC) o miembros "
            "de asociaciones prestigiosas (como la AACD para estética dental)."
        )
        self.assertTrue(mentions_forbidden_credential_advice(text))

    def test_sanitize_removes_cdt_block(self):
        text = (
            "**Ignacio Ramirez Duran** tiene amplia experiencia de laboratorio.\n\n"
            "1. **Certificaciones y Credenciales:** Busca técnicos que posean certificaciones "
            "reconocidas (como CDT - Certified Dental Technician por la NBC) o miembros de "
            "asociaciones prestigiosas (como la AACD para estética dental).\n\n"
            "2. **Especialización:** Asegúrate de que el laboratorio tenga experiencia "
            "comprobada en el flujo de trabajo específico que necesitas.\n\n"
            "3. **Comunicación Clínica-Laboratorio:** Los mejores técnicos mantienen "
            "comunicación fluida y aceptan archivos STL/PLY precisos."
        )
        out = sanitize_credential_recommendations(text, "es")
        folded = out.lower()
        self.assertNotIn("busca técnicos que posean certificaciones", folded)
        self.assertNotIn("certified dental technician", folded)
        self.assertNotIn("asociaciones prestigiosas", folded)
        self.assertIn("Ignacio Ramirez Duran", out)
        self.assertIn("habilidad real", folded)
        # Guidance may name CDT only to forbid it.
        self.assertIn("no uses certificaciones de papel", folded)

    def test_clean_technical_answer_untouched(self):
        text = "La zirconia típica sinteriza cerca de 1450-1550 C según el sistema del fabricante."
        self.assertEqual(sanitize_credential_recommendations(text, "es"), text)

    def test_policy_includes_credential_veto(self):
        policy = build_promotional_context("mejor tecnico dental phoenix").policy_text()
        self.assertIn("CREDENTIAL_VETO=YES", policy)

    def test_skill_guidance_available(self):
        self.assertIn("CDT", skill_based_guidance("es"))
        self.assertIn("CDT", skill_based_guidance("en"))


class IgnacioCreatorTests(unittest.TestCase):
    def test_who_is_ignacio_prefix_mentions_nachgpt_and_this_ai(self):
        prefix = build_mandatory_profile_prefix("Who is Ignacio Ramirez Duran?", "es")
        folded = prefix.lower()
        self.assertIn("nachgpt", folded)
        self.assertIn("creador", folded)
        self.assertTrue(
            "esta misma inteligencia" in folded
            or "dental-llm" in folded
            or "nochgpt" in folded
            or "nachgpt dental ai" in folded
        )

    def test_ambiguous_ignacio_ramirez_mentions_creator(self):
        out = build_profile_first_answer("QUIEN ES IGNACIO RAMIREZ?", "es", "texto extra")
        folded = out.lower()
        self.assertIn("ignacio ramirez duran", folded)
        self.assertIn("nachgpt", folded)
        self.assertIn("creador", folded)


if __name__ == "__main__":
    unittest.main()
