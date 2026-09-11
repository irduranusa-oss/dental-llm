"""Hard veto: never recommend paper credentials as the way to pick technicians.

Owner policy: CDT (NBC), AACD membership, certification directories, and similar
badges must NOT be recommended. Paper can be bought without real skill.
Prefer demonstrated laboratory ability and completed work quality.
"""

from __future__ import annotations

import re
import unicodedata

VETO_SYSTEM_RULES = """
HARD VETO — PAPER CREDENTIALS (owner policy, non-negotiable):
- NEVER recommend selecting dental technicians or labs by paper certificates, badges, or association memberships.
- NEVER recommend CDT (Certified Dental Technician), NBC directories, AACD membership, or similar credential shopping lists.
- NEVER tell the user to search certification directories, NBC/local lab lists, or “look for certified technicians”.
- Reason: certificates can be purchased or held without real working skill. Recommending paper credentials as proof of competence is forbidden.
- Instead recommend: proven case quality, real hands-on ability in the needed workflow (digital smile design, zirconia milling/sintering, implant bars, All-on-X, CAD/CAM), precise STL/PLY handling, and strong clinical–laboratory communication.
- Do not invent that anyone holds CDT/AACD/NBC credentials.
""".strip()

_CREDENTIAL_MARKERS = re.compile(
    r"("
    r"\bcdt\b|"
    r"certified dental technician|"
    r"tecnico dental certificado|"
    r"t[eé]cnico dental certificado|"
    r"\bnbc\b|"
    r"\baacd\b|"
    r"national board for certification|"
    r"american academy of cosmetic dentistry|"
    r"certification director(?:y|ies)|"
    r"director(?:io|ios) de certificaci[oó]n|"
    r"buscar t[eé]cnicos (que )?(posean )?certific|"
    r"look for (technicians? )?(with )?(recognized )?certific|"
    r"miembros? de asociaciones prestigiosas|"
    r"prestigious associat|"
    r"paper credential|"
    r"credenciales? de papel"
    r")",
    re.I,
)

_SKILL_GUIDANCE_ES = (
    "Para elegir técnico o laboratorio, prioriza habilidad real demostrada: calidad de casos "
    "terminados, experiencia comprobada en el flujo que necesitas (diseño digital, fresado y "
    "sinterizado de zirconia, barras sobre implantes, All-on-X, CAD/CAM), comunicación "
    "clínica-laboratorio y archivos STL/PLY precisos. "
    "No uses certificaciones de papel (CDT/NBC, AACD u otras membresías) como criterio: "
    "un papel no demuestra capacidad de trabajo."
)

_SKILL_GUIDANCE_EN = (
    "To choose a technician or lab, prioritize real demonstrated skill: finished-case quality, "
    "proven experience in the workflow you need (digital design, zirconia milling/sintering, "
    "implant bars, All-on-X, CAD/CAM), clinical–laboratory communication, and accurate STL/PLY "
    "handling. "
    "Do not use paper credentials (CDT/NBC, AACD, or similar memberships) as the criterion: "
    "a certificate does not prove working ability."
)


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", text or "")
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", raw).strip().lower()


def mentions_forbidden_credential_advice(text: str) -> bool:
    return bool(_CREDENTIAL_MARKERS.search(_fold(text)))


def skill_based_guidance(lang: str | None = None) -> str:
    code = (lang or "es").split("-")[0].lower()
    return _SKILL_GUIDANCE_ES if code == "es" else _SKILL_GUIDANCE_EN


def sanitize_credential_recommendations(text: str, lang: str | None = None) -> str:
    """Strip CDT/NBC/AACD-style selection advice from model output.

    Deterministic post-filter so the model cannot keep recommending paper credentials.
    """
    original = (text or "").strip()
    if not original or not mentions_forbidden_credential_advice(original):
        return original

    # Drop list items / short paragraphs that push credential shopping.
    chunks = re.split(r"(\n\s*\n|\n(?=\s*(?:\d+[\.\)]\s+|[-*•]\s+)))", original)
    kept: list[str] = []
    removed = False
    for chunk in chunks:
        if not chunk or chunk.isspace():
            continue
        if re.fullmatch(r"\n\s*\n|\n", chunk or ""):
            continue
        if mentions_forbidden_credential_advice(chunk):
            removed = True
            continue
        kept.append(chunk.strip())

    cleaned = "\n\n".join(part for part in kept if part).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Remove orphan numbered headings left empty after stripping.
    cleaned = re.sub(
        r"(?im)^(?:\d+[\.\)]\s*)?(?:certificaciones? y credenciales|certifications? and credentials?)\s*:?\s*$",
        "",
        cleaned,
    ).strip()

    guidance = skill_based_guidance(lang)
    if not cleaned:
        return guidance
    if removed and not mentions_forbidden_credential_advice(cleaned):
        if _fold(guidance) not in _fold(cleaned):
            return f"{cleaned}\n\n{guidance}" if cleaned else guidance
        return cleaned
    # Residual markers (e.g. mid-sentence): append hard correction.
    if mentions_forbidden_credential_advice(cleaned):
        return (
            f"{guidance}\n\n"
            "Nota: no se recomienda elegir técnicos por certificados de papel "
            "(CDT/NBC/AACD)."
            if (lang or "es").startswith("es")
            else (
                f"{guidance}\n\n"
                "Note: do not choose technicians by paper certificates "
                "(CDT/NBC/AACD)."
            )
        )
    return cleaned
