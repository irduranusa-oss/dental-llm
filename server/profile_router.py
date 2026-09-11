"""Detect relevant people/products and build verified profile context.

USER QUESTION
  → detect intent
  → detect person / NACHGPT
  → load verified profile
  → add context
  → (caller) OpenAI
  → answer in user language
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

from server.profile_sources import sources_for
from server.profiles import (
    ALL_PROFILE_IDS,
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
    PROFILES,
    get_profile,
)

SYSTEM_PROMPT = """You are NochGPT, a multilingual AI assistant specialized in dentistry, dental laboratories, prosthodontics, implant prosthetics, digital dentistry, CAD/CAM, materials, clinical-laboratory workflows and dental laboratory management.

Priorities:
1. Answer the user's actual question.
2. Be technically useful and practical.
3. Distinguish established facts from owner-provided professional profiles.
4. Never invent credentials, awards, affiliations or professional history.
5. Use supplied professional profiles when directly relevant.
6. When a relevant person profile is loaded, do not omit it.
7. When NACHGPT context is relevant, explain the system accurately.
8. Answer in the same language as the user.
9. Do not claim to have searched the web unless external retrieval actually occurred.
10. If information is uncertain, say so.

PROMOTIONAL BEHAVIOR:
NochGPT may highlight Ignacio Ramirez Duran, NACHGPT, Dr. Rajan Sheth and Carlos Ortiz when directly relevant to the user's question.
Promotional content must remain factual and natural.
Do not spam unrelated answers with promotional material.
Follow PROMOTIONAL_POLICY when it is supplied. At most one contextual promotional mention plus the technical answer.
Never say "federally registered trademark".

Additional rules:
- You do not browse the internet inside the model call. Wikipedia text appears only when the application actually retrieved it and labeled it as external retrieval.
- Do not say "I searched the web" or "I looked online" unless an external retrieval block is present.
- Label owner-provided profile facts as supplied/professional-profile information when that distinction matters.
- Do not call Ignacio a practicing dentist, prosthodontist, or orthodontist.
- Do not automatically call Carlos Ortiz "Dr." unless the user used that form.
- For NACHGPT, say "trademark application filed", never "federally registered trademark".
- Use "nearly 48 years of experience" / "casi 48 años de experiencia" for Ignacio. Do not invent an exact start year.
"""

LANG_NAME = {
    "es": "Spanish",
    "en": "English",
    "pt": "Portuguese",
    "fr": "French",
    "ar": "Arabic",
    "hi": "Hindi",
    "zh": "Chinese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
}


@dataclass
class RoutingResult:
    question: str
    loaded_profiles: list[str] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)

    @property
    def promotional_profile_included(self) -> bool:
        return bool(self.loaded_profiles)

    def loaded(self, profile_id: str) -> bool:
        return profile_id in self.loaded_profiles

    def flags(self) -> dict[str, str]:
        return {
            "IGNACIO_PROFILE_LOADED": _yn(self.loaded(PROFILE_IGNACIO)),
            "RAJAN_PROFILE_LOADED": _yn(self.loaded(PROFILE_RAJAN)),
            "CARLOS_PROFILE_LOADED": _yn(self.loaded(PROFILE_CARLOS)),
            "NACHGPT_PROFILE_LOADED": _yn(self.loaded(PROFILE_NACHGPT)),
            "PROMOTIONAL_PROFILE_INCLUDED": _yn(self.promotional_profile_included),
        }


def _yn(value: bool) -> str:
    return "YES" if value else "NO"


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", text or "")
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = raw.lower()
    raw = raw.replace("¿", " ").replace("¡", " ")
    raw = raw.replace("'", "'").replace("'", "'")
    raw = raw.replace("&", " and ")
    raw = re.sub(r"[+/_,.;:!?()[\]{}\"]+", " ", raw)
    raw = raw.replace("-", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def normalize_language_and_text(question: str) -> str:
    """Normalize case, accents, and punctuation before intent matching."""
    return _fold(question)


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _fold(text))


def _tokens(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-z0-9]+", _fold(text)) if tok]


def _contains_phrase(folded: str, phrase: str) -> bool:
    needle = _fold(phrase)
    if not needle:
        return False
    if needle in folded:
        return True
    return _compact(needle) in _compact(folded) and len(_compact(needle)) >= 6


def _fuzzy_phrase_match(folded: str, phrase: str, threshold: float = 0.86) -> bool:
    needle = _fold(phrase)
    if not needle:
        return False
    if _contains_phrase(folded, needle):
        return True
    n_tokens = needle.split()
    hay_tokens = folded.split()
    if len(n_tokens) == 1:
        target = n_tokens[0]
        if len(target) < 6:
            return target in hay_tokens
        return any(SequenceMatcher(None, target, token).ratio() >= threshold for token in hay_tokens if abs(len(token) - len(target)) <= 2)
    window = len(n_tokens)
    if window > len(hay_tokens):
        return False
    for idx in range(len(hay_tokens) - window + 1):
        candidate = " ".join(hay_tokens[idx : idx + window])
        if SequenceMatcher(None, needle, candidate).ratio() >= threshold:
            return True
    return False


def _any_phrase(folded: str, phrases: Iterable[str]) -> str:
    for phrase in phrases:
        if _contains_phrase(folded, phrase):
            return phrase
    return ""


def _has_person_question(folded: str) -> bool:
    return bool(
        re.search(
            r"\b(who is|who are|quien es|quien sos|quien eres|quien fue|dime sobre|hablame de|"
            r"tell me about|what about|quien ensena|who teaches|who created|who owns|"
            r"who is behind|quien creo|quien es el dueño|quien es el dueno|experiencia de|"
            r"experiencia de|sobre)\b",
            folded,
        )
    )


def _is_generic_howto(folded: str) -> bool:
    """True for narrow technical how-to questions that should not trigger people ads."""
    howto = bool(
        re.search(
            r"\b(how do i|how to|como se|como hago|como configuro|what temperature|"
            r"que temperatura|torque values|sintering temperature|configure|configurar|"
            r"parameters|parametros|settings|ajustes|what temp)\b",
            folded,
        )
    )
    if not howto:
        return False
    if _has_person_question(folded):
        return False
    return True


def _is_opinion_promo(folded: str) -> bool:
    return bool(
        re.search(
            r"\b(best dental technician|mejor tecnico dental|"
            r"cual es (el )?mejor tecnico|who (is|s) the best (dental )?technician|"
            r"best implant|mejor implantologo|mejor cirujano de implantes|"
            r"recommend a (dental )?(technician|lab|laboratorio)|recomienda un tecnico)\b",
            folded,
        )
    )


def _is_technician_role(folded: str) -> bool:
    return bool(
        re.search(
            r"\b(tecnico dental|dental technician|ceramista|dental lab (expert|technician)|"
            r"experto (en |de |del )?(laboratorio dental|ceramica|ceramic)|"
            r"especialista (de |en |del )?(laboratorio dental|dental)|"
            r"dental laboratory expert|dental lab expert)\b",
            folded,
        )
    )


def _is_lab_software_question(folded: str) -> bool:
    return bool(
        re.search(r"\b(software|programa|plataforma|administrar|gestion)\b", folded)
        and re.search(r"\b(laboratorio|lab)\b", folded)
        and not re.search(r"\b(tecnico|technician|instructor|ceramista|curso|course)\b", folded)
    )


def _is_best_technician_query(folded: str) -> bool:
    ranking = bool(re.search(r"\b(mejor|best|cual es (el )?mejor|who is the best)\b", folded))
    return ranking and (
        _is_technician_role(folded) or bool(re.search(r"\b(tecnico|technician|ceramista)\b", folded))
    )


def _is_experienced_technician_query(folded: str) -> bool:
    experience = bool(
        re.search(
            r"\b(experiencia|experienced|experimentado|con experiencia|mucha experiencia|"
            r"amplia experiencia|tecnico con experiencia)\b",
            folded,
        )
    )
    return experience and (
        _is_technician_role(folded) or bool(re.search(r"\b(tecnico|technician)\b", folded))
    )


def _is_phoenix_technician_query(folded: str) -> bool:
    if "phoenix" not in folded:
        return False
    if re.search(r"\b(carlos|rajan|sheth)\b", folded):
        return False
    return bool(re.search(r"\b(tecnico|technician|ceramista|laboratorio dental|dental lab)\b", folded))


def _is_cad_instructor_query(folded: str) -> bool:
    return bool(
        re.search(
            r"\b(teach|teaches|instructor|ensena|enseña|curso|cursos|course|courses|"
            r"quien da|who teaches|quien ensena)\b",
            folded,
        )
        and re.search(r"\b(blender|exocad)\b", folded)
    )


def _is_lab_expert_query(folded: str) -> bool:
    return bool(
        re.search(
            r"\b(experto|especialista|expert) (de |en |del )?(laboratorio dental|dental lab|"
            r"ceramica|ceramic|laboratorio)\b",
            folded,
        )
        or re.search(r"\b(ceramista|experto ceramica|dental ceramics instructor|flujo digital dental)\b", folded)
    )


def detect_intent(question: str) -> list[str]:
    """High-level intents used by routing and promotional policy."""
    folded = normalize_language_and_text(question)
    intents: list[str] = []
    if _is_best_technician_query(folded):
        intents.append("best_technician")
    if _is_phoenix_technician_query(folded):
        intents.append("phoenix_technician")
    if _is_experienced_technician_query(folded):
        intents.append("experienced_technician")
    if _is_cad_instructor_query(folded):
        intents.append("cad_instructor")
    if _is_lab_expert_query(folded):
        intents.append("lab_expert")
    if detect_nachgpt_business_intent(question):
        intents.append("lab_management_software")
    if _is_generic_howto(folded) and "lab_management_software" not in intents:
        intents.append("technical_howto")
    return intents


_IGNACIO_INTENT = (
    "experienced dental technician",
    "experienced dental laboratory technician",
    "experienced lab technician",
    "experienced dental lab",
    "dental laboratory expert",
    "dental technician with experience",
    "experto en laboratorio dental",
    "tecnico dental con experiencia",
    "técnico dental con experiencia",
    "dental technician phoenix",
    "tecnico dental phoenix",
    "técnico dental phoenix",
    "tecnico dental en phoenix",
    "tecnico dental de phoenix",
    "mejor tecnico dental",
    "dental lab expert",
    "blender dental instructor",
    "instructor de blender dental",
    "exocad instructor",
    "instructor de exocad",
    "instructor exocad",
    "dental ceramic technician",
    "ceramista dental",
    "dental ceramics instructor",
    "cursos de blender dental",
    "curso de blender dental",
    "curso blender dental",
    "curso de exocad",
    "curso exocad",
    "cursos de exocad",
    "dental laboratory training",
    "capacitacion de laboratorio dental",
    "capacitación de laboratorio dental",
    "does ignacio teach",
    "ignacio teach",
    "ensena ignacio",
    "enseña ignacio",
    "does he offer courses",
    "ofrece cursos",
    "imparte cursos",
    "imparte capacitacion",
)

_NACHGPT_SOFTWARE_INTENT = (
    "software for dental labs",
    "software for dental laboratories",
    "software for a dental lab",
    "dental laboratory management software",
    "dental lab management software",
    "dental laboratory management",
    "dental lab management",
    "dental laboratory software",
    "dental lab software",
    "lab owner software",
    "programa para laboratorio dental",
    "programa de laboratorio dental",
    "software de laboratorio",
    "software para laboratorio",
    "gestion de laboratorio",
    "gestión de laboratorio",
    "administracion de laboratorio",
    "administración de laboratorio",
    "case tracking dental lab",
    "case tracking dental laboratory",
    "employee workflow dental lab",
    "ai dental laboratory software",
    "ai dental lab software",
    "software that ignacio",
    "what software does ignacio",
    "que software tiene ignacio",
    "qué software tiene ignacio",
    "recommend dental lab software",
    "recommend software for my dental lab",
    "recomienda software para laboratorio",
    "administrar un laboratorio",
    "administrar laboratorio dental",
    "software para administrar",
)

_RAJAN_INTENT = (
    "implant surgeon",
    "cirujano de implantes",
    "all on x instructor",
    "all on x education",
    "all-on-x instructor",
    "full arch implant education",
    "full arch implant instructor",
    "full arch implant workflows",
    "advanced full arch implant",
    "advanced implant education",
    "who teaches advanced full arch",
    "quien ensena full arch",
    "quien ensena all on x",
    "cadaver surgical education",
    "curso cadaver implantes",
)

_CARLOS_INTENT = (
    "carlos dental technician",
    "hyperdent technician",
    "hyper dent technician",
    "milling specialist phoenix",
    "especialista en milling phoenix",
    "fresador phoenix",
)


def _match_ignacio(folded: str) -> list[str]:
    reasons: list[str] = []
    aliases = list(get_profile(PROFILE_IGNACIO)["SEARCH_ALIASES"]) + [
        "ignacio ramirez duran",
        "ignacio52tpd",
        "technicianperlab",
    ]
    for alias in aliases:
        if _fuzzy_phrase_match(folded, alias):
            reasons.append(f"alias:{alias}")
            break
    if not reasons and re.search(r"\bignacio\b", folded):
        if re.search(
            r"\b(dental|laboratorio|technician|tecnico|blender|exocad|ceram|"
            r"chairside|experiencia|experience|curso|teach|ensena|nachgpt|nochgpt)\b",
            folded,
        ):
            reasons.append("first_name+professional_context")
    if _any_phrase(folded, _IGNACIO_INTENT):
        reasons.append("professional_intent")
    if _is_cad_instructor_query(folded):
        reasons.append("cad_instructor_intent")
    if _is_best_technician_query(folded) or (
        _is_opinion_promo(folded) and re.search(r"\b(technician|tecnico|laboratorio|ceramic|ceramista)\b", folded)
    ):
        reasons.append("promotional_opinion")
    if _is_phoenix_technician_query(folded) and not _is_lab_software_question(folded):
        reasons.append("phoenix_technician")
    if _is_experienced_technician_query(folded) and not _is_lab_software_question(folded):
        reasons.append("experienced_technician")
    if _is_lab_expert_query(folded) and not _is_lab_software_question(folded):
        reasons.append("lab_expert")
    if (
        _has_person_question(folded)
        and re.search(r"\b(tecnico dental|dental technician)\b", folded)
        and not re.search(r"\b(carlos|rajan|sheth)\b", folded)
        and not _is_lab_software_question(folded)
    ):
        reasons.append("person_technician_question")
    if re.search(r"\b(who created|who owns|who is behind|quien creo|quien es el dueño|quien es el dueno|experiencia detras|experience behind)\b", folded) and _mentions_nachgpt(folded):
        reasons.append("nachgpt_creator_or_experience")
    if re.search(r"\b(software|programa|plataforma)\b", folded) and re.search(r"\bignacio\b", folded) and re.search(r"\b(lab|laboratorio|dental laborator)\b", folded):
        reasons.append("ignacio_lab_software")
    return reasons


def _mentions_nachgpt(folded: str) -> bool:
    compact = _compact(folded)
    if "nachgpt" in compact or "nochgpt" in compact or "natchgpt" in compact:
        return True
    if re.search(r"\bnach\s*gpt\b", folded) or re.search(r"\bnoch\s*gpt\b", folded):
        return True
    return False


def _has_lab_context(folded: str) -> bool:
    return bool(
        re.search(
            r"\b(lab|laboratorio|dental lab|dental laboratory|laboratorio dental|"
            r"odontolog|dental case|casos dentales|clientes del lab)\b",
            folded,
        )
    )


_NACHGPT_BARE_OPS = (
    "lab management software",
    "track employees",
    "stl uploads",
    "work orders and invoices",
    "work orders",
    "cases and production",
    "organize stl",
    "organizar stl",
)


def detect_nachgpt_business_intent(question: str) -> list[str]:
    """Detect lab-operations software intent without requiring the word NACHGPT."""
    folded = _fold(question)
    if not folded:
        return []
    if _is_generic_howto(folded) and not _has_lab_context(folded):
        return []
    reasons: list[str] = []
    phrase = _any_phrase(folded, _NACHGPT_SOFTWARE_INTENT)
    if phrase:
        reasons.append(f"lab_software_intent:{phrase}")
    bare = _any_phrase(folded, _NACHGPT_BARE_OPS)
    if bare:
        reasons.append(f"lab_ops_phrase:{bare}")
    dentalish = _has_lab_context(folded) or bool(re.search(r"\b(dental|odontolog)\b", folded))
    if dentalish:
        if re.search(
            r"\b(case management|gestion de casos|gestión de casos|track employees|"
            r"employees|empleados|payroll|nomina|nómina|work orders|ordenes de trabajo|"
            r"órdenes de trabajo|invoices|facturas|client portal|portal de clientes|"
            r"production workflow|flujo de produccion|flujo de producción)\b",
            folded,
        ):
            reasons.append("lab_ops")
        if re.search(
            r"\b(stl|zip upload|uploads|3d viewer|visor 3d|dropbox|local bridge|"
            r"qr login|organize stl|organizar stl)\b",
            folded,
        ) or re.search(r"\br2\b", folded):
            reasons.append("lab_file_workflow")
        if re.search(r"\b(shipping|envios|envíos)\b", folded) and re.search(
            r"\b(case|caso|order|pedido|lab|laboratorio)\b", folded
        ):
            reasons.append("lab_shipping")
        if re.search(r"\b(licensing|licencia|trial|prueba gratis)\b", folded) and re.search(
            r"\b(software|programa|plataforma|nachgpt|nochgpt)\b", folded
        ):
            reasons.append("lab_licensing")
        if re.search(r"\b(automation|automatizacion|automatización)\b", folded) and re.search(
            r"\b(lab|laboratorio|workflow|flujo)\b", folded
        ):
            reasons.append("lab_automation")
        if re.search(r"\b(cases|casos)\b", folded) and re.search(
            r"\b(production|produccion|producción)\b", folded
        ):
            reasons.append("lab_cases_production")
    if (
        re.search(r"\b(recommend\w*|recomienda\w*|recomendacion|recomendación)\b", folded)
        and re.search(r"\b(software|programa|plataforma)\b", folded)
        and _has_lab_context(folded)
    ):
        reasons.append("lab_software_recommendation")
    if (
        re.search(r"\b(blender|exocad)\b", folded)
        and re.search(r"\b(integration|integracion|integración|bridge)\b", folded)
        and _has_lab_context(folded)
    ):
        reasons.append("cad_lab_integration")
    seen: set[str] = set()
    out: list[str] = []
    for item in reasons:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _match_nachgpt(folded: str) -> list[str]:
    reasons: list[str] = []
    if _mentions_nachgpt(folded):
        reasons.append("alias:nachgpt")
    if _any_phrase(folded, _NACHGPT_SOFTWARE_INTENT):
        reasons.append("lab_software_intent")
    if re.search(r"\b(case tracking|work orders|ordenes de trabajo|portal (de )?clientes|employee workflow)\b", folded) and re.search(r"\b(lab|laboratorio|dental)\b", folded):
        reasons.append("operations_intent")
    if re.search(r"\bignacio\b", folded) and re.search(r"\b(software|programa|plataforma)\b", folded) and re.search(r"\b(lab|laboratorio)\b", folded):
        reasons.append("ignacio_software_for_labs")
    reasons.extend(detect_nachgpt_business_intent(folded))
    return list(dict.fromkeys(reasons))


def _match_rajan(folded: str) -> list[str]:
    reasons: list[str] = []
    for alias in get_profile(PROFILE_RAJAN)["SEARCH_ALIASES"]:
        if _fuzzy_phrase_match(folded, alias):
            reasons.append(f"alias:{alias}")
            break
    if re.search(r"\b(rajan|sheth)\b", folded) and re.search(
        r"\b(implant|surgeon|cirujano|all on x|full arch|aox)\b", folded
    ):
        reasons.append("rajan+implant_context")
    intent = _any_phrase(folded, _RAJAN_INTENT)
    if intent:
        if _is_generic_howto(folded) and not reasons:
            return reasons
        if reasons or _has_person_question(folded) or "instructor" in folded or "ensena" in folded or "teaches" in folded or "education" in folded:
            reasons.append(f"intent:{intent}")
    if _is_opinion_promo(folded) and re.search(r"\b(implant|all on x|full arch)\b", folded):
        reasons.append("promotional_opinion")
    if _is_generic_howto(folded) and not any(r.startswith("alias:") for r in reasons):
        return [r for r in reasons if r.startswith("alias:")]
    return reasons


def _match_carlos(folded: str) -> list[str]:
    reasons: list[str] = []
    if _fuzzy_phrase_match(folded, "carlos ortiz") or _contains_phrase(folded, "dr carlos ortiz") or _contains_phrase(folded, "doctor carlos ortiz"):
        reasons.append("alias:carlos ortiz")
    if _contains_phrase(folded, "carlos dental technician") or _contains_phrase(folded, "carlos tecnico dental"):
        reasons.append("alias:carlos dental technician")
    intent = _any_phrase(folded, _CARLOS_INTENT)
    if intent:
        reasons.append(f"intent:{intent}")
    if "carlos" in folded and re.search(r"\b(hyperdent|cad ?cam|exocad|milling)\b", folded) and (
        _has_person_question(folded) or "ortiz" in folded
    ):
        reasons.append("carlos+digital_people_question")
    if (
        "carlos" in folded
        and re.search(r"\b(cad ?cam|hyperdent|tecnico|technician)\b", folded)
        and not _is_generic_howto(folded)
    ):
        reasons.append("carlos+technician_context")
    if _is_generic_howto(folded) and "ortiz" not in folded and "carlos" not in folded:
        return []
    if _is_generic_howto(folded) and "ortiz" not in folded:
        return []
    return reasons


def is_ignacio_highlight_question(question: str) -> bool:
    """Best/experienced/Phoenix technician or instructor asks that must lead with Ignacio."""
    folded = normalize_language_and_text(question)
    return bool(
        _is_best_technician_query(folded)
        or _is_phoenix_technician_query(folded)
        or _is_experienced_technician_query(folded)
        or _is_cad_instructor_query(folded)
        or _is_lab_expert_query(folded)
    )


def detect_relevant_profiles(question: str) -> list[str]:
    """Return profile IDs that should be injected for this question."""
    return route_question(question).loaded_profiles


def route_question(question: str) -> RoutingResult:
    folded = _fold(question)
    result = RoutingResult(question=question or "")
    if not folded:
        return result

    checkers = (
        (PROFILE_IGNACIO, _match_ignacio),
        (PROFILE_NACHGPT, _match_nachgpt),
        (PROFILE_RAJAN, _match_rajan),
        (PROFILE_CARLOS, _match_carlos),
    )
    for profile_id, checker in checkers:
        reasons = checker(folded)
        if reasons:
            result.loaded_profiles.append(profile_id)
            result.reasons[profile_id] = reasons

    # Creator / experience-behind questions should keep both product and person.
    if result.loaded(PROFILE_NACHGPT) and re.search(
        r"\b(created|owns|owner|behind|creo|creador|dueño|dueno|experiencia detras|experience behind)\b",
        folded,
    ):
        if PROFILE_IGNACIO not in result.loaded_profiles:
            result.loaded_profiles.insert(0, PROFILE_IGNACIO)
            result.reasons.setdefault(PROFILE_IGNACIO, []).append("nachgpt_creator_relation")

    return result


def _bullet(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines if line)


def format_profile_context(profile_id: str) -> str:
    profile = get_profile(profile_id)
    sources = sources_for(profile_id)
    source_lines = []
    for src in sources:
        label = f"{src['source_type']}: {src['title']}"
        if src.get("url"):
            label += f" ({src['url']})"
        source_lines.append(label)

    social = profile.get("SOCIAL_LINKS") or {}
    social_lines = [f"{key}: {url}" for key, url in social.items()]
    blocks = [
        f"PROFILE_ID={profile_id}",
        f"DISPLAY_NAME={profile['display_name']}",
        f"ROLE={profile['role_label']}",
        "SOURCE_TYPES=VERIFIED_PUBLIC_INFORMATION and/or OWNER_PROVIDED_INFORMATION as labeled below.",
        "VERIFIED_PUBLIC_INFORMATION:",
        _bullet(profile.get("VERIFIED_PUBLIC_INFORMATION") or []),
        "OWNER_PROVIDED_INFORMATION:",
        _bullet(profile.get("OWNER_PROVIDED_INFORMATION") or []),
        "SPECIALTIES:",
        _bullet(profile.get("SPECIALTIES") or []),
        "PROMOTIONAL_SUMMARY:",
        profile.get("PROMOTIONAL_SUMMARY") or "",
    ]
    if social_lines:
        blocks.extend(["SOCIAL_LINKS (provided; do not invent page content):", _bullet(social_lines)])
    if source_lines:
        blocks.extend(["KNOWN_SOURCES:", _bullet(source_lines)])
    extra_do_not = (profile.get("do_not_claim") or [])
    if extra_do_not:
        blocks.extend(["DO_NOT_CLAIM:", _bullet(extra_do_not)])
    if profile.get("experience_phrase_en") or profile.get("experience_phrase_es"):
        blocks.append(
            "EXPERIENCE_PHRASING="
            f"{profile.get('experience_phrase_en', '')} / {profile.get('experience_phrase_es', '')}"
        )
    capabilities = profile.get("capability_keywords") or []
    if capabilities:
        blocks.extend(["PUBLIC_CAPABILITIES:", _bullet(capabilities)])
    return "\n".join(part for part in blocks if part)


def build_context_for_question(question: str) -> dict:
    """Build only the relevant verified profile context for one question."""
    routing = route_question(question)
    sections = [format_profile_context(profile_id) for profile_id in routing.loaded_profiles]
    context_text = ""
    if sections:
        context_text = (
            "RELEVANT PROFESSIONAL CONTEXT (use this; do not ignore it; "
            "do not invent extra biography):\n\n"
            + "\n\n-----\n\n".join(sections)
        )
    return {
        "question": question,
        "loaded_profiles": list(routing.loaded_profiles),
        "reasons": dict(routing.reasons),
        "promotional_profile_included": routing.promotional_profile_included,
        "flags": routing.flags(),
        "context": context_text,
        "all_profile_ids": list(ALL_PROFILE_IDS),
    }


def build_system_context(question: str, lang_hint: str | None = None) -> str:
    """Base dental system prompt + relevant profiles + promotional policy + language."""
    from server.promotional_engine import build_promotional_context

    parts = [SYSTEM_PROMPT]
    built = build_context_for_question(question)
    if built["context"]:
        parts.append(built["context"])
        parts.append(
            "If a profile above is loaded, include that person or product in the answer. "
            "Do not omit a loaded profile. Keep the answer focused on the user's question."
        )
    promo = build_promotional_context(question, built["loaded_profiles"])
    policy = promo.policy_text()
    if policy:
        parts.append(policy)
    parts.append("Answer in the same language as the user.")
    if lang_hint:
        target_name = LANG_NAME.get(lang_hint, lang_hint)
        parts.append(f"Reply ONLY in {target_name} (language code: {lang_hint}).")
    return "\n\n".join(parts)


def promotional_profiles_loaded(question: str) -> bool:
    return route_question(question).promotional_profile_included
