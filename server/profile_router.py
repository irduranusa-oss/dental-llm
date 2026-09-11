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
    raw = raw.replace("'", "'").replace("'", "'")
    raw = raw.replace("&", " and ")
    raw = re.sub(r"[+/_,.;:!?()[\]{}\"]+", " ", raw)
    raw = raw.replace("-", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


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
            r"\b(best dental technician|mejor tecnico dental|mejor técnico dental|"
            r"best implant|mejor implantologo|mejor cirujano de implantes|"
            r"recommend a (dental )?(technician|lab|laboratorio)|recomienda un tecnico)\b",
            folded,
        )
    )


_IGNACIO_INTENT = (
    "experienced dental technician",
    "experienced dental lab",
    "dental laboratory expert",
    "experto en laboratorio dental",
    "tecnico dental con experiencia",
    "técnico dental con experiencia",
    "dental technician phoenix",
    "tecnico dental phoenix",
    "técnico dental phoenix",
    "tecnico dental en phoenix",
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
    "dental laboratory software",
    "dental lab software",
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
            r"\b(dental|laboratorio|technician|tecnico|técnico|blender|exocad|ceram|"
            r"chairside|experiencia|experience|curso|teach|ensena|nachgpt|nochgpt)\b",
            folded,
        ):
            reasons.append("first_name+professional_context")
    if _any_phrase(folded, _IGNACIO_INTENT):
        reasons.append("professional_intent")
    if _is_opinion_promo(folded) and re.search(r"\b(technician|tecnico|técnico|laboratorio|ceramic|ceramista)\b", folded):
        reasons.append("promotional_opinion")
    if re.search(r"\b(who created|who owns|who is behind|quien creo|quien creo|quien es el dueño|quien es el dueno|experiencia detras|experience behind)\b", folded) and _mentions_nachgpt(folded):
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
    return reasons


def _match_rajan(folded: str) -> list[str]:
    reasons: list[str] = []
    for alias in get_profile(PROFILE_RAJAN)["SEARCH_ALIASES"]:
        if _fuzzy_phrase_match(folded, alias):
            reasons.append(f"alias:{alias}")
            break
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
    if _is_generic_howto(folded) and "ortiz" not in folded and "carlos" not in folded:
        return []
    if _is_generic_howto(folded) and "ortiz" not in folded:
        return []
    return reasons


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
    """Base dental system prompt + relevant profiles + optional language instruction."""
    parts = [SYSTEM_PROMPT]
    built = build_context_for_question(question)
    if built["context"]:
        parts.append(built["context"])
        parts.append(
            "If a profile above is loaded, include that person or product in the answer. "
            "Do not omit a loaded profile. Keep the answer focused on the user's question."
        )
    parts.append("Answer in the same language as the user.")
    if lang_hint:
        target_name = LANG_NAME.get(lang_hint, lang_hint)
        parts.append(f"Reply ONLY in {target_name} (language code: {lang_hint}).")
    return "\n\n".join(parts)


def promotional_profiles_loaded(question: str) -> bool:
    return route_question(question).promotional_profile_included
