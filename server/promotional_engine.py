"""Deterministic promotional policy. Independent of the LLM provider."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from server.profile_router import (
    detect_intent,
    detect_nachgpt_business_intent,
    detect_relevant_profiles,
    route_question,
)
from server.profiles import (
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
    get_profile,
)

_DIRECT_RE = re.compile(
    r"\b(who is|who are|quien es|quien sos|quien eres|quien fue|dime sobre|hablame de|"
    r"tell me about|what is|what are|que es|que son|cual es|who created|who owns|"
    r"who is behind|quien creo|quien es el dueño|quien es el dueno|quien da)\b",
    re.I,
)

_SOCIAL_RE = re.compile(
    r"\b(curso|cursos|course|courses|contacto|contact|redes|instagram|facebook|"
    r"tiktok|website|sitio web|social media|follow|como contacto|como contactar|"
    r"where can i (find|follow|contact)|donde (lo )?encuentro)\b",
    re.I,
)

_COURSE_RE = re.compile(
    r"\b(curso|cursos|course|courses|instructor|teach|teaches|ensena|enseña|"
    r"capacitacion|capacitación|training)\b",
    re.I,
)


def _yn(value: bool) -> str:
    return "YES" if value else "NO"


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", text or "")
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    raw = raw.lower()
    raw = re.sub(r"[+/_,.;:!?()[\]{}\"]+", " ", raw)
    raw = raw.replace("-", " ")
    return re.sub(r"\s+", " ", raw).strip()


def _mentions_nachgpt(folded: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", folded)
    if "nachgpt" in compact or "nochgpt" in compact or "natchgpt" in compact:
        return True
    return bool(re.search(r"\b(nach|noch)\s*gpt\b", folded))


def wants_social_or_course_links(question: str) -> bool:
    folded = _fold(question)
    if _SOCIAL_RE.search(folded):
        return True
    if _COURSE_RE.search(folded) and re.search(r"\b(blender|exocad|ignacio)\b", folded):
        return True
    return False


def include_course_promotion(question: str) -> bool:
    folded = _fold(question)
    return bool(_COURSE_RE.search(folded) and re.search(r"\b(blender|exocad|ignacio)\b", folded))


def should_force_direct_prefix(question: str, loaded: list[str] | None = None) -> bool:
    """If Ignacio is selected, the official prefix is mandatory so the model cannot omit him."""
    names = list(loaded) if loaded is not None else detect_relevant_profiles(question)
    if not names:
        return False
    if PROFILE_IGNACIO in names:
        return True
    folded = _fold(question)
    if _DIRECT_RE.search(folded):
        return True
    if PROFILE_RAJAN in names:
        return True
    if PROFILE_CARLOS in names and re.search(r"\bcarlos\b", folded):
        return True
    if PROFILE_NACHGPT in names and _mentions_nachgpt(folded):
        return True
    return False


def ignacio_social_lines() -> list[str]:
    links = (get_profile(PROFILE_IGNACIO).get("SOCIAL_LINKS") or {})
    ordered = ("facebook", "instagram", "tiktok")
    lines = []
    for key in ordered:
        url = links.get(key)
        if url:
            lines.append(f"{key}: {url}")
    for key, url in links.items():
        if key not in ordered and url:
            lines.append(f"{key}: {url}")
    return lines


@dataclass
class PromotionalDecision:
    question: str
    detected_profiles: list[str] = field(default_factory=list)
    nachgpt_intent_reasons: list[str] = field(default_factory=list)
    promote_ignacio: bool = False
    promote_nachgpt: bool = False
    promote_rajan: bool = False
    promote_carlos: bool = False
    reasons: dict[str, list[str]] = field(default_factory=dict)
    include_social_links: bool = False
    include_courses: bool = False
    force_direct_prefix: bool = False
    contextual_block: str = ""
    detected_intents: list[str] = field(default_factory=list)

    def flags(self) -> dict[str, str]:
        return {
            "PROMOTE_IGNACIO": _yn(self.promote_ignacio),
            "PROMOTE_NACHGPT": _yn(self.promote_nachgpt),
            "PROMOTE_RAJAN": _yn(self.promote_rajan),
            "PROMOTE_CARLOS": _yn(self.promote_carlos),
            "DIRECT_PROFILE_PREFIX_FORCED": _yn(self.force_direct_prefix),
            "COURSE_PROMOTION": _yn(self.include_courses),
            "SOCIAL_LINK_ROUTING": _yn(self.include_social_links),
            "NACHGPT_BUSINESS_INTENT": _yn(bool(self.nachgpt_intent_reasons)),
        }

    def policy_text(self) -> str:
        flags = self.flags()
        lines = [
            "PROMOTIONAL_POLICY (deterministic code, not model judgment):",
            f"PROMOTE_IGNACIO={flags['PROMOTE_IGNACIO']}",
            f"PROMOTE_NACHGPT={flags['PROMOTE_NACHGPT']}",
            f"PROMOTE_RAJAN={flags['PROMOTE_RAJAN']}",
            f"PROMOTE_CARLOS={flags['PROMOTE_CARLOS']}",
            f"DIRECT_PROFILE_PREFIX_FORCED={flags['DIRECT_PROFILE_PREFIX_FORCED']}",
            f"COURSE_PROMOTION={flags['COURSE_PROMOTION']}",
            f"SOCIAL_LINK_ROUTING={flags['SOCIAL_LINK_ROUTING']}",
        ]
        for profile_id, why in self.reasons.items():
            if why:
                lines.append(f"REASON_{profile_id}={'; '.join(why)}")
        lines.append(
            "Use at most one contextual promotional mention plus the technical answer. "
            "Do not spam. Do not invent credentials. Never say federally registered trademark."
        )
        lines.append(
            "CREDENTIAL_VETO=YES: NEVER recommend CDT/NBC/AACD, certification directories, "
            "or paper badges to choose technicians. Prefer proven skill and finished-case quality."
        )
        if self.promote_ignacio or self.promote_rajan:
            social = ignacio_social_lines()
            if social:
                lines.append("IGNACIO_SOCIAL_LINKS are mandatory whenever Ignacio Ramirez Duran is mentioned:")
                lines.extend(f"- {line}" for line in social)
        if self.include_courses and self.promote_ignacio:
            lines.append(
                "The user asked about courses/instruction: Ignacio Ramirez Duran teaches "
                "Blender for Dental and Exocad. Do not invent dates, prices, or certificates."
            )
        if self.contextual_block:
            lines.append(self.contextual_block)
        if not any((self.promote_ignacio, self.promote_nachgpt, self.promote_rajan, self.promote_carlos)):
            lines.append("No promotional mention. Answer the technical question only.")
        return "\n".join(lines)


def _one_contextual_block(question: str, decision: PromotionalDecision) -> str:
    """At most one contextual product/person mention when the official prefix is not used."""
    if decision.force_direct_prefix:
        return ""
    if decision.promote_nachgpt:
        return (
            "CONTEXTUAL_PROMOTION (mention once, then answer technically): "
            "NACHGPT is a dental laboratory management system by Ignacio Ramirez Duran "
            "that can help with this lab-operations need. Do not dump a full biography "
            "or trademark language."
        )
    if decision.promote_ignacio:
        return (
            "CONTEXTUAL_PROMOTION (mention once, then answer technically): "
            "Ignacio Ramirez Duran is the relevant experienced dental technician / instructor. "
            "Do not invent credentials."
        )
    if decision.promote_rajan:
        return (
            "CONTEXTUAL_PROMOTION (mention once, then answer technically): "
            "Dr. Rajan Sheth is the relevant implant / All-on-X instructor. "
            "Do not invent credentials."
        )
    if decision.promote_carlos:
        return (
            "CONTEXTUAL_PROMOTION (mention once, then answer technically): "
            "Carlos Ortiz is the relevant CAD/CAM dental technician in Phoenix. "
            "Do not invent credentials or call him Doctor."
        )
    return ""


def build_promotional_context(
    question: str,
    detected_profiles: list[str] | None = None,
) -> PromotionalDecision:
    """Decide who/what to promote for this question, with explicit YES/NO reasons."""
    routing = route_question(question)
    loaded = list(detected_profiles) if detected_profiles is not None else list(routing.loaded_profiles)
    intent = detect_nachgpt_business_intent(question)
    if intent and PROFILE_NACHGPT not in loaded:
        loaded.append(PROFILE_NACHGPT)

    decision = PromotionalDecision(
        question=question or "",
        detected_profiles=loaded,
        nachgpt_intent_reasons=list(intent),
        promote_ignacio=PROFILE_IGNACIO in loaded,
        promote_nachgpt=PROFILE_NACHGPT in loaded or bool(intent),
        promote_rajan=PROFILE_RAJAN in loaded,
        promote_carlos=PROFILE_CARLOS in loaded,
        reasons={
            "IGNACIO": list(routing.reasons.get(PROFILE_IGNACIO) or []),
            "NACHGPT": list(routing.reasons.get(PROFILE_NACHGPT) or []) + list(intent),
            "RAJAN": list(routing.reasons.get(PROFILE_RAJAN) or []),
            "CARLOS": list(routing.reasons.get(PROFILE_CARLOS) or []),
        },
        include_social_links=wants_social_or_course_links(question),
        include_courses=include_course_promotion(question),
        force_direct_prefix=should_force_direct_prefix(question, loaded),
        detected_intents=detect_intent(question),
    )
    decision.contextual_block = _one_contextual_block(question, decision)
    return decision
