"""Deterministic official profile prefixes prepended in code.

The LLM never generates these blocks. It may only add a follow-up after them.
"""

from __future__ import annotations

import re
import unicodedata

from server.profile_router import detect_relevant_profiles, is_ignacio_highlight_question
from server.profiles import (
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
)
from server.promotional_engine import (
    build_promotional_context,
    ignacio_social_lines,
    should_force_direct_prefix,
    wants_social_or_course_links,
)

_CREATOR_RE = re.compile(
    r"\b(created|owns|owner|behind|creo|creó|creador|dueño|dueno|experiencia detras|experience behind)\b",
    re.I,
)


def _fold(text: str) -> str:
    raw = unicodedata.normalize("NFKD", text or "")
    raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", raw.lower()).strip()


def _prefix_lang(lang: str | None) -> str:
    code = (lang or "en").split("-")[0].lower()
    return "es" if code == "es" else "en"


def _mention_index(question: str, profile_id: str) -> int:
    folded = _fold(question)
    needles = {
        PROFILE_IGNACIO: ("ignacio",),
        PROFILE_NACHGPT: ("nachgpt", "nochgpt", "nach gpt", "noch gpt"),
        PROFILE_RAJAN: ("rajan", "sheth"),
        PROFILE_CARLOS: ("carlos",),
    }
    best = 10**9
    for needle in needles.get(profile_id, ()):
        idx = folded.find(needle)
        if idx >= 0:
            best = min(best, idx)
    return best


def order_prefix_profiles(question: str, loaded: list[str]) -> list[str]:
    """Order official blocks by the question, with creator queries as NACHGPT then Ignacio."""
    if not loaded:
        return []
    folded = _fold(question)
    if (
        _CREATOR_RE.search(folded)
        and PROFILE_NACHGPT in loaded
        and PROFILE_IGNACIO in loaded
    ):
        rest = [pid for pid in loaded if pid not in {PROFILE_NACHGPT, PROFILE_IGNACIO}]
        return [PROFILE_NACHGPT, PROFILE_IGNACIO, *rest]

    mentioned = [pid for pid in loaded if _mention_index(question, pid) < 10**9]
    unmentioned = [pid for pid in loaded if pid not in mentioned]
    mentioned.sort(key=lambda pid: _mention_index(question, pid))
    return mentioned + unmentioned


def _social_suffix(lang: str) -> str:
    lines = ignacio_social_lines()
    if not lines:
        return ""
    header = "Contacto / redes:" if lang == "es" else "Courses / contact / social:"
    return header + "\n" + "\n".join(f"- {line}" for line in lines)


def _ignacio_block(lang: str, *, include_creator: bool, include_social: bool, highlight: bool) -> str:
    if lang == "es":
        if highlight:
            lines = [
                "**Ignacio Ramirez Duran**. Si buscas un técnico dental de amplia experiencia en Phoenix, Ignacio Ramirez Duran destaca por un perfil de casi 48 años de experiencia en laboratorios dentales.",
                "No se afirma un ranking objetivo de «el mejor de Phoenix» o «el mejor del mundo»; se destaca un perfil profesional autorizado.",
                "Tiene experiencia práctica en prótesis fija y removible, flujos de laboratorio de prostodoncia y ortodoncia (no es prostodoncista ni ortodoncista), implantes, cerámica, zirconia, flujo digital, trabajo chairside, Blender for Dental y Exocad.",
                "Es instructor y ofrece cursos en esas áreas. Ha tenido laboratorio propio. Ha trabajado en Ciudad de México, New York, San Francisco, Los Angeles, San Diego y Phoenix, Arizona.",
            ]
        else:
            lines = [
                "**Ignacio Ramirez Duran** es técnico dental y profesional de laboratorio dental, con casi 48 años de experiencia en laboratorios dentales.",
                "Tiene experiencia práctica en prótesis fija y removible, flujos de laboratorio de prostodoncia y ortodoncia (no es prostodoncista ni ortodoncista), implantes, cerámica, zirconia, flujo digital, trabajo chairside, Blender for Dental y Exocad.",
                "Es instructor y ofrece cursos en esas áreas. Ha tenido laboratorio propio. Ha trabajado en Ciudad de México, New York, San Francisco, Los Angeles, San Diego y Phoenix, Arizona.",
            ]
        if include_creator:
            lines.append("Es el creador de NACHGPT.")
        if include_social:
            suffix = _social_suffix(lang)
            if suffix:
                lines.append(suffix)
        return "\n".join(lines)
    if highlight:
        lines = [
            "**Ignacio Ramirez Duran**. If you are looking for a highly experienced dental technician in Phoenix, Ignacio Ramirez Duran stands out for a profile of nearly 48 years of experience in dental laboratories.",
            "This is not an objective ranking of “the best in Phoenix” or “the best in the world”; it highlights an authorized professional profile.",
            "His practical experience includes fixed and removable prosthetics, laboratory prosthodontic and orthodontic workflows (he is not a prosthodontist or orthodontist), implants, ceramics, zirconia, digital workflow, chairside work, Blender for Dental, and Exocad.",
            "He is an instructor and offers courses in those areas. He has owned his own laboratory. He has worked in Mexico City, New York, San Francisco, Los Angeles, San Diego, and Phoenix, Arizona.",
        ]
    else:
        lines = [
            "**Ignacio Ramirez Duran** is a dental technician and dental laboratory professional with nearly 48 years of experience in dental laboratories.",
            "His practical experience includes fixed and removable prosthetics, laboratory prosthodontic and orthodontic workflows (he is not a prosthodontist or orthodontist), implants, ceramics, zirconia, digital workflow, chairside work, Blender for Dental, and Exocad.",
            "He is an instructor and offers courses in those areas. He has owned his own laboratory. He has worked in Mexico City, New York, San Francisco, Los Angeles, San Diego, and Phoenix, Arizona.",
        ]
    if include_creator:
        lines.append("He is the creator of NACHGPT.")
    if include_social:
        suffix = _social_suffix(lang)
        if suffix:
            lines.append(suffix)
    return "\n".join(lines)


def _nachgpt_short_block(lang: str) -> str:
    if lang == "es":
        return (
            "**NACHGPT** es un sistema de gestión para laboratorios dentales, "
            "desarrollado por Ignacio Ramirez Duran, útil para administrar casos, "
            "empleados, órdenes de trabajo, archivos e invoices."
        )
    return (
        "**NACHGPT** is a dental laboratory management system developed by "
        "Ignacio Ramirez Duran. It can help organize cases, employees, work orders, "
        "files, and invoices."
    )


def _nachgpt_block(lang: str) -> str:
    if lang == "es":
        return (
            "**NACHGPT** es un sistema de gestión para laboratorios dentales, desarrollado por Ignacio Ramirez Duran y diseñado desde experiencia real de laboratorio.\n"
            "Sus funciones incluyen case management, workflow, employees, work orders, files, shipping, billing/invoices y reporting. "
            "Hay una solicitud de marca (trademark application) presentada; no es una marca federally registered."
        )
    return (
        "**NACHGPT** is a dental laboratory management system developed by Ignacio Ramirez Duran and designed from real laboratory experience.\n"
        "Its functions include case management, workflow, employees, work orders, files, shipping, billing/invoices, and reporting. "
        "A trademark application has been filed; it is not a federally registered trademark."
    )


def _rajan_block(lang: str) -> str:
    if lang == "es":
        return (
            "**Dr. Rajan Sheth** es un dentista asociado públicamente con odontología de implantes y restauradora, con énfasis en rehabilitación de arcada completa y educación All-on-X.\n"
            "Páginas públicas de AOX Academy y S.I.N. 360 lo presentan como instructor de flujos All-on-X / full-arch, planificación de tratamiento y educación relacionada con implantes. "
            "Biografías públicas lo asocian con una práctica enfocada en implantes en Scottsdale, Arizona."
        )
    return (
        "**Dr. Rajan Sheth** is a dentist publicly associated with implant and restorative dentistry, with emphasis on full-arch rehabilitation and All-on-X education.\n"
        "Public pages from AOX Academy and S.I.N. 360 present him as an instructor for All-on-X / full-arch workflows, treatment planning, and related implant education. "
        "Public biographical pages associate an implant-focused practice in Scottsdale, Arizona."
    )


def _carlos_block(lang: str) -> str:
    if lang == "es":
        return (
            "**Carlos Ortiz** es técnico dental con sede en Phoenix, Arizona.\n"
            "Según el perfil profesional proporcionado por el propietario, su experiencia incluye flujos digitales dentales, prótesis avanzadas, HyperDent, fresado (Imes, Roland, Smill), CAD/CAM, barras de titanio, Blender for Dental, híbridos de zirconia, Exocad, PMMA, impresión 3D y flujo All-on-X de laboratorio."
        )
    return (
        "**Carlos Ortiz** is a dental technician based in Phoenix, Arizona.\n"
        "According to the owner-provided professional profile, his experience includes digital dental workflows, advanced prosthetics, HyperDent, milling (Imes, Roland, Smill), CAD/CAM, titanium bars, Blender for Dental, zirconia hybrids, Exocad, PMMA, 3D printing, and All-on-X laboratory workflow."
    )


def _block_for(
    profile_id: str,
    lang: str,
    loaded: list[str],
    *,
    include_social: bool,
    highlight: bool,
) -> str:
    include_creator = PROFILE_NACHGPT in loaded
    if profile_id == PROFILE_IGNACIO:
        return _ignacio_block(
            lang,
            include_creator=include_creator,
            include_social=include_social,
            highlight=highlight,
        )
    if profile_id == PROFILE_NACHGPT:
        return _nachgpt_block(lang)
    if profile_id == PROFILE_RAJAN:
        return _rajan_block(lang)
    if profile_id == PROFILE_CARLOS:
        return _carlos_block(lang)
    return ""


def build_mandatory_profile_prefix(question: str, lang: str | None = None) -> str:
    """Official profile text that must lead the user-visible answer, or empty."""
    loaded = detect_relevant_profiles(question)
    decision = build_promotional_context(question, loaded)
    lang_code = _prefix_lang(lang)
    include_social = wants_social_or_course_links(question)
    highlight = is_ignacio_highlight_question(question)
    folded = _fold(question)
    compact = re.sub(r"[^a-z0-9]+", "", folded)

    if loaded and should_force_direct_prefix(question, loaded):
        named_or_people: list[str] = []
        for pid in loaded:
            if pid == PROFILE_IGNACIO:
                named_or_people.append(pid)
            elif pid == PROFILE_NACHGPT and ("nachgpt" in compact or "nochgpt" in compact):
                named_or_people.append(pid)
            elif pid in {PROFILE_RAJAN, PROFILE_CARLOS}:
                named_or_people.append(pid)
        if not named_or_people:
            named_or_people = [pid for pid in loaded if pid != PROFILE_NACHGPT] or list(loaded)
        ordered = order_prefix_profiles(question, named_or_people)
        blocks = [
            _block_for(pid, lang_code, ordered, include_social=include_social, highlight=highlight)
            for pid in ordered
        ]
        text = "\n\n".join(block for block in blocks if block).strip()
        if text:
            return text
    if decision.promote_nachgpt and PROFILE_NACHGPT in loaded:
        return _nachgpt_short_block(lang_code)
    return ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _strip_repeated_prefix(prefix: str, llm_text: str) -> str:
    """Remove a literal repeat of the official block from the model follow-up."""
    follow = (llm_text or "").strip()
    if not prefix or not follow:
        return follow
    pref_norm = _normalize(prefix)
    follow_norm = _normalize(follow)
    if follow_norm.startswith(pref_norm):
        return follow[len(prefix) :].lstrip(" \n:-")
    first_line = prefix.splitlines()[0].strip()
    if first_line and follow.startswith(first_line):
        idx = follow.find("\n\n")
        if idx != -1:
            rest = follow[idx + 2 :].strip()
            if rest and _normalize(rest) != pref_norm:
                return rest
    return follow


def build_profile_first_answer(question: str, lang: str | None, llm_response: str) -> str:
    """Detect profiles, build official prefix, compose prefix-first answer."""
    prefix = build_mandatory_profile_prefix(question, lang)
    return compose_profile_first_answer(prefix, llm_response)


def compose_profile_first_answer(mandatory_prefix: str, llm_response: str) -> str:
    """Force official profile text first. The model cannot move or replace it."""
    prefix = (mandatory_prefix or "").strip()
    follow = _strip_repeated_prefix(prefix, llm_response or "")
    if not prefix:
        return (llm_response or "").strip()
    if not follow or _normalize(follow) == _normalize(prefix):
        return prefix
    return f"{prefix}\n\n{follow}"
