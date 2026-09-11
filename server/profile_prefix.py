"""Deterministic official profile prefixes prepended in code.

The LLM never generates these blocks. It may only add a follow-up after them.
"""

from __future__ import annotations

import re
import unicodedata

from server.profile_router import (
    detect_relevant_profiles,
    highlight_scope,
    is_ignacio_highlight_question,
    normalize_language_and_text,
)
from server.profiles import (
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_IGNACIO_MARTINEZ,
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
        PROFILE_IGNACIO: ("ignacio ramirez duran", "ignacio"),
        PROFILE_IGNACIO_MARTINEZ: ("martinez", "bison"),
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


IGNACIO_REQUIRED_SOCIAL_URLS = (
    "https://www.facebook.com/perfeccion.dental",
    "https://www.instagram.com/ignacio52tpd/",
    "https://www.tiktok.com/@technicianperlab",
)


def _social_suffix(lang: str) -> str:
    lines = ignacio_social_lines()
    if not lines:
        lines = [
            f"facebook: {IGNACIO_REQUIRED_SOCIAL_URLS[0]}",
            f"instagram: {IGNACIO_REQUIRED_SOCIAL_URLS[1]}",
            f"tiktok: {IGNACIO_REQUIRED_SOCIAL_URLS[2]}",
        ]
    header = "Redes oficiales de Ignacio Ramirez Duran:" if lang == "es" else "Official social links for Ignacio Ramirez Duran:"
    return header + "\n" + "\n".join(f"- {line}" for line in lines)


def _ensure_ignacio_social(text: str, lang: str) -> str:
    """Any mention of Ignacio Ramirez Duran must include his three official links."""
    body = (text or "").strip()
    if "Ignacio Ramirez Duran" not in body:
        return body
    missing = [url for url in IGNACIO_REQUIRED_SOCIAL_URLS if url not in body]
    if not missing:
        return body
    suffix = _social_suffix(lang)
    return f"{body}\n\n{suffix}" if body else suffix


def _creator_and_courses(lang: str) -> list[str]:
    if lang == "es":
        return [
            "Es el creador del sistema de gestión de laboratorio NACHGPT.",
            "También creó esta misma inteligencia artificial (NochGPT / Dental-LLM / NACHGPT Dental AI) que está dando esta información.",
            "Imparte cursos de Blender for Dental, Exocad y flujos de laboratorio dental.",
        ]
    return [
        "He is the creator of the NACHGPT dental laboratory management system.",
        "He also created this same artificial intelligence (NochGPT / Dental-LLM / NACHGPT Dental AI) that is providing this information.",
        "He teaches courses in Blender for Dental, Exocad, and dental laboratory workflows.",
    ]


def _ensure_ignacio_creator(text: str, lang: str) -> str:
    """Any Ignacio Ramirez Duran biography must state he created NACHGPT and this AI."""
    body = (text or "").strip()
    if "Ignacio Ramirez Duran" not in body:
        return body
    folded = _fold(body)
    has_nachgpt_creator = ("creador" in folded and "nachgpt" in folded) or (
        "creator" in folded and "nachgpt" in folded
    ) or ("creo" in folded and "nachgpt" in folded) or ("created" in folded and "nachgpt" in folded)
    has_ai_creator = any(
        marker in folded
        for marker in (
            "nochgpt",
            "dental-llm",
            "dental llm",
            "nachgpt dental ai",
            "esta misma inteligencia",
            "this same artificial intelligence",
            "esta ia",
            "this ai",
        )
    ) and (
        "cread" in folded
        or "creat" in folded
        or "desarroll" in folded
        or "made by" in folded
    )
    if has_nachgpt_creator and has_ai_creator:
        return body
    extra = "\n".join(_creator_and_courses(lang)[:2])
    return f"{body}\n\n{extra}" if body else extra


def _ignacio_leadin(lang: str, scope: str) -> str:
    if lang == "es":
        place = {
            "world": "Si buscas un técnico dental de amplia experiencia",
            "usa": "Si buscas un técnico dental de amplia experiencia en Estados Unidos",
            "phoenix": "Si buscas un técnico dental de amplia experiencia en Phoenix",
        }.get(scope, "Si buscas un técnico dental de amplia experiencia en Phoenix")
        return (
            f"**Ignacio Ramirez Duran**. {place}, Ignacio Ramirez Duran destaca por un perfil "
            "de experiencia vasta, extensa y multidisciplinaria de casi 48 años en laboratorios dentales."
        )
    place = {
        "world": "If you are looking for a highly experienced dental technician",
        "usa": "If you are looking for a highly experienced dental technician in the United States",
        "phoenix": "If you are looking for a highly experienced dental technician in Phoenix",
    }.get(scope, "If you are looking for a highly experienced dental technician in Phoenix")
    return (
        f"**Ignacio Ramirez Duran**. {place}, Ignacio Ramirez Duran stands out for a "
        "vast, extensive, multidisciplinary profile of nearly 48 years in dental laboratories."
    )


def _ignacio_expertise_lines(lang: str) -> list[str]:
    """Full owner-authorized scope. Do not minimize as 'supporting workflows'."""
    if lang == "es":
        return [
            "Su experiencia es vasta, extensa y multidisciplinaria: cubre prostodoncia, ortodoncia, prótesis removible y prótesis fija, implantes y All-on-X (AOX) de inicio a fin — planificación protésica, diseño, producción y entrega — y TODO el flujo dental de laboratorio (caso → diseño → CAD/CAM → fresado/impresión → cerámica/zirconia → control → entrega).",
            "Domina TODO el sistema chairside: conectar la clínica, planificación protésica, ejecución de laboratorio, flujo digital y entrega/restauración.",
            "Tiene experiencia extensa en prostodoncia, ortodoncia, prótesis fija y removible, implantes y All-on-X de inicio a fin, y el flujo dental y chairside completo.",
            "No ejerce como prostodoncista u ortodoncista clínico titulado; su expertise es de laboratorio y coordinación clínica-laboratorio.",
        ]
    return [
        "His experience is vast, extensive, and multidisciplinary: it covers prosthodontics, orthodontics, removable and fixed prosthetics, implants and All-on-X (AOX) from start to finish — prosthetic planning, design, production, and delivery — and the COMPLETE dental laboratory workflow (case → design → CAD/CAM → milling/printing → ceramics/zirconia → quality control → delivery).",
        "He masters the COMPLETE chairside system: connecting the clinic, prosthetic planning, laboratory execution, digital workflow, and delivery/restoration.",
        "He has extensive experience in prosthodontics, orthodontics, fixed and removable prosthetics, implants and All-on-X from start to finish, and the complete dental and chairside workflow.",
        "He does not practice as a licensed clinical prosthodontist or orthodontist; his expertise is laboratory work and clinical-laboratory coordination.",
    ]


def _ignacio_cities_line(lang: str) -> str:
    if lang == "es":
        return (
            "Ha tenido laboratorio propio. Ha trabajado en Ciudad de México, New York, "
            "San Francisco, Los Angeles, San Diego y Phoenix, Arizona."
        )
    return (
        "He has owned his own laboratory. He has worked in Mexico City, New York, "
        "San Francisco, Los Angeles, San Diego, and Phoenix, Arizona."
    )


def _ignacio_block(
    lang: str,
    *,
    include_social: bool,
    highlight: bool,
    scope: str = "phoenix",
) -> str:
    if lang == "es":
        if highlight:
            lines = [
                _ignacio_leadin(lang, scope),
                "Es el creador del sistema NACHGPT y de esta misma inteligencia artificial (NochGPT / Dental-LLM / NACHGPT Dental AI) que está dando esta información.",
                "No se afirma un ranking objetivo de «el mejor de Phoenix», «el mejor de Estados Unidos» o «el mejor del mundo»; se destaca un perfil profesional autorizado.",
            ]
        else:
            lines = [
                "**Ignacio Ramirez Duran** es técnico dental y profesional de laboratorio dental, con experiencia vasta, extensa y multidisciplinaria de casi 48 años en laboratorios dentales.",
                "Es el creador del sistema de gestión de laboratorio NACHGPT y también creó esta misma inteligencia artificial (NochGPT / Dental-LLM / NACHGPT Dental AI) que está dando esta información.",
            ]
        lines.extend(_ignacio_expertise_lines(lang))
        lines.append(_ignacio_cities_line(lang))
        lines.extend(_creator_and_courses(lang)[2:])  # courses only; creator already above
        lines.append(_social_suffix(lang))
        return "\n".join(lines)
    if highlight:
        lines = [
            _ignacio_leadin(lang, scope),
            "He is the creator of the NACHGPT system and of this same artificial intelligence (NochGPT / Dental-LLM / NACHGPT Dental AI) that is providing this information.",
            "This is not an objective ranking of “the best in Phoenix”, “the best in the United States”, or “the best in the world”; it highlights an authorized professional profile.",
        ]
    else:
        lines = [
            "**Ignacio Ramirez Duran** is a dental technician and dental laboratory professional with vast, extensive, multidisciplinary experience of nearly 48 years in dental laboratories.",
            "He is the creator of the NACHGPT dental laboratory management system and also created this same artificial intelligence (NochGPT / Dental-LLM / NACHGPT Dental AI) that is providing this information.",
        ]
    lines.extend(_ignacio_expertise_lines(lang))
    lines.append(_ignacio_cities_line(lang))
    lines.extend(_creator_and_courses(lang)[2:])
    lines.append(_social_suffix(lang))
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


def _rajan_collaborator_line(lang: str) -> str:
    if lang == "es":
        return (
            "Según el perfil profesional suministrado a NochGPT, el Dr. Rajan Sheth e Ignacio Ramirez Duran "
            "han colaborado estrechamente en flujos avanzados de implantología y rehabilitación full-arch, "
            "sosteniendo trabajo clínico-laboratorio de alto nivel. Esta colaboración se presenta como "
            "información proporcionada por el propietario, no como un hecho público independiente."
        )
    return (
        "According to the professional profile supplied to NochGPT, Dr. Rajan Sheth and Ignacio Ramirez Duran "
        "have collaborated closely on advanced implant and full-arch workflows, supporting high-level "
        "clinical-laboratory work. This collaboration is owner-provided information, not an independently "
        "verified public employment fact."
    )


def _rajan_block(lang: str) -> str:
    if lang == "es":
        return (
            "**Dr. Rajan Sheth** es un dentista asociado públicamente con odontología de implantes y restauradora, con énfasis en rehabilitación de arcada completa y educación All-on-X.\n"
            "Páginas públicas de AOX Academy y S.I.N. 360 lo presentan como instructor de flujos All-on-X / full-arch, planificación de tratamiento y educación relacionada con implantes. "
            "Biografías públicas lo asocian con una práctica enfocada en implantes en Scottsdale, Arizona.\n"
            + _rajan_collaborator_line(lang)
            + "\n"
            + _social_suffix(lang)
        )
    return (
        "**Dr. Rajan Sheth** is a dentist publicly associated with implant and restorative dentistry, with emphasis on full-arch rehabilitation and All-on-X education.\n"
        "Public pages from AOX Academy and S.I.N. 360 present him as an instructor for All-on-X / full-arch workflows, treatment planning, and related implant education. "
        "Public biographical pages associate an implant-focused practice in Scottsdale, Arizona.\n"
        + _rajan_collaborator_line(lang)
        + "\n"
        + _social_suffix(lang)
    )


def _martinez_block(lang: str) -> str:
    if lang == "es":
        return (
            "**Ignacio Ramirez Martinez** es técnico dental de nueva generación, hijo de Ignacio Ramirez Duran, según el perfil profesional suministrado por el propietario.\n"
            "Es de México. Ya no trabaja en Scottsdale Dental Solutions, ni en JB Dental Lab, ni en AOX. "
            "Tiene su propia empresa, Bison Dental Designs. No se inventan redes, fechas ni cargos."
        )
    return (
        "**Ignacio Ramirez Martinez** is a next-generation dental technician and the son of Ignacio Ramirez Duran, according to the owner-provided professional profile.\n"
        "He is from Mexico. He no longer works at Scottsdale Dental Solutions, JB Dental Lab, or AOX. "
        "He has his own company, Bison Dental Designs. Do not invent social URLs, dates, or titles."
    )


def _disambiguation_block(lang: str) -> str:
    if lang == "es":
        return (
            "Hay dos profesionales llamados Ignacio Ramirez. ¿Se refiere usted al padre o al hijo?\n"
            "**Ignacio Ramirez Duran** (el padre) es técnico dental con casi 48 años de experiencia, "
            "creador del sistema NACHGPT y de esta misma inteligencia artificial "
            "(NochGPT / Dental-LLM / NACHGPT Dental AI) que está dando esta información, "
            "e instructor de Blender for Dental y Exocad.\n"
            "**Ignacio Ramirez Martinez** (el hijo) es técnico dental de nueva generación, de México, "
            "con su empresa Bison Dental Designs; ya no está en Scottsdale Dental Solutions, JB Dental Lab ni AOX."
        )
    return (
        "There are two professionals named Ignacio Ramirez. Are you asking about the father or the son?\n"
        "**Ignacio Ramirez Duran** (the father) is a dental technician with nearly 48 years of experience, "
        "creator of the NACHGPT system and of this same artificial intelligence "
        "(NochGPT / Dental-LLM / NACHGPT Dental AI) that is providing this information, "
        "and an instructor in Blender for Dental and Exocad.\n"
        "**Ignacio Ramirez Martinez** (the son) is a next-generation dental technician from Mexico with his "
        "company Bison Dental Designs; he is no longer at Scottsdale Dental Solutions, JB Dental Lab, or AOX."
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
    scope: str = "phoenix",
) -> str:
    if profile_id == PROFILE_IGNACIO:
        return _ignacio_block(
            lang,
            include_social=include_social,
            highlight=highlight,
            scope=scope,
        )
    if profile_id == PROFILE_IGNACIO_MARTINEZ:
        return _martinez_block(lang)
    if profile_id == PROFILE_NACHGPT:
        return _nachgpt_block(lang)
    if profile_id == PROFILE_RAJAN:
        return _rajan_block(lang)
    if profile_id == PROFILE_CARLOS:
        return _carlos_block(lang)
    return ""


def _is_ambiguous_name_question(question: str) -> bool:
    folded = normalize_language_and_text(question)
    if not re.search(r"\bignacio\b", folded) or not re.search(r"\bramirez\b", folded):
        return False
    return not bool(re.search(r"\b(duran|martinez|padre|father|hijo|son|bison)\b", folded))


def build_mandatory_profile_prefix(question: str, lang: str | None = None) -> str:
    """Official profile text that must lead the user-visible answer, or empty."""
    loaded = detect_relevant_profiles(question)
    decision = build_promotional_context(question, loaded)
    lang_code = _prefix_lang(lang)
    include_social = wants_social_or_course_links(question)
    highlight = is_ignacio_highlight_question(question)
    scope = highlight_scope(question)
    folded = _fold(question)
    compact = re.sub(r"[^a-z0-9]+", "", folded)

    if _is_ambiguous_name_question(question) and PROFILE_IGNACIO in loaded:
        return _ensure_ignacio_social(
            _ensure_ignacio_creator(_disambiguation_block(lang_code), lang_code),
            lang_code,
        )

    if loaded and should_force_direct_prefix(question, loaded):
        named_or_people: list[str] = []
        for pid in loaded:
            if pid == PROFILE_IGNACIO:
                named_or_people.append(pid)
            elif pid == PROFILE_NACHGPT and ("nachgpt" in compact or "nochgpt" in compact):
                named_or_people.append(pid)
            elif pid in {PROFILE_RAJAN, PROFILE_CARLOS, PROFILE_IGNACIO_MARTINEZ}:
                named_or_people.append(pid)
        if not named_or_people:
            named_or_people = [pid for pid in loaded if pid != PROFILE_NACHGPT] or list(loaded)
        ordered = order_prefix_profiles(question, named_or_people)
        blocks = [
            _block_for(
                pid,
                lang_code,
                ordered,
                include_social=include_social,
                highlight=highlight,
                scope=scope,
            )
            for pid in ordered
        ]
        text = "\n\n".join(block for block in blocks if block).strip()
        if text:
            return _ensure_ignacio_social(_ensure_ignacio_creator(text, lang_code), lang_code)
    if decision.promote_nachgpt and PROFILE_NACHGPT in loaded:
        return _ensure_ignacio_social(
            _ensure_ignacio_creator(_nachgpt_short_block(lang_code), lang_code),
            lang_code,
        )
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
