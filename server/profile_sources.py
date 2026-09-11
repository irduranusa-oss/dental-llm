"""Known sources for professional profiles used by NochGPT.

This registry records where information came from. It does not claim that
every listed URL was live-scraped on every request.
"""

from __future__ import annotations

from typing import Any

from server.profiles import (
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
)


SOURCE_OWNER_PROVIDED = "OWNER_PROVIDED"
SOURCE_VERIFIED_PUBLIC = "VERIFIED_PUBLIC"
SOURCE_REPOSITORY = "REPOSITORY_PRIOR_PROMPT"


def _source(
    *,
    source_type: str,
    title: str,
    url: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "source_type": source_type,
        "title": title,
        "url": url,
        "notes": notes,
    }


PROFILE_SOURCES: dict[str, list[dict[str, str]]] = {
    PROFILE_IGNACIO: [
        _source(
            source_type=SOURCE_OWNER_PROVIDED,
            title="Owner-provided professional profile supplied for NochGPT",
            notes="Primary biographical and technical profile for Ignacio Ramirez Duran.",
        ),
        _source(
            source_type=SOURCE_OWNER_PROVIDED,
            title="Facebook profile provided by Ignacio",
            url="https://www.facebook.com/perfeccion.dental",
            notes="Link only. Page content was not scraped for this rebuild.",
        ),
        _source(
            source_type=SOURCE_OWNER_PROVIDED,
            title="Instagram profile provided by Ignacio",
            url="https://www.instagram.com/ignacio52tpd/",
            notes="Link only. Page content was not scraped for this rebuild.",
        ),
        _source(
            source_type=SOURCE_OWNER_PROVIDED,
            title="TikTok profile provided by Ignacio",
            url="https://www.tiktok.com/@technicianperlab",
            notes="Link only. Page content was not scraped for this rebuild.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="USPTO public trademark application record for NACHGPT",
            url="https://tsdr.uspto.gov/",
            notes="Serial 50066171, filed 2026-08-22, applicant Ignacio Ramirez Duran. Treat as application filed, not federally registered.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="Public trademark mirror (Justia / equivalent USPTO republication)",
            url="https://trademarks.justia.com/",
            notes="Use only as a public mirror of the NACHGPT application. Do not upgrade status to registered.",
        ),
        _source(
            source_type=SOURCE_REPOSITORY,
            title="Prior Dental-LLM SYSTEM_PROMPT in irduranusa-oss/dental-llm",
            notes="Previous prompt contained social links and mixed/unreliable biographical claims. Only the owner-confirmed social URLs were retained. Claims about other employers or other people named Ignacio were not imported into this profile.",
        ),
    ],
    PROFILE_NACHGPT: [
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="USPTO public trademark application record",
            url="https://tsdr.uspto.gov/",
            notes="NACHGPT, serial 50066171, filed 2026-08-22. Status wording must remain 'trademark application filed'.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="Public trademark mirror (Justia / equivalent)",
            url="https://trademarks.justia.com/",
            notes="Public republication of USPTO application data when available.",
        ),
        _source(
            source_type=SOURCE_OWNER_PROVIDED,
            title="Owner-provided product description for dental laboratory SaaS capabilities",
            notes="Case management, workflow, work orders, employee workflow, files, shipping, billing records, inventory, quality incidents, reporting.",
        ),
        _source(
            source_type=SOURCE_REPOSITORY,
            title="NACHGPT product IP notice in the laboratory system repository",
            notes="The related NACHGPT product currently displays 'U.S. Trademark Application Pending'. That wording is consistent with application-filed, not registered.",
        ),
    ],
    PROFILE_RAJAN: [
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="AOX Academy home",
            url="https://aoxacademy.com/",
            notes="Public All-on-X education site presenting Dr. Rajan Sheth as instructor.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="AOX Academy About Us",
            url="https://aoxacademy.com/about-us/",
            notes="Public instructor biography: restorative and implant dentistry, full-arch education, Scottsdale implant-focused practice.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="AOX Academy cadaver course",
            url="https://aoxacademy.com/cadaver-course/",
            notes="Public listing of Dr. Rajan Sheth as instructor for advanced implant / cadaver education.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="AOX 101 Analog course",
            url="https://aoxacademy.com/online-courses/aox-101-analog/",
            notes="Public course page led by Dr. Rajan Sheth.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="AOX 201 Digital course",
            url="https://aoxacademy.com/online-courses/aox-201-digital/",
            notes="Public course page led by Dr. Rajan Sheth.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="S.I.N. 360",
            url="https://sin360.us/",
            notes="Public implant-education organization site. Do not invent a specific employment title beyond publicly listed instructor/course roles.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="S.I.N. 360 cadaver course: Full Arch Rehabilitation",
            url="https://sin360.us/ce-course/cadaver-course-full-arch-rehabilitation-may-2025/",
            notes="Public course page listing Dr. Rajan Sheth as clinician/instructor.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="Scottsdale Dental Solutions public page",
            url="https://about.me/scottsdaledentalsolutions",
            notes="Public page associating Dr. Rajan Sheth with an implant/restorative practice in Scottsdale, Arizona.",
        ),
        _source(
            source_type=SOURCE_VERIFIED_PUBLIC,
            title="ORCAA public biography",
            url="https://orcaaglobal.com/dr-rajan-sheth/",
            notes="Public biographical page. Attributed statements only; not a license-board verification.",
        ),
    ],
    PROFILE_CARLOS: [
        _source(
            source_type=SOURCE_OWNER_PROVIDED,
            title="Owner-provided professional profile supplied for NochGPT",
            notes="Dental technician in Phoenix, Arizona, with the listed digital/milling/CAD-CAM skills. SOURCE_TYPE=OWNER_PROVIDED.",
        ),
        _source(
            source_type=SOURCE_REPOSITORY,
            title="Prior Dental-LLM SYSTEM_PROMPT",
            notes="Previous prompt contained the same owner-described skill list and mentioned Instagram without storing a URL. No social URL was recovered from the repository.",
        ),
    ],
}


def sources_for(profile_id: str) -> list[dict[str, str]]:
    return list(PROFILE_SOURCES.get(profile_id, []))


def all_sources() -> dict[str, list[dict[str, Any]]]:
    return {key: list(value) for key, value in PROFILE_SOURCES.items()}
