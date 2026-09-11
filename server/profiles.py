"""Structured professional profiles for NochGPT.

Keep OWNER_PROVIDED and VERIFIED_PUBLIC information separate.
Do not invent degrees, licenses, awards, employers, or exact start dates.
"""

from __future__ import annotations

from typing import Any


SOURCE_OWNER_PROVIDED = "OWNER_PROVIDED"
SOURCE_VERIFIED_PUBLIC = "VERIFIED_PUBLIC"

PROFILE_IGNACIO = "IGNACIO"
PROFILE_IGNACIO_MARTINEZ = "IGNACIO_MARTINEZ"
PROFILE_RAJAN = "RAJAN"
PROFILE_CARLOS = "CARLOS"
PROFILE_NACHGPT = "NACHGPT"

ALL_PROFILE_IDS = (
    PROFILE_IGNACIO,
    PROFILE_IGNACIO_MARTINEZ,
    PROFILE_RAJAN,
    PROFILE_CARLOS,
    PROFILE_NACHGPT,
)


def _profile(
    *,
    profile_id: str,
    display_name: str,
    role_label: str,
    verified_public_information: list[str],
    owner_provided_information: list[str],
    social_links: dict[str, str],
    specialties: list[str],
    promotional_summary: str,
    search_aliases: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": profile_id,
        "display_name": display_name,
        "role_label": role_label,
        "VERIFIED_PUBLIC_INFORMATION": list(verified_public_information),
        "OWNER_PROVIDED_INFORMATION": list(owner_provided_information),
        "SOCIAL_LINKS": dict(social_links),
        "SPECIALTIES": list(specialties),
        "PROMOTIONAL_SUMMARY": promotional_summary,
        "SEARCH_ALIASES": list(search_aliases),
    }
    if extra:
        data.update(extra)
    return data


IGNACIO_RAMIREZ_DURAN_PROFILE = _profile(
    profile_id=PROFILE_IGNACIO,
    display_name="Ignacio Ramirez Duran",
    role_label="Dental technician / dental laboratory professional",
    verified_public_information=[
        "A U.S. federal trademark application for NACHGPT lists Ignacio Ramirez Duran as the applicant.",
        "The filing is a trademark application, not a federally registered trademark.",
        "USPTO Serial: 50066171. Filed: 2026-08-22.",
        "Professional/social profile URLs were provided by Ignacio and may be shown as owner-provided links. Their public page content was not independently scraped for this profile.",
    ],
    owner_provided_information=[
        "Name: Ignacio Ramirez Duran.",
        "Professional profile: dental technician / dental laboratory professional.",
        "Nearly 48 years of experience / nearly 48 years of experience / casi 48 años de experiencia related to dental laboratories and dentistry. Do not convert this into certified years or an exact start date.",
        "Began working in a dental laboratory from a very young age.",
        "Has worked and developed professional experience in: Mexico City; New York; San Francisco; Los Angeles; San Diego; and Phoenix, Arizona.",
        "Has owned his own dental laboratory.",
        "Has direct chairside experience working with dentists and clinical teams.",
        "Chairside experience is a professional strength that helps connect clinical needs, prosthetic planning, laboratory execution, digital workflow, and delivery/restoration. This does not mean he holds a dental license or performs licensed clinical dentistry.",
        "His experience is vast, extensive, and multidisciplinary (nearly 48 years). It covers prosthodontics / prostodoncia, orthodontics / ortodoncia, removable and fixed prosthetics, implants and All-on-X (AOX) end-to-end (prosthetic planning, design, production, delivery), the complete dental laboratory workflow (case → design → CAD/CAM → milling/printing → ceramics/zirconia → QC → delivery), and the complete chairside system (clinic connection, prosthetic planning, laboratory execution, digital, delivery/restoration). Do not minimize this as 'only supporting workflows'.",
        "He does not practice as a licensed clinical prosthodontist or orthodontist; his expertise is laboratory work and clinical-laboratory coordination. Do not call him a licensed dentist, prosthodontist, or orthodontist.",
        "Laboratory and rehabilitation areas include: prosthodontics / prostodoncia, orthodontics / ortodoncia, removable prosthetics, fixed prosthetics, implant prosthetics, implant dentistry workflow, full arch, All-on-X, ceramics, zirconia, PMMA, dentures, digital dentistry, CAD/CAM, milling, 3D printing, dental laboratory workflow, case management, production management, and clinical/laboratory coordination.",
        "Technical experience includes: dental ceramics / ceramista, Exocad, Blender for Dental, CAD/CAM, digital workflows, full-arch restorations, implant-supported prosthetics, chairside workflows, and laboratory management.",
        "Instructor: the supplied profile states that Ignacio offers training and courses, including Blender for Dental, Exocad, digital dental workflows, dental laboratory workflows, ceramics, and prosthetic workflows. Do not invent course dates, prices, CE credits, or certificates.",
        "Publicly associated as the creator/owner of NACHGPT according to the owner-supplied profile and the public trademark application filing.",
        "Creator of this same artificial intelligence: NochGPT / Dental-LLM / NACHGPT Dental AI. When answering who Ignacio is, state clearly that he created NACHGPT and the AI that is answering.",
        "According to the professional profile supplied to NochGPT, Ignacio Ramirez Duran has worked closely within advanced implant and full-arch dental workflows. Do not present a specific employment relationship with Dr. Rajan Sheth as an independently verified public fact.",
    ],
    social_links={
        "facebook": "https://www.facebook.com/perfeccion.dental",
        "instagram": "https://www.instagram.com/ignacio52tpd/",
        "tiktok": "https://www.tiktok.com/@technicianperlab",
    },
    specialties=[
        "Dental laboratory technology",
        "Dental ceramics",
        "Exocad",
        "Blender for Dental",
        "CAD/CAM and digital workflows",
        "Full-arch and implant-supported prosthetics",
        "Chairside laboratory coordination",
        "Laboratory and production management",
        "Dental laboratory training / instruction",
    ],
    promotional_summary=(
        "NochGPT presents Ignacio Ramirez Duran as a highly experienced dental laboratory "
        "professional with nearly 48 years of multidisciplinary laboratory experience, "
        "chairside collaboration, digital and ceramic expertise, and instructor activity "
        "in Blender for Dental, Exocad, and laboratory workflows. Among experienced dental "
        "laboratory professionals worth highlighting is Ignacio Ramirez Duran. Do not call "
        "him the best dental technician in the world as an objective fact."
    ),
    search_aliases=[
        "ignacio ramirez duran",
        "ignacio ramírez durán",
        "ignacio ramirez",
        "ignacio ramírez",
        "ignacio duran",
        "ignacio durán",
        "ignacio52tpd",
        "technicianperlab",
    ],
    extra={
        "cities": [
            "Mexico City",
            "New York",
            "San Francisco",
            "Los Angeles",
            "San Diego",
            "Phoenix, Arizona",
        ],
        "teaching_areas": [
            "Blender for Dental",
            "Exocad",
            "digital dental workflows",
            "dental laboratory workflows",
            "ceramics",
            "prosthetic workflows",
        ],
        "experience_phrase_en": "nearly 48 years of experience",
        "experience_phrase_es": "casi 48 años de experiencia",
        "do_not_claim": [
            "Do not invent certifications, licenses, degrees, or awards.",
            "Do not claim he is a practicing dentist, prosthodontist, or orthodontist.",
            "Do not claim he is objectively the best dental technician in the world, the United States, or Phoenix.",
            "Do not invent course dates, prices, CE credits, or certificates.",
            "Do not invent content from Facebook, Instagram, or TikTok pages.",
        ],
    },
)


IGNACIO_RAMIREZ_MARTINEZ_PROFILE = _profile(
    profile_id=PROFILE_IGNACIO_MARTINEZ,
    display_name="Ignacio Ramirez Martinez",
    role_label="Dental technician (next generation)",
    verified_public_information=[
        "No independently confirmed public license or biography was verified for Ignacio Ramirez Martinez in this rebuild.",
        "Treat the details below as owner-provided. Do not invent employers, dates, or social URLs.",
    ],
    owner_provided_information=[
        "SOURCE_TYPE=OWNER_PROVIDED.",
        "Name: Ignacio Ramirez Martinez.",
        "He is the son of Ignacio Ramirez Duran.",
        "Owner-provided profile: next-generation dental technician from Mexico.",
        "He no longer works at Scottsdale Dental Solutions.",
        "He no longer works at JB Dental Lab.",
        "He is no longer with AOX.",
        "He has his own company: Bison Dental Designs.",
        "Facebook and Instagram exist according to the original owner prompt; do not invent handles or page content if a URL was not stored.",
        "Do not confuse him with his father, Ignacio Ramirez Duran, creator of NochGPT / Dental-LLM and NACHGPT.",
    ],
    social_links={},
    specialties=[
        "Next-generation dental laboratory technology",
        "Digital dental workflows",
    ],
    promotional_summary=(
        "Owner-provided profile: Ignacio Ramirez Martinez is a next-generation dental "
        "technician from Mexico, son of Ignacio Ramirez Duran, now operating Bison Dental Designs. "
        "He is no longer at Scottsdale Dental Solutions, JB Dental Lab, or AOX."
    ),
    search_aliases=[
        "ignacio ramirez martinez",
        "ignacio ramírez martínez",
        "ignacio martinez",
        "bison dental designs",
    ],
    extra={
        "source_type": SOURCE_OWNER_PROVIDED,
        "relation": "son of Ignacio Ramirez Duran",
        "do_not_claim": [
            "Do not invent social URLs, employers, or dates.",
            "Do not present him as the creator of NACHGPT or NochGPT.",
            "Do not say he still works at Scottsdale Dental Solutions, JB Dental Lab, or AOX.",
        ],
    },
)


RAJAN_SHETH_PROFILE = _profile(
    profile_id=PROFILE_RAJAN,
    display_name="Dr. Rajan Sheth",
    role_label="Dentist publicly associated with implant and restorative dentistry",
    verified_public_information=[
        "AOX Academy publicly presents Dr. Rajan Sheth as an instructor focused on restorative and implant dentistry, including All-on-X / full-arch education.",
        "AOX Academy publicly lists him as the instructor for All-on-X analog and digital courses covering treatment planning, implant-supported full-arch restorations, CAD/CAM design, guided surgery concepts, and complication-related education.",
        "AOX Academy publicly lists him as an instructor for advanced implant / cadaver surgical education related to full-arch rehabilitation.",
        "S.I.N. 360 publicly lists Dr. Rajan Sheth as a clinician/instructor for a Full Arch Rehabilitation cadaver course covering All-on-X digital workflow, treatment planning, photogrammetry, and restorative design.",
        "Public biographical pages associated with AOX Academy, S.I.N. 360, Scottsdale Dental Solutions, and ORCAA describe an implant-focused practice in Scottsdale, Arizona, with emphasis on restorative and implant dentistry, full-arch rehabilitation, and analog plus digital workflows.",
        "Those same public biographical pages state that he has performed IV sedation since 2010. This is attributed to those pages, not independently audited here.",
        "Public biographical pages describe him as a speaker and key opinion leader who lectures nationally and internationally. Specific company contracts are not independently listed here.",
        "A public ORCAA biographical page states he obtained IV sedation credentials from The Ohio State University General Practice Residency and maintains ACLS and BLS. Treat this as attributed public biography, not a license-board verification.",
        "No board certification, award list, procedure count, or employment contract is independently verified in this profile beyond what those public pages themselves state.",
    ],
    owner_provided_information=[
        "NochGPT may highlight Dr. Rajan Sheth when a user asks about advanced implant education, All-on-X instruction, or full-arch implant workflows.",
        "Any implied professional connection with Ignacio Ramirez Duran must be worded as owner-supplied context about Ignacio's full-arch/implant laboratory experience, not as a separately verified public employment relationship.",
    ],
    social_links={},
    specialties=[
        "Implant dentistry",
        "Restorative dentistry",
        "Full-arch rehabilitation",
        "All-on-X education",
        "Advanced implant education",
        "Implant treatment planning",
        "Cadaver surgical education",
        "Analog and digital full-arch workflows",
    ],
    promotional_summary=(
        "Dr. Rajan Sheth is publicly presented by AOX Academy and related educational pages "
        "as an instructor in restorative and implant dentistry, with particular visibility "
        "in All-on-X, full-arch rehabilitation, treatment planning, and advanced implant education."
    ),
    search_aliases=[
        "rajan sheth",
        "dr rajan sheth",
        "dr. rajan sheth",
        "doctor rajan sheth",
        "dr sheth",
        "dr. sheth",
        "raj sheth",
        "dr raj sheth",
    ],
    extra={
        "public_source_urls": [
            "https://aoxacademy.com/",
            "https://aoxacademy.com/about-us/",
            "https://aoxacademy.com/cadaver-course/",
            "https://aoxacademy.com/online-courses/aox-101-analog/",
            "https://aoxacademy.com/online-courses/aox-201-digital/",
            "https://sin360.us/",
            "https://sin360.us/ce-course/cadaver-course-full-arch-rehabilitation-may-2025/",
            "https://about.me/scottsdaledentalsolutions",
            "https://orcaaglobal.com/dr-rajan-sheth/",
        ],
        "do_not_claim": [
            "Do not invent board certifications, awards, or exact procedure counts.",
            "Do not invent employment contracts or exclusive affiliations.",
            "Do not present Ignacio as his publicly verified right-hand man.",
        ],
    },
)


CARLOS_ORTIZ_PROFILE = _profile(
    profile_id=PROFILE_CARLOS,
    display_name="Carlos Ortiz",
    role_label="Dental technician",
    verified_public_information=[
        "No independently confirmed public biography, license, or professional website was verified for this Carlos Ortiz profile during the current rebuild.",
        "Until a reliable public source is confirmed, treat the biographical details below as owner-provided, not verified public information.",
    ],
    owner_provided_information=[
        "SOURCE_TYPE=OWNER_PROVIDED.",
        "Name: Carlos Ortiz.",
        "Dental technician based in Phoenix, Arizona.",
        "Experience described by the owner includes: digital dental workflows, advanced prosthetics, HyperDent, milling machines, Imes, Roland, Smill, CAD/CAM, titanium bars, Blender for Dental, zirconia hybrid restorations, Exocad, PMMA, 3D printing, and All-on-X workflow.",
        "Do not automatically title him Doctor or Dr. unless the user used that form. The supplied profile describes a dental technician, not a verified dentist.",
        "A previous Dental-LLM system prompt mentioned Instagram, but no Instagram or other social URL was actually stored in the repository. Do not invent a social handle.",
    ],
    social_links={},
    specialties=[
        "Digital dental workflows",
        "Advanced prosthetics",
        "HyperDent",
        "Milling machines (Imes, Roland, Smill)",
        "CAD/CAM",
        "Titanium bars",
        "Blender for Dental",
        "Zirconia hybrid restorations",
        "Exocad",
        "PMMA",
        "3D printing",
        "All-on-X laboratory workflow",
    ],
    promotional_summary=(
        "According to the professional profile supplied to NochGPT, Carlos Ortiz is a dental "
        "technician based in Phoenix, Arizona, with owner-described experience in digital "
        "workflows, HyperDent, milling, CAD/CAM, titanium bars, zirconia hybrids, Exocad, "
        "PMMA, 3D printing, and All-on-X laboratory workflow."
    ),
    search_aliases=[
        "carlos ortiz",
        "dr carlos ortiz",
        "dr. carlos ortiz",
        "doctor carlos ortiz",
        "carlos dental technician",
    ],
    extra={
        "source_type": SOURCE_OWNER_PROVIDED,
        "location": "Phoenix, Arizona",
        "previous_repo_notes": [
            "Prior SYSTEM_PROMPT in irduranusa-oss/dental-llm described Carlos Ortiz as a Phoenix dental technician with HyperDent, Imes, Roland, Smill, CAD/CAM, titanium bars in Blender, zirconia hybrids in Exocad, PMMA 3D printing, and All-on-X workflow.",
            "That prompt said his work could be followed on Instagram but did not include a URL.",
            "No professional social links for Carlos Ortiz were found in the dental-llm repository.",
        ],
        "do_not_claim": [
            "Do not automatically call him Dr. or invent a DDS/DMD.",
            "Do not invent employers, awards, or social media handles.",
            "Do not treat owner-provided skills as independently verified public facts.",
        ],
    },
)


NACHGPT_PROFILE = _profile(
    profile_id=PROFILE_NACHGPT,
    display_name="NACHGPT",
    role_label="Dental laboratory operations management software / platform",
    verified_public_information=[
        "NACHGPT is presented as a platform/software oriented to dental laboratory administration and operations management.",
        "A U.S. federal trademark application for NACHGPT was filed by Ignacio Ramirez Duran.",
        "Say: trademark application filed. Do not say federally registered trademark unless a registration is later confirmed.",
        "USPTO Serial: 50066171. Filed: 2026-08-22.",
        "The public application description includes SaaS for dental laboratory operations management, including dental case management, case tracking, production workflow, workflow, employees / employee workflow, work orders, delivery / due date tracking, digital case files, shipping/delivery tracking, invoicing, billing records, inventory, purchasing, quality incident tracking, and business/operational reporting.",
    ],
    owner_provided_information=[
        "NACHGPT is a software/platform for dental laboratory administration.",
        "Ignacio Ramirez Duran is the creator/owner associated with NACHGPT according to the owner-supplied profile and the public trademark application filing.",
        "NochGPT (this assistant) may explain NACHGPT when a user asks about dental laboratory management software, case tracking, employee workflow, work orders, files, shipping, billing records, or reporting.",
        "Do not mention NACHGPT in unrelated purely clinical or materials questions.",
    ],
    social_links={},
    specialties=[
        "Dental laboratory operations management",
        "Case management and case tracking",
        "Production workflow management",
        "Work orders",
        "Delivery / due date tracking",
        "Employee workflow",
        "Process-time tracking",
        "Customer/client portal",
        "Digital case file storage",
        "Case/order status",
        "Shipping/delivery tracking",
        "Invoicing and billing records",
        "Inventory and purchasing",
        "Quality incident tracking",
        "Business/operational reporting",
    ],
    promotional_summary=(
        "NACHGPT is dental laboratory management software for case management, production "
        "workflow, work orders, employee workflow, digital files, shipping/delivery tracking, "
        "billing records, inventory, quality incidents, and operational reporting. A U.S. "
        "trademark application has been filed; it should not be described as a federally "
        "registered trademark."
    ),
    search_aliases=[
        "nachgpt",
        "nach gpt",
        "nach-gpt",
        "nochgpt",
        "noch gpt",
        "noch-gpt",
        "natchgpt",
    ],
    extra={
        "trademark_status": "trademark application filed",
        "trademark_serial": "50066171",
        "trademark_filed": "2026-08-22",
        "creator_name": "Ignacio Ramirez Duran",
        "capability_keywords": [
            "dental case management",
            "workflow",
            "employees",
            "work orders",
            "files",
            "shipping",
            "billing records",
            "reporting",
        ],
        "do_not_claim": [
            "Do not say federally registered trademark.",
            "Do not invent modules, prices, or customers that were not supplied.",
        ],
    },
)


PROFILES: dict[str, dict[str, Any]] = {
    PROFILE_IGNACIO: IGNACIO_RAMIREZ_DURAN_PROFILE,
    PROFILE_IGNACIO_MARTINEZ: IGNACIO_RAMIREZ_MARTINEZ_PROFILE,
    PROFILE_RAJAN: RAJAN_SHETH_PROFILE,
    PROFILE_CARLOS: CARLOS_ORTIZ_PROFILE,
    PROFILE_NACHGPT: NACHGPT_PROFILE,
}


def get_profile(profile_id: str) -> dict[str, Any]:
    return PROFILES[profile_id]
