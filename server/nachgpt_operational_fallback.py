from __future__ import annotations

import json


def _clean(value, limit=500):
    if value is None:
        return ""
    return str(value).strip()[:limit]


def _yes_no(value, lang="es"):
    if lang == "en":
        return "Yes" if bool(value) else "No"
    return "Sí" if bool(value) else "No"


def _none_label(lang="es"):
    return "none" if lang == "en" else "ninguno"


def _lines_for_mapping(name: str, payload: dict) -> list[str]:
    out = [f"{name}:"]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out.append(f"- {key}: {_clean(value)}")
        elif isinstance(value, list):
            out.append(f"- {key}: {len(value)} item(s)")
    return out


def _processor_run_summary(categories: dict, lang: str = "es") -> list[str]:
    run = categories.get("processor_run") if isinstance(categories.get("processor_run"), dict) else {}
    processor = categories.get("processor") if isinstance(categories.get("processor"), dict) else {}
    intake = categories.get("intake") if isinstance(categories.get("intake"), dict) else {}
    if not run:
        return []

    case_names = [str(x).strip() for x in (run.get("case_names") or []) if str(x).strip()]
    file_rows = run.get("technical_sheet_files") if isinstance(run.get("technical_sheet_files"), list) else []
    error = _clean(run.get("error")) or _none_label(lang)
    message = _clean(run.get("message")) or _none_label(lang)
    overall = _clean(intake.get("overall") or processor.get("overall")) or ("UNKNOWN" if lang == "en" else "DESCONOCIDO")

    if lang == "en":
        lines = [
            "Verified processor audit:",
            f"- Processor / intake health: {overall}",
            f"- Run ID: {_clean(run.get('run_id'))}",
            f"- Run status: {_clean(run.get('status'))}",
            f"- Started: {_clean(run.get('started_at'))}",
            f"- Finished: {_clean(run.get('finished_at'))}",
            f"- Emails processed: {_clean(run.get('processed'))}",
            f"- New cases: {_clean(run.get('new_cases'))}",
            "- Cases created: " + (", ".join(case_names) if case_names else "none"),
            f"- Technical-sheet files: {_clean(run.get('technical_sheet_file_count'))}",
            f"- Already-processed emails skipped: {_clean(run.get('skipped_already_processed'))}",
            f"- Existing-case skips: {_clean(run.get('skipped_existing_case'))}",
            f"- Repaired cases: {len(run.get('repaired_cases') or [])}",
            f"- Partial run: {_yes_no(run.get('partial'), lang)}",
            f"- Time budget hit: {_yes_no(run.get('time_budget_hit'), lang)}",
            f"- Registered error: {error}",
            f"- Processor message: {message}",
        ]
    else:
        lines = [
            "Auditoría verificada del procesador:",
            f"- Salud procesador / intake: {overall}",
            f"- Corrida ID: {_clean(run.get('run_id'))}",
            f"- Estado de la corrida: {_clean(run.get('status'))}",
            f"- Inicio: {_clean(run.get('started_at'))}",
            f"- Fin: {_clean(run.get('finished_at'))}",
            f"- Correos procesados: {_clean(run.get('processed'))}",
            f"- Casos nuevos: {_clean(run.get('new_cases'))}",
            "- Casos creados: " + (", ".join(case_names) if case_names else "ninguno"),
            f"- Archivos disponibles en ficha técnica: {_clean(run.get('technical_sheet_file_count'))}",
            f"- Correos omitidos por ya procesados: {_clean(run.get('skipped_already_processed'))}",
            f"- Casos omitidos por existir previamente: {_clean(run.get('skipped_existing_case'))}",
            f"- Casos reparados: {len(run.get('repaired_cases') or [])}",
            f"- Corrida parcial: {_yes_no(run.get('partial'), lang)}",
            f"- Límite de tiempo alcanzado: {_yes_no(run.get('time_budget_hit'), lang)}",
            f"- Error registrado: {error}",
            f"- Mensaje del procesador: {message}",
        ]

    if file_rows:
        names = []
        for row in file_rows[:20]:
            if isinstance(row, dict):
                label = _clean(row.get("file_name") or row.get("name") or row.get("key"))
            else:
                label = _clean(row)
            if label:
                names.append(label)
        if names:
            lines.append(("Technical-sheet files: " if lang == "en" else "Archivos en ficha técnica: ") + ", ".join(names))
    return lines


def build_operational_fallback(question: str, fact_packet: dict, lang: str = "es") -> str:
    categories = fact_packet.get("categories") if isinstance(fact_packet, dict) else {}
    categories = categories if isinstance(categories, dict) else {}
    if not categories:
        return (
            "NO VERIFICADO: no hay categorias de evidencia disponibles."
            if lang == "es"
            else "NOT VERIFIED: no evidence categories are available."
        )

    processor_summary = _processor_run_summary(categories, lang)
    if processor_summary:
        lines = processor_summary
    elif lang == "en":
        lines = ["Verified NACHGPT evidence:"]
    else:
        lines = ["Evidencia verificada de NACHGPT:"]

    if not processor_summary:
        for category, payload in categories.items():
            if isinstance(payload, dict):
                lines.extend(_lines_for_mapping(str(category), payload))
            else:
                lines.append(f"{category}: {_clean(payload)}")

    sources = fact_packet.get("sources") if isinstance(fact_packet.get("sources"), list) else []
    if sources:
        if lang == "en":
            lines.append("Sources: " + ", ".join(_clean(x, 120) for x in sources))
        else:
            lines.append("Fuentes: " + ", ".join(_clean(x, 120) for x in sources))

    if lang == "en":
        lines.append("No unsupported causes or conclusions were inferred.")
    else:
        lines.append("No se infirieron causas ni conclusiones no respaldadas.")

    return "\n".join(lines)
