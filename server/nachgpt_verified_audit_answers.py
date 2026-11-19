"""Deterministic answers to three high-impact operational audit questions.

All figures come exclusively from contract-validated facts. No LLM guesswork,
write privileges, or substitution of generated reports for proof of dispatch.
"""
from __future__ import annotations

from server.nachgpt_intent_router import _fold


def verified_audit_answer(question: str, categories: dict, lang: str = "es") -> str | None:
    q = _fold(question)
    cats = categories if isinstance(categories, dict) else {}
    spanish = not str(lang).lower().startswith("en")

    if ("ceo" in q and any(word in q for word in ("reporte", "report")) and
        any(word in q for word in ("salio", "enviado", "envio", "send", "sent", "delivered", "hoy", "today"))):
        executive = cats.get("executive_reports") or {}
        delivery = executive.get("delivery_audit") if isinstance(executive, dict) else {}
        if not isinstance(delivery, dict) or not delivery.get("ok"):
            return "NO VERIFICADO: falta el registro de envío al CEO." if spanish else "NOT VERIFIED: CEO send log is unavailable."
        day = str(delivery.get("date") or "")
        events = [e for e in delivery.get("events") or [] if isinstance(e, dict) and e.get("smtp_accepted")]
        if events:
            lines = [f"Sí. El registro de envíos de LAB001 confirma {len(events)} envío(s) al CEO el {day}." if spanish
                     else f"Yes. The LAB001 outbound log confirms {len(events)} CEO report send(s) on {day}."]
            for item in events[:8]:
                lines.append(f"- {item.get('report_type')}: período {item.get('period_start')} a {item.get('period_end')}; envío {item.get('sent_at_local')}.")
            lines.append("Confirmación del transporte de correo; no comprueba apertura o lectura." if spanish
                         else "Outbound acceptance only; inbox receipt and reading are not verified.")
            return "\n".join(lines)
        return (f"NO VERIFICADO: no se encontró un envío exitoso al CEO el {day} en el registro consultado. "
                "La creación del informe no demuestra su envío." if spanish else
                f"NOT VERIFIED: no confirmed CEO send on {day}; generating a report does not prove delivery.")

    if ("employee_activity" in cats and
        any(term in q for term in ("acciones de los trabajadores", "actividad de empleados", "acciones de empleados",
                                  "actividad de trabajadores", "employee activity", "employee actions"))):
        activity = cats.get("employee_activity") or {}
        if not isinstance(activity, dict) or not activity.get("ok"):
            return "NO VERIFICADO: falta actividad del personal." if spanish else "NOT VERIFIED: missing employee activity."
        start = str(activity.get("period_start") or activity.get("day") or "")
        end = str(activity.get("period_end") or activity.get("day") or "")
        rows = activity.get("actions") if isinstance(activity.get("actions"), list) else []
        count = activity.get("action_count")
        lines = [f"Acciones registradas en LAB001 del {start} al {end}: {count}." if spanish
                 else f"LAB001 recorded employee actions from {start} through {end}: {count}."]
        for item in rows[:100]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('employee_name') or item.get('employee_id') or '?'} | "
                         f"{item.get('timestamp') or '?'} | {item.get('action') or '?'} | "
                         f"{item.get('case_name') or '?'} | estado: {item.get('current_case_status') or 'NO VERIFICADO'}")
        lines.append(f"Fuente: {activity.get('source') or 'NO VERIFICADO'}. La ausencia de registros no prueba ausencia de trabajo.")
        return "\n".join(lines)

    if "sent_cases_overview" in cats and any(term in q for term in
            ("casos enviados", "casos despachados", "shipped cases", "cases shipped")):
        overview = cats.get("sent_cases_overview") or {}
        if not isinstance(overview, dict) or not overview.get("ok"):
            return "NO VERIFICADO: falta el historial de casos enviados."
        start = overview.get("period_start")
        end = overview.get("period_end")
        count = overview.get("sent_count")
        lines = [f"Casos enviados de LAB001 del {start} al {end}: {count if count is not None else 'conteo incompleto'}." ]
        for item in (overview.get("cases") or [])[:100]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('case_name') or '?'} | enviado: {item.get('sent_date') or '?'} | "
                         f"rastreo: {item.get('tracking') or 'NO VERIFICADO'} | "
                         f"estado: {item.get('current_status') or 'NO VERIFICADO'}")
        if overview.get("truncated"):
            lines.append("Listado parcial; se necesita paginación para consultar todos los envíos.")
        lines.append("Fuente: casos enviados canónicos (R2/PostgreSQL), fecha de envío corroborada.")
        return "\n".join(lines)

    if "case_overview" in cats and any(term in q for term in
            ("casos creados", "casos nuevos", "casos de esta semana", "cases created", "new cases")):
        overview = cats.get("case_overview") or {}
        if not isinstance(overview, dict) or not overview.get("ok"):
            return "NO VERIFICADO: falta el listado canónico de casos." if spanish else "NOT VERIFIED: canonical case list missing."
        start = overview.get("period_start")
        end = overview.get("period_end")
        count = overview.get("created_count")
        lines = [f"Casos creados en LAB001 del {start} al {end}: {count if count is not None else 'conteo incompleto'}." if spanish
                 else f"Cases created in LAB001 from {start} through {end}: {count if count is not None else 'incomplete count'}."]
        for item in (overview.get("cases") or [])[:100]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- {item.get('case_name') or '?'} | {item.get('patient_name') or '?'} | "
                         f"estado actual: {item.get('current_status') or 'NO VERIFICADO'} | "
                         f"creado: {item.get('created_at') or '?'}")
        if overview.get("truncated"):
            lines.append("Listado parcial; no se puede asegurar un total exacto.")
        lines.append("Fuente: casos PostgreSQL, fecha de creación. La cobertura histórica podría ser incompleta.")
        return "\n".join(lines)
    return None
