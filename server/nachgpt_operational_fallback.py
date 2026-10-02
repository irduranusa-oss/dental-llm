from __future__ import annotations

import json


def _clean(value, limit=500):
    return str(value or "").strip()[:limit]


def _lines_for_mapping(name: str, payload: dict) -> list[str]:
    out = [f"{name}:"]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out.append(f"- {key}: {_clean(value)}")
        elif isinstance(value, list):
            out.append(f"- {key}: {len(value)} item(s)")
    return out


def build_operational_fallback(question: str, fact_packet: dict, lang: str = "es") -> str:
    categories = fact_packet.get("categories") if isinstance(fact_packet, dict) else {}
    categories = categories if isinstance(categories, dict) else {}
    if not categories:
        return (
            "NO VERIFICADO: no hay categorias de evidencia disponibles."
            if lang == "es"
            else "NOT VERIFIED: no evidence categories are available."
        )

    if lang == "en":
        lines = ["Verified NACHGPT evidence:"]
    else:
        lines = ["Evidencia verificada de NACHGPT:"]

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
