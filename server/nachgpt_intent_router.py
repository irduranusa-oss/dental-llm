from __future__ import annotations

import re
import unicodedata

INTENT_CASE = "case"
INTENT_CASE_OVERVIEW = "case_overview"
INTENT_SENT_CASES_OVERVIEW = "sent_cases_overview"
INTENT_PROCESSOR = "processor"
INTENT_STORAGE = "storage"
INTENT_BACKUP = "backup"
INTENT_HYPERDENT = "hyperdent"
INTENT_BLENDER = "blender"
INTENT_EXOCAD = "exocad"
INTENT_INVENTORY = "inventory"
INTENT_PURCHASES = "purchases"
INTENT_MACHINES = "machines"
INTENT_PRODUCTION = "production"
INTENT_QUALITY = "quality"
INTENT_DELIVERY = "delivery"
INTENT_FINANCE = "finance"
INTENT_EXECUTIVE = "executive"
INTENT_EMPLOYEES = "employees"
INTENT_CLIENTS = "clients"
INTENT_PLATFORM = "platform"

_INTENT_RULES = (
    (INTENT_SENT_CASES_OVERVIEW, ("casos enviados", "casos que se enviaron", "casos despachados", "shipped cases", "cases shipped", "envios de esta semana"), ("sent_cases_overview",)),
    (INTENT_CASE_OVERVIEW, ("casos creados", "casos nuevos", "casos recibidos", "casos ingresados", "casos de esta semana", "cases created", "new cases this week"), ("case_overview",)),
    (INTENT_PROCESSOR, ("procesador", "processor", "gmail", "correo", "email intake", "ultima corrida", "última corrida"), ("processor", "intake")),
    (INTENT_STORAGE, ("r2", "archivo fisico", "archivo físico", "object key", "storage", "guardado en nube", "archivos en nube"), ("r2_evidence",)),
    (INTENT_BACKUP, ("dropbox", "backup", "respaldo", "oauth", "refresh token", "access token"), ("backup", "r2_evidence")),
    (INTENT_HYPERDENT, ("hyperdent", "hdproj", "hdprojz", "nesting"), ("hyperdent", "production", "case_files")),
    (INTENT_BLENDER, ("blender", "b4d", "ibar", "hybrid", "tubos"), ("blender", "production", "case_files")),
    (INTENT_EXOCAD, ("exocad",), ("exocad", "production", "case_files")),
    (INTENT_PURCHASES, ("compras", "ordenes de compra", "orden de compra", "purchases", "purchase orders", "purchase order", "recepcion de compras"), ("purchases",)),
    (INTENT_INVENTORY, ("inventario", "inventory", "zirconia", "pmma", "titania", "material"), ("inventory",)),
    (INTENT_MACHINES, ("maquina", "máquina", "machines", "fresadora", "milling", "imes", "roland", "smill"), ("machines", "production")),
    (INTENT_QUALITY, ("calidad", "quality", "remake", "repeticion", "repetición", "incidencia"), ("quality",)),
    (INTENT_DELIVERY, ("entrega", "delivery", "vencido", "overdue", "tracking", "envio", "envío"), ("delivery",)),
    (INTENT_FINANCE, ("facturacion", "facturación", "factura", "invoice", "cobranza", "saldo", "pago", "finance", "financial"), ("finance",)),
    (INTENT_EXECUTIVE, ("ceo", "reporte ejecutivo", "executive report", "ceo pulse", "resumen ejecutivo"), ("executive_reports",)),
    (INTENT_EMPLOYEES, ("acciones de trabajadores", "acciones de empleados", "actividad de trabajadores", "actividad de empleados", "trabajador", "trabajadores", "empleado", "empleados", "employee", "employees"), ("employee_activity",)),
    (INTENT_CLIENTS, ("clientes", "clients", "clinicas", "clínicas", "doctores", "doctors"), ("clients",)),
    (INTENT_PLATFORM, ("plataforma", "platform", "todos los laboratorios", "all laboratories"), ("platform_health",)),
    (INTENT_PRODUCTION, ("produccion", "producción", "production", "fase", "phase", "cola", "queue", "finalizado"), ("production",)),
    (INTENT_CASE, ("caso", "case", "paciente", "patient", "ficha del caso", "case master"), ("case",)),
)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", normalized).strip().lower()


def _term_matches(folded: str, term: str) -> bool:
    token = _fold(term)
    if not token:
        return False
    if " " in token:
        return token in folded
    return bool(re.search(r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])", folded))


def _is_author_book_question(folded: str) -> bool:
    # A request about an author's general method is not a request for case data.
    # No access grant: contract validation still requires a verified training fact.
    personal = ("mi libro","mi workflow","mi flujo de trabajo","mi protocolo",
                "mi criterio profesional","mi personalidad","criterio del autor",
                "protocolo del autor","libro aox","libro all-on-x","metodo de ignacio",
                "mis seis capitulos")
    return any(term in folded for term in personal) and not any(
        term in folded for term in ("este caso","caso actual","caso del paciente")
    )


def route_nachgpt_question(question: str) -> dict:
    folded = _fold(question)
    if _is_author_book_question(folded):
        return {
            "operational":True, "intents":["author_workflow"],
            "required_categories":["training"], "matched_terms":["author_workflow"],
        }
    intents: list[str] = []
    required: list[str] = []
    matched_terms: list[str] = []

    for intent, terms, categories in _INTENT_RULES:
        hit = next((term for term in terms if _term_matches(folded, term)), "")
        if not hit:
            continue
        intents.append(intent)
        matched_terms.append(hit)
        for category in categories:
            if category not in required:
                required.append(category)

    if INTENT_EMPLOYEES in intents and "employee_activity" in required:
        required = [category for category in required if category != "case"]
        intents = [intent for intent in intents if intent != INTENT_CASE]

    if INTENT_CASE in intents and any(x in intents for x in (INTENT_STORAGE, INTENT_HYPERDENT, INTENT_BLENDER, INTENT_EXOCAD)):
        if "case_files" not in required:
            required.append("case_files")

    return {
        "operational": bool(intents),
        "intents": intents,
        "required_categories": required,
        "matched_terms": matched_terms,
    }


def missing_required_categories(question: str, categories: dict | None, *, role: str = "") -> list[str]:
    routed = route_nachgpt_question(question)
    required = list(routed["required_categories"])
    if str(role or "").upper() == "CLIENT":
        # NACHGPT sends only authorized OWN_CASE evidence, not general case facts.
        # Do not remap finance/processor requests: those remain forbidden.
        own_case_map = {
            "case": "own_case",
            "case_files": "own_case_files",
            "r2_evidence": "own_case_files",
            "delivery": "own_case_tracking",
        }
        required = [own_case_map.get(name, name) for name in required]
    if "employee_activity" in required:
        required = [name for name in required if name != "case"]
    present = set((categories or {}).keys()) if isinstance(categories, dict) else set()
    return sorted({name for name in required if name not in present})
