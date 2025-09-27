from fastapi import FastAPI, Query
from pydantic import BaseModel
import requests

app = FastAPI()

# ==========================================
# MODELOS DE ENTRADA Y RESPUESTA
# ==========================================

class ToolCallRequest(BaseModel):
    query: str
    lang: str = "es"
    top_k: int = 3
    max_chars: int = 500

class ToolCallResponse(BaseModel):
    query: str
    language: str
    candidates: list

# ==========================================
# FUNCIÓN PRINCIPAL DE CONSULTA A WIKIPEDIA
# ==========================================

@app.post("/tool")
def tool(req: ToolCallRequest):
    lang = req.lang
    query = req.query
    top_k = req.top_k
    max_chars = req.max_chars

    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "format": "json",
        "srlimit": top_k
    }

    response = requests.get(url, params=params)
    data = response.json()

    candidates = []
    for r in data.get("query", {}).get("search", []):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        snippet_clean = snippet.replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        snippet_trimmed = snippet_clean[:max_chars]
        candidates.append({
            "title": title,
            "snippet": snippet_trimmed,
            "chunks": [snippet_trimmed]
        })

    return ToolCallResponse(query=query, language=lang, candidates=candidates)

# ==========================================
# ESPECIFICACIÓN DE LA HERRAMIENTA
# ==========================================

TOOL_SPEC = {
    "name": "wikipedia_tool",
    "description": "Busca artículos en Wikipedia y devuelve texto relevante en el idioma especificado.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Tema o pregunta a buscar"},
            "lang": {"type": "string", "description": "Código de idioma (es, en, fr, etc.)"},
            "top_k": {"type": "integer", "description": "Número de resultados a devolver"},
            "max_chars": {"type": "integer", "description": "Máximo de caracteres por resultado"},
        },
        "required": ["query"]
    }
}

@app.get("/tool/spec")
def tool_spec():
    return TOOL_SPEC

# ==========================================
# INICIO DEL SERVIDOR (para Render o local)
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
