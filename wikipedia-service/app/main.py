from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# ---------- salud ----------
@app.get("/health")
def health():
    return {"ok": True, "message": "service alive"}

# ---------- modelos ----------
class ToolCallRequest(BaseModel):
    query: str
    lang: str = "es"
    top_k: int = 3
    max_chars: int = 500

class ToolCallResponse(BaseModel):
    query: str
    language: str
    candidates: list

# ---------- endpoint principal /tool ----------
@app.post("/tool")
def tool(req: ToolCallRequest):
    lang = req.lang or "es"
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": req.query,
        "utf8": 1,
        "format": "json",
        "srlimit": req.top_k
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()

    cands = []
    for hit in data.get("query", {}).get("search", []):
        title = hit.get("title", "")
        snippet = (hit.get("snippet", "") or "").replace(
            '<span class="searchmatch">', ""
        ).replace("</span>", "")
        snippet = snippet[: req.max_chars]
        cands.append({"title": title, "snippet": snippet, "chunks": [snippet]})

    return ToolCallResponse(query=req.query, language=lang, candidates=cands)

# ---------- spec de herramienta ----------
TOOL_SPEC = {
    "name": "wikipedia_tool",
    "description": "Busca artículos en Wikipedia y devuelve texto relevante.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Tema a buscar"},
            "lang": {"type": "string", "description": "Idioma (es, en, fr, ...)", "default": "es"},
            "top_k": {"type": "integer", "description": "Resultados", "default": 3},
            "max_chars": {"type": "integer", "description": "Máx. caracteres", "default": 500}
        },
        "required": ["query"]
    }
}

@app.get("/tool/spec")
def tool_spec():
    return TOOL_SPEC

# ---------- ejecutar local/render ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
