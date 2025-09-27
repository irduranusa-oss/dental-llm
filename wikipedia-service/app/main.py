from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Wikipedia Tool está activo"}

@app.get("/health")
def health():
    return {"ok": True, "message": "service alive"}

class ToolCallRequest(BaseModel):
    query: str
    lang: str = "es"
    top_k: int = 3
    max_chars: int = 500

@app.post("/tool")
def tool(req: ToolCallRequest):
    url = f"https://{req.lang}.wikipedia.org/w/api.php"
    params = {"action":"query","list":"search","srsearch":req.query,"utf8":1,"format":"json","srlimit":req.top_k}
    data = requests.get(url, params=params, timeout=15).json()
    cands=[]
    for hit in data.get("query",{}).get("search",[]):
        s=(hit.get("snippet","").replace('<span class="searchmatch">',"").replace("</span>",""))[:req.max_chars]
        cands.append({"title":hit.get("title",""),"snippet":s,"chunks":[s]})
    return {"query": req.query, "language": req.lang, "candidates": cands}

TOOL_SPEC = {
    "name": "wikipedia_tool",
    "description": "Busca artículos en Wikipedia y devuelve texto relevante.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "lang": {"type": "string", "default": "es"},
            "top_k": {"type": "integer", "default": 3},
            "max_chars": {"type": "integer", "default": 500}
        },
        "required": ["query"]
    }
}

@app.get("/tool/spec")
def tool_spec():
    return TOOL_SPEC

# (opcional solo para probar GET en navegador)
@app.get("/tool")
def tool_get(query: str, lang: str = "es", top_k: int = 3, max_chars: int = 500):
    return tool(ToolCallRequest(query=query, lang=lang, top_k=top_k, max_chars=max_chars))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
