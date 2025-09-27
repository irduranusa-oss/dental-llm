from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Wikipedia Tool está activo"}

@app.get("/tool/spec")
def tool_spec():
    return {
        "name": "wikipedia_tool",
        "description": "Busca artículos en Wikipedia y devuelve texto resumido.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tema a buscar"},
                "lang": {"type": "string", "description": "Código de idioma"},
                "top_k": {"type": "integer", "description": "Cantidad de artículos"},
                "max_chars": {"type": "integer", "description": "Máx. caracteres por resultado"},
            },
            "required": ["query"]
        }
    }
