from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Dental-LLM API")

# CORS (para que Wix o cualquier front pueda consumirlo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Puedes cambiar "*" por tu dominio Wix si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Endpoints ----------

@app.get("/")
def home():
    return "<h1>Dental-LLM corriendo en Render 🚀</h1>"

@app.get("/info")
def info():
    return {"ok": True, "model": "gpt-4o-mini", "history_lines": 0}

class Pregunta(BaseModel):
    pregunta: str

@app.post("/chat")
def chat(p: Pregunta):
    if not p.pregunta.strip():
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")
    
    # Respuesta de prueba (luego conectamos OpenAI)
    return {"respuesta": f"Recibí tu pregunta: {p.pregunta}"}
