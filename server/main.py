from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI(title="Dental-LLM API")

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Puedes cambiar "*" por tu dominio Wix
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

# Conexión con OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

@app.post("/chat")
def chat(p: Pregunta):
    if not p.pregunta.strip():
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")
    if not client.api_key:
        return {"respuesta": "⚠️ Falta configurar OPENAI_API_KEY en Render"}

    system_prompt = (
        "Eres NochGPT, asistente dental práctico para un técnico protésico senior. "
        "Responde en español, claro, breve y con pasos accionables."
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{p.pregunta}"
        )
        texto = resp.output_text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error OpenAI: {e}")

    return {"respuesta": texto}
