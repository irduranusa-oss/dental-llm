from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Dental-LLM API")

# Configuración CORS (para que funcione en Wix o local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # puedes cambiar * por tu dominio Wix si quieres
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------- Rutas -----------

@app.get("/")
def home():
    return "Dental-LLM corriendo en Render 🚀"

@app.get("/info")
def info():
    return {"ok": True, "model": "gpt-4o-mini", "history_lines": 0}

class Pregunta(BaseModel):
    pregunta: str

@app.post("/chat")
def chat(p: Pregunta):
    if not p.pregunta.strip():
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")
    
    # Respuesta provisional (luego conectamos con OpenAI)
    return {"respuesta": f"Recibí tu pregunta: {p.pregunta}"}
