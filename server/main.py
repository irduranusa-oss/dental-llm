# server/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import OpenAI
import os, time, re

# --- IMPORTA LA CAJITA (cache en memoria) ---
# Nota: requiere server/__init__.py para import absoluto
from server.cache import get_from_cache, save_to_cache

app = FastAPI(title="Dental-LLM API")

# --- CORS (mientras pruebas, "*"; luego pon tu dominio de Wix) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ejemplo: ["https://www.dentodo.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- OpenAI client ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ Falta OPENAI_API_KEY en variables de entorno")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Modelo (rápido y económico) ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Prompt dental (siempre responder en el mismo idioma del usuario) ---
SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise, practical, and provide ranges (e.g., temperatures or times) when relevant.
- If the question is not dental-related, politely say you are focused on dental topics and offer a helpful redirection.
- IMPORTANT: Always reply in the same language as the user's question.
"""

# --- Entrada para /chat ---
class ChatIn(BaseModel):
    pregunta: str

# --- Historial simple en memoria (opcional) ---
HIST = []      # cada item: {"t": timestamp, "pregunta": ..., "respuesta": ...}
MAX_HIST = 200

def detect_lang(text: str) -> str:
    """
    Heurística simple para la LLAVE de la cache.
    La respuesta final la controla el SYSTEM_PROMPT (mismo idioma del usuario).
    """
    t = text.lower()
    if re.search(r"[áéíóúñ¿¡]", t):
        return "es"
    if re.search(r"[ãõáéíóúç]", t):
        return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t):
        return "fr"
    return "en"

def call_openai(question: str) -> str:
    """Llama al modelo con el system prompt dental."""
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="Error con el modelo")

# --- Rutas ---
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h3>Dental-LLM corriendo ✅</h3>"

@app.post("/chat")
def chat(body: ChatIn):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")

    # 1) Idioma solo para la LLAVE de cache
    lang = detect_lang(q)

    # 2) Intentar cache
    cached = get_from_cache(q, lang)
    if cached is not None:
        return {"respuesta": cached, "cached": True}

    # 3) Llamar al modelo
    a = call_openai(q)

    # 4) Guardar cache + historial
    save_to_cache(q, lang, a)
    HIST.append({"t": time.time(), "pregunta": q, "respuesta": a})
    if len(HIST) > MAX_HIST:
        del HIST[: len(HIST) - MAX_HIST]

    return {"respuesta": a, "cached": False}

# Compatibilidad: /chat_multi sigue funcionando igual
@app.post("/chat_multi")
def chat_multi(body: ChatIn):
    return chat(body)

# Historial para tu widget
@app.get("/history")
def history(q: str = "", limit: int = 10):
    q = (q or "").strip().lower()
    out = []
    for item in reversed(HIST):
        if q and q not in item["pregunta"].lower():
            continue
        out.append(item)
        if len(out) >= max(1, min(limit, 50)):
            break
    return list(reversed(out))
