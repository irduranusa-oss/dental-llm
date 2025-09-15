from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os, time, re, requests, mimetypes, base64, pathlib

# --- IMPORTA LA CAJITA (cache en memoria) ---
from server.cache import get_from_cache, save_to_cache

app = FastAPI(title="Dental-LLM API")

# ----------------------------
# CORS
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OpenAI client
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ Falta OPENAI_API_KEY en variables de entorno")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMP = float(os.getenv("OPENAI_TEMP", "0.2"))

SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise, practical, and provide ranges (e.g., temperatures or times) when relevant.
- If the question is not dental-related, politely say you are focused on dental topics and offer a helpful redirection.
- IMPORTANT: Always reply in the same language as the user's question.
"""

# ---- Mapeo de códigos a nombres ----
LANG_NAME = {
    "es": "Spanish",
    "en": "English",
    "pt": "Portuguese",
    "fr": "French",
    "ar": "Arabic",
    "hi": "Hindi",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

class ChatIn(BaseModel):
    pregunta: str

# Historial en memoria
HIST = []
MAX_HIST = 200

# ----------------------------
# Detección de idioma
# ----------------------------
def detect_lang(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõáéíóúç]", t): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    if re.search(r"[\u0600-\u06FF]", t): return "ar"
    if re.search(r"[\u0900-\u097F]", t): return "hi"
    if re.search(r"[\u4e00-\u9fff]", t): return "zh"
    if re.search(r"[\u3040-\u309F\u30A0-\u30FF]", t): return "ja"
    if re.search(r"[\uac00-\ud7af]", t): return "ko"
    return "en"

# ----------------------------
# Llamada al modelo con traducción forzada si es necesario
# ----------------------------
def call_openai(question: str, lang_hint: str | None = None) -> str:
    sys = SYSTEM_PROMPT
    if lang_hint in LANG_NAME:
        sys += f"\n- The user's language is {LANG_NAME[lang_hint]}. Always reply in {LANG_NAME[lang_hint]}."

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": question},
            ],
            temperature=OPENAI_TEMP,
        )
        answer = (resp.choices[0].message.content or "").strip()

        # Si usuario habló en otro idioma y respuesta salió en inglés → traducir
        if lang_hint and lang_hint != "en":
            if re.match(r"^[a-zA-Z0-9 ,.\-!?()]+$", answer[:100]):  
                try:
                    tr = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": f"Traduce al {LANG_NAME[lang_hint]} sin agregar explicaciones."},
                            {"role": "user", "content": answer},
                        ],
                        temperature=0,
                    )
                    answer = (tr.choices[0].message.content or "").strip()
                except:
                    pass
        return answer
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="Error con el modelo")

# ======================================================================
# RUTAS BÁSICAS
# ======================================================================
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h3>Dental-LLM corriendo ✅</h3>"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(body: ChatIn):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")

    lang = detect_lang(q)
    cached = get_from_cache(q, lang)
    if cached is not None:
        return {"respuesta": cached, "cached": True}

    a = call_openai(q, lang_hint=lang)
    save_to_cache(q, lang, a)

    HIST.append({"t": time.time(), "pregunta": q, "respuesta": a})
    if len(HIST) > MAX_HIST:
        del HIST[: len(HIST) - MAX_HIST]

    return {"respuesta": a, "cached": False}

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
