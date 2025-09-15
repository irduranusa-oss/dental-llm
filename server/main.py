# server/main.py
import os, re, time, base64, mimetypes, pathlib, requests, json
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Tuple
from openai import OpenAI

# --- cache en memoria (opcional) ---
try:
    from server.cache import get_from_cache, save_to_cache
except Exception:
    def get_from_cache(*a, **k): return None
    def save_to_cache(*a, **k): return None

app = FastAPI(title="Dental-LLM API")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- OpenAI ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ Falta OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # chat + visión
OPENAI_TEMP  = float(os.getenv("OPENAI_TEMP", "0.2"))

SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise, practical, and provide ranges (e.g., temperatures or times) when relevant.
- If the question is not dental-related, politely say you are focused on dental topics and offer a helpful redirection.
- IMPORTANT: Always reply in the same language as the user's question.
"""

LANG_NAME = {
    "es": "Spanish", "en": "English", "pt": "Portuguese", "fr": "French",
}

class ChatIn(BaseModel):
    pregunta: str

HIST, MAX_HIST = [], 200

# ---------- Autodetección de idioma MEJORADA ----------
_ES_WORDS = set("""
hola gracias por favor que como cuando donde porque para con sin desde hasta muy tambien aunque pues dentadura protesis implante zirconia
""".split())
_EN_WORDS = set("""the and for with how why what when where hello thanks please""".split())
_PT_WORDS = set("""olá obrigado por favor que como quando onde porque para com sem""".split())
_FR_WORDS = set("""bonjour merci s'il vous plaît que comment quand où pourquoi pour avec sans""".split())

def _score_lang(text: str) -> Tuple[str,int]:
    # Cuenta palabras comunes (stopwords sencillas)
    t = re.sub(r"[^a-záéíóúñüçãõâêôàèìòùœ\- ]", " ", text.lower())
    tokens = set(t.split())
    scores = {
        "es": len(tokens & _ES_WORDS),
        "en": len(tokens & _EN_WORDS),
        "pt": len(tokens & _PT_WORDS),
        "fr": len(tokens & _FR_WORDS),
    }
    best = max(scores, key=scores.get)
    return best, scores[best]

def detect_lang(text: str) -> str:
    t = (text or "").strip().lower()
    if not t: return "en"
    # 1) pistas por acentos
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõçâêô]", t):    return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    # 2) conteo de palabras comunes
    best, score = _score_lang(t)
    if score >= 1:
        return best
    # 3) fallback por defecto
    return "en"

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
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="Error con el modelo")

# --------- Utilidades de media (imagen/pdf/audio) ----------
def _mime_from_path(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

def _to_data_url(path: str) -> str:
    mime = _mime_from_path(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def analyze_image_with_openai(image_path: str, extra_prompt: str = "") -> str:
    user_msg = [
        {"type":"text","text":"Analiza brevemente esta imagen desde el punto de vista dental. Sé conciso." +
         (f"\nContexto del usuario: {extra_prompt}" if extra_prompt else "")},
        {"type":"image_url","image_url":{"url":_to_data_url(image_path)}},
    ]
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":user_msg}],
            temperature=OPENAI_TEMP,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print("Vision error:", e)
        return "Recibí tu imagen, pero no pude analizarla en este momento."

def transcribe_audio_with_openai(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        return (tr.text or "").strip()
    except Exception as e1:
        print("whisper-1 falló, intento gpt-4o-mini-transcribe:", e1)
        try:
            with open(audio_path, "rb") as f:
                tr = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
            return (tr.text or "").strip()
        except Exception as e2:
            print("Transcripción falló:", e2)
            return ""

# ---------------- Rutas básicas ----------------
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h3>Dental-LLM corriendo ✅</h3>"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(body: ChatIn):
    q = (body.pregunta or "").strip()
    if not q: raise HTTPException(status_code=400, detail="Falta 'pregunta'")
    lang = detect_lang(q)
    cached = get_from_cache(q, lang)
    if cached is not None:
        return {"respuesta": cached, "cached": True}
    a = call_openai(q, lang_hint=lang)
    save_to_cache(q, lang, a)
    HIST.append({"t": time.time(), "pregunta": q, "respuesta": a})
    if len(HIST) > MAX_HIST: del HIST[:len(HIST)-MAX_HIST]
    return {"respuesta": a, "cached": False}

@app.get("/history")
def history(q: str = "", limit: int = 10):
    q = (q or "").strip().lower()
    out = []
    for item in reversed(HIST):
        if q and q not in item["pregunta"].lower(): continue
        out.append(item)
        if len(out) >= max(1, min(limit, 50)): break
    return list(reversed(out))

# ===================== WhatsApp =====================
WHATSAPP_TOKEN     = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID  = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN  = os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123")
FB_API             = "https://graph.facebook.com/v20.0"
SHEETS_WEBHOOK_URL = os.getenv("SHEETS_WEBHOOK_URL", "")

def _e164_no_plus(num: str) -> str:
    num = (num or "").strip().replace(" ", "").replace("-", "")
    return num[1:] if num.startswith("+") else num

def _wa_url() -> str:
    return f"{FB_API}/{WHATSAPP_PHONE_ID}/messages"

def wa_send_text(to_number: str, body: str):
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_ID):
        print("⚠️ Falta WHATSAPP_TOKEN o WHATSAPP_PHONE_ID")
        return
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    data = {
        "messaging_product":"whatsapp", "to": _e164_no_plus(to_number),
        "type":"text", "text":{"preview_url":False, "body": body[:3900]}
    }
    try:
        requests.post(_wa_url(), headers=headers, json=data, timeout=20)
    except Exception as e:
        print("wa_send_text error:", e)

def wa_get_media_url(media_id: str) -> str:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(f"{FB_API}/{media_id}", headers=headers, timeout=15); r.raise_for_status()
    return (r.json() or {}).get("url", "")

def wa_download_media(signed_url: str, dest_prefix: str="/tmp/wa_media/") -> Tuple[str,str]:
    pathlib.Path(dest_prefix).mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(signed_url, headers=headers, stream=True, timeout=30); r.raise_for_status()
    mime = r.headers.get("Content-Type", "application/octet-stream")
    ext  = mimetypes.guess_extension(mime) or ""
    path = os.path.join(dest_prefix, f"{int(time.time())}{ext}")
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    return path, mime

# --------- Ticket a Google Sheets (Apps Script) ----------
def send_ticket_to_sheets(payload: dict):
    if not SHEETS_WEBHOOK_URL:
        print("⚠️ Falta SHEETS_WEBHOOK_URL (no se envió ticket)")
        return
    try:
        requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=15)
    except Exception as e:
        print("Ticket error:", e)

# --- Verificación GET ---
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode      = request.query_params.get("hub.mode", "")
    token     = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")
    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="forbidden", status_code=403)

# --- Recepción POST ---
@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"received": False, "error": "invalid_json"})

    print("📩 Payload recibido:", data)
    try:
        entry   = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        value   = changes.get("value") or {}

        # A) Mensajes nuevos
        msgs = value.get("messages") or []
        if msgs:
            msg         = msgs[0]
            from_number = msg.get("from")
            mtype       = msg.get("type")

            # 1) Texto
            if mtype == "text":
                user_text = (msg.get("text") or {}).get("body", "").strip()
                if user_text and from_number:
                    lang    = detect_lang(user_text)
                    answer  = call_openai(user_text, lang_hint=lang)
                    wa_send_text(from_number, answer)
                    # ticket
                    send_ticket_to_sheets({
                        "ts": int(time.time()),
                        "from": from_number,
                        "nombre": "",
                        "tema": "",
                        "contacto": from_number,
                        "horario": "",
                        "mensaje": user_text,
                        "label": "NochGPT"
                    })
                return {"status":"ok"}

            # 2) Imagen
            if mtype == "image":
                img = msg.get("image") or {}
                media_id = img.get("id"); caption = (img.get("caption") or "").strip()
                if media_id and from_number:
                    try:
                        url = wa_get_media_url(media_id)
                        path, mime = wa_download_media(url)
                        analysis = analyze_image_with_openai(path, caption)
                        wa_send_text(from_number, f"🖼️ {analysis}")
                        send_ticket_to_sheets({
                            "ts": int(time.time()), "from": from_number,
                            "nombre":"", "tema":"imagen", "contacto":from_number,
                            "horario":"", "mensaje": caption or "(imagen)", "label":"NochGPT"
                        })
                    except Exception as e:
                        print("Error imagen:", e)
                        wa_send_text(from_number, "No pude analizar la imagen.")
                return {"status":"ok"}

            # 3) Audio (nota de voz)
            if mtype == "audio":
                aud = msg.get("audio") or {}; media_id = aud.get("id")
                if media_id and from_number:
                    try:
                        url = wa_get_media_url(media_id)
                        path, mime = wa_download_media(url)
                        transcript = transcribe_audio_with_openai(path)
                        if transcript:
                            lang   = detect_lang(transcript)
                            answer = call_openai(
                                f"Transcripción del audio del usuario:\n\"\"\"{transcript}\"\"\"\n\n"
                                "Responde de forma útil y breve, enfocada en odontología cuando aplique.",
                                lang_hint=lang
                            )
                            wa_send_text(from_number, f"🗣️ {transcript}\n\n{answer}")
                            send_ticket_to_sheets({
                                "ts": int(time.time()), "from": from_number,
                                "nombre":"", "tema":"audio", "contacto":from_number,
                                "horario":"", "mensaje": transcript, "label":"NochGPT"
                            })
                        else:
                            wa_send_text(from_number, "No pude transcribir el audio.")
                    except Exception as e:
                        print("Error audio:", e)
                        wa_send_text(from_number, "No pude procesar el audio.")
                return {"status":"ok"}

            # 4) Documento (ignoramos PDFs para simplificar)
            if from_number:
                wa_send_text(from_number, "Recibí tu mensaje. Por ahora manejo texto e incluso notas de voz.")
                return {"status":"ok"}

        # B) Status
        if value.get("statuses"):
            return {"status":"status_ok"}

        return {"status":"no_message"}
    except Exception as e:
        print("❌ Error en webhook:", e)
        return {"status":"error"}
