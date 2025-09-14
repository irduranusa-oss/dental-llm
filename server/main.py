# server/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os, time, re, requests, mimetypes

# --- IMPORTA LA CAJITA (cache en memoria) ---
# Nota: requiere server/__init__.py para import absoluto
from server.cache import get_from_cache, save_to_cache

app = FastAPI(title="Dental-LLM API")

# ----------------------------
# CORS (en pruebas = "*")
# Luego fija tu dominio (p. ej. "https://www.dentodo.com")
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

SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise, practical, and provide ranges (e.g., temperatures or times) when relevant.
- If the question is not dental-related, politely say you are focused on dental topics and offer a helpful redirection.
- IMPORTANT: Always reply in the same language as the user's question.
"""

class ChatIn(BaseModel):
    pregunta: str

# Historial simple en memoria
HIST = []      # cada item: {"t": timestamp, "pregunta": ..., "respuesta": ...}
MAX_HIST = 200

def detect_lang(text: str) -> str:
    """
    Heurística SOLO para la llave de la cache.
    El idioma final lo fuerza el SYSTEM_PROMPT.
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

# ----------------------------
# Rutas base
# ----------------------------
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

    # 1) Idioma solo para cache
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

# Compatibilidad
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

# ======================================================================
#                         WHATSAPP WEBHOOK
#   - GET  /webhook : verificación (hub.challenge)
#   - POST /webhook : recepción y auto-respuesta (texto, botones, imagen, documento)
# ======================================================================

WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123")

FB_API  = "https://graph.facebook.com/v20.0"

def _e164_no_plus(num: str) -> str:
    """Normaliza a E.164 sin '+' (Meta acepta sin '+')."""
    num = (num or "").strip().replace(" ", "").replace("-", "")
    return num[1:] if num.startswith("+") else num

def _wa_base_url() -> str:
    """Construye la URL en base al PHONE_ID actual (por seguridad)."""
    return f"{FB_API}/{WHATSAPP_PHONE_ID}/messages"

def wa_send_text(to_number: str, body: str):
    """Envía texto por la Cloud API."""
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_ID):
        print("⚠️ Falta WHATSAPP_TOKEN o WHATSAPP_PHONE_ID")
        return {"ok": False, "error": "missing_credentials"}

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": _e164_no_plus(to_number),
        "type": "text",
        "text": {"preview_url": False, "body": body[:3900]},  # < 4096 chars
    }
    try:
        r = requests.post(_wa_base_url(), headers=headers, json=data, timeout=20)
        j = r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}
        return {"ok": r.ok, "status": r.status_code, "resp": j}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def wa_get_media_url(media_id: str) -> str:
    """Paso 1: con el media_id, obtén la URL temporal firmada desde Graph."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(f"{FB_API}/{media_id}", headers=headers, timeout=15)
    r.raise_for_status()
    j = r.json()
    return j.get("url", "")

def wa_download_media(signed_url: str, dest_prefix: str = "/tmp/wa_media") -> tuple[str, str]:
    """
    Paso 2: descarga el binario usando la URL firmada.
    Devuelve (ruta_archivo, mimetype).
    """
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(signed_url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()

    mime = r.headers.get("Content-Type", "application/octet-stream")
    ext = mimetypes.guess_extension(mime) or ""
    dest_path = f"{dest_prefix}{int(time.time())}{ext}"

    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    return dest_path, mime

# --- VERIFICACIÓN (GET) ---
@app.get("/webhook")
async def verify_webhook(request: Request):
    # Meta envía: hub.mode, hub.verify_token, hub.challenge
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")

    print("WEBHOOK VERIFY =>", {"mode": mode, "token": token, "challenge": challenge})

    # Debe devolver EXACTAMENTE el challenge en texto plano (200)
    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(content=challenge, status_code=200)

    return PlainTextResponse(content="forbidden", status_code=403)

# --- RECEPCIÓN DE MENSAJES (POST) ---
@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        # Meta requiere 200 siempre; devuelve algo útil para logs
        return JSONResponse({"received": False, "error": "invalid_json"})

    print("📩 Payload recibido:", data)

    try:
        entry   = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        value   = changes.get("value") or {}
        msgs    = value.get("messages") or []

        if not msgs:
            # También pueden llegar "statuses" (entregas, lectura, etc.)
            return {"status": "no_message"}

        msg = msgs[0]
        from_number = msg.get("from")
        mtype = msg.get("type")

        # ---------- TEXTO / BOTÓN ----------
        if mtype == "text":
            user_text = (msg.get("text") or {}).get("body", "").strip()
        elif mtype == "button":
            user_text = (msg.get("button") or {}).get("text", "").strip()
        else:
            user_text = ""

        if user_text:
            try:
                answer = call_openai(user_text)
            except Exception:
                answer = "Lo siento, tuve un problema procesando tu mensaje."
            if from_number:
                wa_send_text(from_number, answer)
            return {"status": "ok"}

        # ---------- IMAGEN ----------
        if mtype == "image":
            media_id = (msg.get("image") or {}).get("id")
            caption  = (msg.get("image") or {}).get("caption") or ""
            if media_id and from_number:
                try:
                    url = wa_get_media_url(media_id)
                    path, mime = wa_download_media(url)
                    print(f"🖼️ Imagen guardada en {path} ({mime})")
                    reply = "🖼️ Recibí tu imagen."
                    if caption.strip():
                        reply += f"\n📎 *Caption:* {caption.strip()}"
                    reply += "\n\n¿Qué te gustaría analizar de esta imagen?"
                    wa_send_text(from_number, reply)
                except Exception as e:
                    print("Error descargando imagen:", e)
                    wa_send_text(from_number, "No pude descargar la imagen. ¿Puedes intentar de nuevo?")
            return {"status": "ok"}

        # ---------- DOCUMENTO (PDF/Word/etc.) ----------
        if mtype == "document":
            doc = msg.get("document") or {}
            media_id = doc.get("id")
            filename = doc.get("filename") or "archivo"
            if media_id and from_number:
                try:
                    url = wa_get_media_url(media_id)
                    path, mime = wa_download_media(url)
                    print(f"📄 Documento guardado en {path} ({mime})")
                    wa_send_text(
                        from_number,
                        f"📄 Recibí tu documento *{filename}*.\n"
                        "Puedo revisarlo (si es texto/PDF) o archivarlo. ¿Qué deseas que haga?"
                    )
                except Exception as e:
                    print("Error descargando documento:", e)
                    wa_send_text(from_number, "No pude descargar el documento. ¿Puedes intentar de nuevo?")
            return {"status": "ok"}

        # ---------- OTROS TIPOS ----------
        if from_number:
            wa_send_text(
                from_number,
                "Recibí tu mensaje. Por ahora manejo texto, imágenes y documentos. "
                "Si necesitas algo con audio/video, avísame."
            )
        return {"status": "ok"}

    except Exception as e:
        print("❌ Error en webhook:", e)
        return {"status": "error", "detail": str(e)}
