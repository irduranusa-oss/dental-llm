# --- IMPORTS ---
import os, re, time, json, base64, mimetypes, pathlib
from datetime import datetime
import requests

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from pydantic import BaseModel

# OpenAI client
from openai import OpenAI
# --------------------------------------------------------------------

app = FastAPI(title="Dental-LLM API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOW_ORIGIN", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# ENV & OpenAI
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ Falta OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # chat + visión
OPENAI_TEMP = float(os.getenv("OPENAI_TEMP", "0.2"))

SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise, practical, and provide ranges (e.g., temperatures or times) when relevant.
- If the question is not dental-related, politely say you are focused on dental topics and offer a helpful redirection.
- IMPORTANT: Always reply in the same language as the user's question.
"""

LANG_NAME = {"es":"Spanish","en":"English","pt":"Portuguese","fr":"French"}

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõáéíóúç]", t):  return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
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
        return "Lo siento, hubo un problema con el modelo. Intenta de nuevo."

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

# ----------------------------
# WhatsApp API helpers
# ----------------------------
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123")
FB_API = "https://graph.facebook.com/v20.0"

def _e164_no_plus(num: str) -> str:
    num = (num or "").strip().replace(" ", "").replace("-", "")
    return num[1:] if num.startswith("+") else num

def _wa_base_url() -> str:
    return f"{FB_API}/{WHATSAPP_PHONE_ID}/messages"

def wa_send_text(to_number: str, body: str):
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
        "text": {"preview_url": False, "body": body[:3900]},
    }
    try:
        r = requests.post(_wa_base_url(), headers=headers, json=data, timeout=20)
        j = r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}
        if not r.ok:
            print("WA send error:", j)
        return {"ok": r.ok, "status": r.status_code, "resp": j}
    except Exception as e:
        print("WA send exception:", e)
        return {"ok": False, "error": str(e)}

def wa_get_media_url(media_id: str) -> str:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(f"{FB_API}/{media_id}", headers=headers, timeout=15)
    r.raise_for_status()
    return (r.json() or {}).get("url", "")

def wa_download_media(signed_url: str, dest_prefix: str = "/tmp/wa_media/") -> tuple[str, str]:
    pathlib.Path(dest_prefix).mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(signed_url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    mime = r.headers.get("Content-Type", "application/octet-stream")
    ext = mimetypes.guess_extension(mime) or ".bin"
    path = os.path.join(dest_prefix, f"{int(time.time())}{ext}")
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    return path, mime

# ----------------------------
# Google Sheets webhook (Apps Script)
# ----------------------------
SHEETS_WEBHOOK_URL = os.getenv("SHEET_WEBHOOK", "")  # Tu env ya se llama así
def send_ticket_to_sheet(numero: str, mensaje: str, respuesta: str, etiqueta: str = "NochGPT"):
    if not SHEETS_WEBHOOK_URL:
        print("⚠️ Falta SHEET_WEBHOOK (URL de Apps Script) — no se envía ticket")
        return {"ok": False, "error": "missing_sheet_webhook"}
    payload = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "numero": (numero or ""),
        "nombre": "",
        "tema": "",
        "contacto": (numero or ""),
        "horario": "",
        "mensaje": (mensaje or ""),
        "respuesta": (respuesta or ""),
        "etiqueta": etiqueta,
    }
    try:
        r = requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=15)
        ok = r.status_code == 200
        if not ok:
            print("Sheet webhook error:", r.text)
        return {"ok": ok, "status": r.status_code, "resp": r.text}
    except Exception as e:
        print("Sheet webhook exception:", e)
        return {"ok": False, "error": str(e)}

# ----------------------------
# Rutas simples
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return "<h3>Dental-LLM corriendo ✅</h3>"

@app.get("/health")
def health():
    return {"ok": True}

# ----------------------------
# Webhook Verify (GET)
# ----------------------------
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode", "")
    token = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")
    print("WEBHOOK VERIFY =>", {"mode": mode, "token": token, "challenge": challenge})
    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="forbidden", status_code=403)

# ----------------------------
# Webhook Receive (POST)
# ----------------------------
@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"received": False, "error": "invalid_json"})
    print("📩 Payload:", json.dumps(data)[:1200], "...")

    try:
        entry = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        value = changes.get("value") or {}

        # A) Mensajes nuevos
        msgs = value.get("messages") or []
        if msgs:
            msg = msgs[0]
            from_number = msg.get("from")
            mtype = msg.get("type")

            # --- TEXTO ---
            if mtype == "text":
                user_text = (msg.get("text") or {}).get("body", "").strip()
                if not user_text:
                    return {"status": "empty_text"}
                lang = detect_lang(user_text)
                answer = call_openai(user_text, lang_hint=lang)
                # Responder al usuario
                if from_number: wa_send_text(from_number, answer)
                # Ticket
                send_ticket_to_sheet(from_number, user_text, answer, etiqueta="NochGPT")
                return {"status": "ok_text"}

            # --- AUDIO (nota de voz) ---
            if mtype == "audio":
                aud = msg.get("audio") or {}
                media_id = aud.get("id")
                if media_id and from_number:
                    try:
                        url = wa_get_media_url(media_id)
                        path, mime = wa_download_media(url)
                        print(f"🎧 Audio guardado en {path} ({mime})")
                        transcript = transcribe_audio_with_openai(path)
                        if not transcript:
                            wa_send_text(from_number, "🎧 Recibí tu audio pero no pude transcribirlo. ¿Puedes intentar otra vez?")
                            return {"status": "audio_no_transcript"}
                        lang = detect_lang(transcript)
                        answer = call_openai(
                            f"Transcripción del audio del usuario:\n\"\"\"{transcript}\"\"\"",
                            lang_hint=lang
                        )
                        wa_send_text(from_number, f"🗣️ *Transcripción*:\n{transcript}\n\n💬 *Respuesta*:\n{answer}")
                        # Ticket
                        send_ticket_to_sheet(from_number, transcript, answer, etiqueta="NochGPT")
                        return {"status": "ok_audio"}
                    except Exception as e:
                        print("Audio error:", e)
                        if from_number: wa_send_text(from_number, "No pude procesar el audio. Intenta nuevamente, por favor.")
                        return {"status": "audio_error"}

            # Otros tipos: ignorar amablemente
            if from_number:
                wa_send_text(from_number, "Recibí tu mensaje. Por ahora manejo texto y notas de voz.")
            return {"status": "other_type"}

        # B) Status (entregado/leído)
        if value.get("statuses"):
            return {"status": "status_ok"}

        return {"status": "no_message"}

    except Exception as e:
        print("❌ Error webhook:", e)
        return {"status": "error"}
