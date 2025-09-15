# server/main.py
import os, re, time, json, base64, mimetypes, pathlib
from typing import Any, Dict, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# === OpenAI client ===
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ Falta OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMP = float(os.getenv("OPENAI_TEMP", "0.2"))

# === WhatsApp / Meta ===
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123")
FB_API = "https://graph.facebook.com/v20.0"

# === Google Sheets (Apps Script) ===
SHEETS_WEBHOOK_URL = os.getenv("SHEETS_WEBHOOK_URL", "")  # URL del Web App

# === App FastAPI ===
app = FastAPI(title="Dental-LLM (simple)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ========= Utilidades =========
LANG_NAME = {"es": "Spanish", "en": "English", "pt":"Portuguese", "fr":"French"}

SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise and practical. Give ranges (temperatures, times) if relevant.
- If the question is not dental-related, say you are focused on dental topics and offer a useful redirection.
- IMPORTANT: Always reply in the same language as the user.
"""

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõáéíóúç]", t):  return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    return "en"

def call_openai_chat(user_text: str, lang_hint: str) -> str:
    sys = SYSTEM_PROMPT
    if lang_hint in LANG_NAME:
        sys += f"\n- The user's language is {LANG_NAME[lang_hint]}. Always reply in {LANG_NAME[lang_hint]}."
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=OPENAI_TEMP,
            messages=[
                {"role":"system","content":sys},
                {"role":"user","content":user_text},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        return {
            "es": "Tuve un problema procesando tu mensaje. ¿Puedes intentar de nuevo?",
            "en": "I had an issue processing your message. Could you try again?",
            "pt": "Tive um problema ao processar sua mensagem. Pode tentar novamente?",
            "fr": "J’ai rencontré un problème en traitant votre message. Pouvez-vous réessayer ?",
        }.get(lang_hint, "I had an issue processing your message. Please try again.")

def transcribe_audio(audio_path: str) -> str:
    # Intenta Whisper; si falla usa gpt-4o-mini-transcribe
    try:
        with open(audio_path, "rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1", file=f)
        return (tr.text or "").strip()
    except Exception as e1:
        print("Whisper falló -> mini-transcribe:", e1)
        try:
            with open(audio_path, "rb") as f:
                tr = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
            return (tr.text or "").strip()
        except Exception as e2:
            print("Transcripción falló:", e2)
            return ""

def _e164_no_plus(num: str) -> str:
    num = (num or "").strip().replace(" ","").replace("-","")
    return num[1:] if num.startswith("+") else num

def wa_send_text(to_number: str, body: str, retries: int = 2) -> Dict[str,Any]:
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_ID):
        print("⚠️ Falta WHATSAPP_TOKEN/WHATSAPP_PHONE_ID")
        return {"ok": False, "error":"missing_credentials"}

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": _e164_no_plus(to_number),
        "type": "text",
        "text": {"preview_url": False, "body": body[:3900]},
    }
    url = f"{FB_API}/{WHATSAPP_PHONE_ID}/messages"

    for i in range(retries+1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=20)
            if r.ok: return {"ok": True, "status": r.status_code, "resp": r.json()}
            # Reintenta en 400/5xx que valgan la pena
            if r.status_code >= 500 or r.status_code == 429:
                time.sleep(1.5*(i+1))
                continue
            return {"ok": False, "status": r.status_code, "resp": r.json()}
        except Exception as e:
            err = str(e)
            if i < retries:
                time.sleep(1.5*(i+1))
                continue
            return {"ok": False, "error": err}

def wa_get_media_url(media_id: str) -> str:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(f"{FB_API}/{media_id}", headers=headers, timeout=15)
    r.raise_for_status()
    return (r.json() or {}).get("url", "")

def wa_download_media(signed_url: str, dest_prefix: str = "/tmp/wa_media/") -> Tuple[str, str]:
    pathlib.Path(dest_prefix).mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(signed_url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    mime = r.headers.get("Content-Type","application/octet-stream")
    ext = mimetypes.guess_extension(mime) or ".bin"
    path = os.path.join(dest_prefix, f"{int(time.time())}{ext}")
    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk: f.write(chunk)
    return path, mime

def send_ticket_to_sheet(number: str, message: str, name: str = "", tag: str = "NochGPT") -> None:
    """Envía un ticket simple a tu Google Sheets (Apps Script doPost)."""
    if not SHEETS_WEBHOOK_URL:
        print("⚠️ Falta SHEETS_WEBHOOK_URL (no se envió ticket)")
        return
    payload = {
        "ts": int(time.time()),
        "from": number,
        "nombre": name or "",
        "tema": "",
        "contacto": number,
        "horario": "",
        "mensaje": message,
        "label": tag,
    }
    try:
        r = requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=15)
        print("Ticket Sheets status:", r.status_code, r.text[:200])
    except Exception as e:
        print("Error enviando ticket Sheets:", e)

# ========= Rutas simples =========
class ChatIn(BaseModel):
    pregunta: str

@app.get("/", response_class=HTMLResponse)
def home(): return "<h3>Dental-LLM simple ✅</h3>"

@app.get("/health")
def health(): return {"ok": True}

@app.post("/chat")
def chat(body: ChatIn):
    q = (body.pregunta or "").strip()
    if not q: raise HTTPException(status_code=400, detail="Falta 'pregunta'")
    lang = detect_lang(q)
    a = call_openai_chat(q, lang)
    return {"respuesta": a}

# ========= Webhook WhatsApp =========
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode","")
    token = request.query_params.get("hub.verify_token","")
    challenge = request.query_params.get("hub.challenge","")
    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="forbidden", status_code=403)

@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"received": False, "error":"invalid_json"})

    print("📩 WA payload:", json.dumps(data)[:1200])

    try:
        entry = (data.get("entry") or [{}])[0]
        value = ((entry.get("changes") or [{}])[0]).get("value", {})
        msgs = value.get("messages") or []

        if not msgs:
            return {"status":"no_message"}

        msg = msgs[0]
        from_number = msg.get("from")
        mtype = msg.get("type")

        # --- Texto ---
        if mtype == "text":
            user_text = (msg.get("text") or {}).get("body","").strip()
            if user_text:
                lang = detect_lang(user_text)
                # Guarda ticket
                send_ticket_to_sheet(from_number, user_text)
                # Responde
                answer = call_openai_chat(user_text, lang)
                if from_number: wa_send_text(from_number, answer)
                return {"status":"ok_text"}

        # --- Audio / Nota de voz ---
        if mtype == "audio":
            aud = msg.get("audio") or {}
            media_id = aud.get("id")
            if media_id and from_number:
                try:
                    url = wa_get_media_url(media_id)
                    path, mime = wa_download_media(url)
                    transcript = transcribe_audio(path)
                    if transcript:
                        lang = detect_lang(transcript)
                        send_ticket_to_sheet(from_number, transcript)  # guarda lo transcrito
                        answer = call_openai_chat(
                            f"Transcription of user's audio:\n\"\"\"{transcript}\"\"\"", lang
                        )
                        wa_send_text(from_number, f"🗣️ {transcript}\n\n{answer}")
                    else:
                        wa_send_text(from_number, {
                            "es":"No pude transcribir tu audio. ¿Puedes intentar de nuevo?",
                            "en":"I couldn’t transcribe your audio. Could you try again?",
                            "pt":"Não consegui transcrever seu áudio. Pode tentar novamente?",
                            "fr":"Je n’ai pas pu transcrire votre audio. Pourriez-vous réessayer ?",
                        }[detect_lang("")])
                except Exception as e:
                    print("Error audio:", e)
                    wa_send_text(from_number, "Hubo un problema con tu audio. ¿Puedes intentar de nuevo?")
                return {"status":"ok_audio"}

        # Otros tipos → mensaje genérico
        if from_number:
            wa_send_text(
                from_number,
                "Recibí tu mensaje. Por ahora manejo texto y notas de voz. ¿En qué puedo ayudarte?"
            )
        return {"status":"ok_other"}

    except Exception as e:
        print("❌ Error webhook:", e)
        return {"status":"error"}
