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
# CORS (en pruebas = "*")
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

# ----------------------------
# Configuración Google Sheets
# ----------------------------
SHEETS_WEBHOOK_URL = os.getenv("SHEETS_WEBHOOK_URL", "").strip()

def send_ticket_to_sheets(data: dict):
    if not SHEETS_WEBHOOK_URL:
        print("⚠️ Falta SHEETS_WEBHOOK_URL (no se envió ticket)")
        return {"ok": False, "error": "missing_webhook"}
    try:
        r = requests.post(SHEETS_WEBHOOK_URL, json=data, timeout=10)
        ok = r.status_code == 200
        print(f"📨 Ticket a Sheets -> status={r.status_code} ok={ok} resp={r.text[:200]}")
        return {"ok": ok, "status": r.status_code, "resp": r.text}
    except Exception as e:
        print("❌ Error enviando ticket a Sheets:", e)
        return {"ok": False, "error": str(e)}

# ----------------------------
# Utilidades
# ----------------------------
LANG_NAME = {"es": "Spanish","en": "English","pt": "Portuguese","fr": "French"}

class ChatIn(BaseModel):
    pregunta: str

HIST = []
MAX_HIST = 200

def detect_lang(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõáéíóúç]", t): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    return "en"

def call_openai(question: str, lang_hint: str|None=None) -> str:
    sys = SYSTEM_PROMPT
    if lang_hint in LANG_NAME:
        sys += f"\n- The user's language is {LANG_NAME[lang_hint]}. Always reply in {LANG_NAME[lang_hint]}."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":question}],
            temperature=OPENAI_TEMP,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="Error con el modelo")

def transcribe_audio_with_openai(audio_path: str) -> str:
    try:
        with open(audio_path,"rb") as f:
            tr = client.audio.transcriptions.create(model="whisper-1",file=f)
        return (tr.text or "").strip()
    except Exception as e1:
        print("whisper-1 falló:", e1)
        return ""

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
    lang = detect_lang(q)
    cached = get_from_cache(q, lang)
    if cached is not None:
        return {"respuesta": cached, "cached": True}
    a = call_openai(q, lang_hint=lang)
    save_to_cache(q, lang, a)
    HIST.append({"t": time.time(), "pregunta": q, "respuesta": a})
    if len(HIST) > MAX_HIST: del HIST[:len(HIST)-MAX_HIST]
    return {"respuesta": a, "cached": False}

# ======================================================================
# WHATSAPP WEBHOOK
# ======================================================================
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123")
FB_API = "https://graph.facebook.com/v20.0"

def _e164_no_plus(num: str) -> str:
    num = (num or "").strip().replace(" ","").replace("-","")
    return num[1:] if num.startswith("+") else num

def _wa_base_url() -> str:
    return f"{FB_API}/{WHATSAPP_PHONE_ID}/messages"

def wa_send_text(to_number: str, body: str):
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_ID):
        print("⚠️ Falta WHATSAPP_TOKEN o WHATSAPP_PHONE_ID")
        return {"ok": False, "error": "missing_credentials"}
    headers = {"Authorization":f"Bearer {WHATSAPP_TOKEN}","Content-Type":"application/json"}
    data = {"messaging_product":"whatsapp","to":_e164_no_plus(to_number),"type":"text","text":{"preview_url":False,"body":body[:3900]}}
    try:
        r = requests.post(_wa_base_url(),headers=headers,json=data,timeout=20)
        j = r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw":r.text}
        return {"ok":r.ok,"status":r.status_code,"resp":j}
    except Exception as e:
        return {"ok":False,"error":str(e)}

# --- VERIFICACIÓN (GET) ---
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode=request.query_params.get("hub.mode","")
    token=request.query_params.get("hub.verify_token","")
    challenge=request.query_params.get("hub.challenge","")
    if mode=="subscribe" and token==META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(content=challenge,status_code=200)
    return PlainTextResponse(content="forbidden",status_code=403)

# --- RECEPCIÓN DE MENSAJES (POST) ---
@app.post("/webhook")
async def webhook_handler(request: Request):
    try: data=await request.json()
    except Exception: return JSONResponse({"received":False,"error":"invalid_json"})
    print("📩 Payload recibido:",data)
    try:
        entry=(data.get("entry") or [{}])[0]
        changes=(entry.get("changes") or [{}])[0]
        value=changes.get("value") or {}
        msgs=value.get("messages") or []
        if msgs:
            msg=msgs[0]; from_number=msg.get("from"); mtype=msg.get("type")
            user_text=""
            if mtype=="text": user_text=(msg.get("text") or {}).get("body","").strip()
            if mtype=="audio":
                aud=msg.get("audio") or {}; media_id=aud.get("id")
                if media_id and from_number:
                    # (ejemplo simplificado, descarga de audio no implementada aquí)
                    wa_send_text(from_number,"🎧 Recibí tu audio. Transcripción no implementada aquí.")
            if user_text:
                lang=detect_lang(user_text)
                answer=call_openai(user_text,lang_hint=lang)
                wa_send_text(from_number,answer)
                ticket={"fecha":time.strftime("%Y-%m-%d %H:%M:%S"),"numero":from_number,"mensaje":user_text,"respuesta":answer,"etiqueta":"NochGPT"}
                send_ticket_to_sheets(ticket)
                return {"status":"ok"}
        return {"status":"no_message"}
    except Exception as e:
        print("❌ Error en webhook:",e)
        return {"status":"error"}
