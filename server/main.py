# server/main.py — NochGPT WhatsApp v3.2 (idioma+botones+audio+imagen+sheets)
# ---------------------------------------------------------------------------
# ✅ Autodetecta idioma y responde en el mismo
# ✅ “hola/hello/привет/नमस्ते/こんにちは/مرحبا” → menú de botones en el idioma
# ✅ Transcribe audios con Whisper (whisper-1)
# ✅ Analiza imágenes con GPT-4o-mini (visión)
# ✅ Tickets: guarda en /tmp/handoff.json y envía a Google Sheets (Apps Script)
# ✅ Endpoints: /webhook (GET verify + POST), /_debug/health, /handoff, /tickets, /panel
# ---------------------------------------------------------------------------

from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
import os, re, json, time, requests, mimetypes, pathlib
from datetime import datetime, timezone

app = FastAPI(title="Dental-LLM API")

# CORS amplio
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# --- Configuración de APIs/Modelos ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMP    = float(os.getenv("OPENAI_TEMP", "0.2"))

WA_TOKEN    = os.getenv("WA_TOKEN", "")
WA_PHONE_ID = os.getenv("WA_PHONE_ID", "")
SHEETS_WEBHOOK_URL = os.getenv("SHEETS_WEBHOOK_URL", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Paths de runtime ---
DATA_DIR = "/tmp"
HANDOFF_FILE = f"{DATA_DIR}/handoff.json"
TICKETS_FILE = f"{DATA_DIR}/tickets.json"
pathlib.Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# --- Utilidades de idioma ---
HELLO_PATTERNS = {
    "es": r"\b(hola|buenas|qué\s+tal|saludos)\b",
    "en": r"\b(hi|hello|hey|howdy)\b",
    "ru": r"\b(привет|здравствуй|здравствуйте)\b",
    "hi": r"\b(नमस्ते|नमस्कार)\b",
    "ar": r"\b(مرحبا|اهلا|السلام\s+عليكم)\b",
    "ja": r"\b(こんにちは|こんちは|やあ)\b",
}

def detect_lang(text: str) -> str:
    """Heurística simple + signos diacríticos para es/ru/hi/ar/ja/en."""
    t = (text or "").strip()
    if not t:
        return "es"
    low = t.lower()

    # by script / characters
    if re.search(r"[\u0600-\u06FF]", t):  # Arabic
        return "ar"
    if re.search(r"[\u0400-\u04FF]", t):  # Cyrillic (Russian)
        return "ru"
    if re.search(r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9D]", t):  # Japanese
        return "ja"
    if re.search(r"[\u0900-\u097F]", t):  # Devanagari (Hindi)
        return "hi"
    if re.search(r"[áéíóúñ¿¡]", low):
        return "es"
    # keywords hello
    for lang, pat in HELLO_PATTERNS.items():
        if re.search(pat, low):
            return lang
    return "en"

def tr(lang: str, key: str) -> str:
    """Traducciones básicas para botones y textos fijos."""
    TXT = {
        "menu_title": {
            "es": "¿En qué te ayudo hoy?",
            "en": "How can I help today?",
            "ru": "Чем могу помочь сегодня?",
            "hi": "मैं आज आपकी कैसे मदद कर सकता हूँ?",
            "ar": "كيف يمكنني مساعدتك اليوم؟",
            "ja": "今日は何をお手伝いできますか？",
        },
        "btn_prices": {
            "es": "Planes y precios", "en": "Plans & Pricing", "ru": "Тарифы и цены",
            "hi": "प्लान और कीमतें", "ar": "الخطط والأسعار", "ja": "料金プラン"
        },
        "btn_human": {
            "es": "Hablar con humano", "en": "Talk to human", "ru": "Связаться с оператором",
            "hi": "मानव से बात करें", "ar": "التحدث مع موظف", "ja": "担当者に相談"
        },
        "btn_quote": {
            "es": "Cotizar un caso", "en": "Request a quote", "ru": "Рассчитать стоимость",
            "hi": "कोटेशन चाहिए", "ar": "طلب تسعير", "ja": "見積もり依頼"
        },
        "audio_receipt": {
            "es": "🎙️ Recibí tu audio, lo transcribo…",
            "en": "🎙️ Got your audio, transcribing…",
            "ru": "🎙️ Аудио получено, расшифровываю…",
            "hi": "🎙️ आपका ऑडियो मिला, ट्रांसक्राइब कर रहा हूँ…",
            "ar": "🎙️ استلمت رسالتك الصوتية، أقوم بالنسخ…",
            "ja": "🎙️ 音声を受け取りました。文字起こし中…",
        },
        "image_receipt": {
            "es": "🖼️ Recibí tu imagen, la analizo…",
            "en": "🖼️ Got your image, analyzing…",
            "ru": "🖼️ Изображение получено, анализирую…",
            "hi": "🖼️ आपकी तस्वीर मिली, विश्लेषण कर रहा हूँ…",
            "ar": "🖼️ استلمت الصورة، أقوم بالتحليل…",
            "ja": "🖼️ 画像を受け取りました。解析します…",
        },
        "ticket_thanks": {
            "es": "✅ Gracias. Tu solicitud fue registrada y la atiende un asesor.",
            "en": "✅ Thanks. Your request was recorded; an agent will contact you soon.",
            "ru": "✅ Спасибо. Запрос зарегистрирован, с вами скоро свяжется специалист.",
            "hi": "✅ धन्यवाद। आपका अनुरोध दर्ज हो गया है; एक प्रतिनिधि आपसे शीघ्र संपर्क करेगा।",
            "ar": "✅ شكرًا. تم تسجيل طلبك وسيتصل بك أحد المختصين قريبًا.",
            "ja": "✅ ありがとうございます。担当者が折り返しご連絡します。",
        },
    }
    table = TXT.get(key, {})
    return table.get(lang, table.get("en", key))

SYSTEM_PROMPT = (
    "You are NochGPT, a helpful dental laboratory assistant. "
    "Focus only on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials). "
    "Be concise and actionable. Provide ranges when useful. "
    "Always answer in the user's language. If off-topic, politely steer back to dental."
)

# --- WhatsApp helpers ---
def wa_url(path: str) -> str:
    return f"https://graph.facebook.com/v20.0/{path}"

def wa_headers():
    return {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}

def wa_send_text(to: str, text: str):
    payload = {"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": text}}
    requests.post(wa_url(f"{WA_PHONE_ID}/messages"), headers=wa_headers(), json=payload, timeout=30)

def wa_send_buttons(to: str, lang: str):
    body = tr(lang, "menu_title")
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "prices", "title": tr(lang, "btn_prices")}},
                    {"type": "reply", "reply": {"id": "quote",  "title": tr(lang, "btn_quote")}},
                    {"type": "reply", "reply": {"id": "human",  "title": tr(lang, "btn_human")}},
                ]
            },
        },
    }
    requests.post(wa_url(f"{WA_PHONE_ID}/messages"), headers=wa_headers(), json=payload, timeout=30)

def wa_download_media(media_id: str) -> tuple[str, bytes, str]:
    # Step 1: get URL
    r = requests.get(wa_url(media_id), headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=30)
    r.raise_for_status()
    media_url = r.json().get("url")
    # Step 2: download
    r2 = requests.get(media_url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=60)
    r2.raise_for_status()
    content_type = r2.headers.get("Content-Type", "application/octet-stream")
    ext = mimetypes.guess_extension(content_type) or ""
    return content_type, r2.content, ext

# --- Tickets (Sheets + archivo local) ---
def save_ticket(row: dict):
    # guarda en archivo local
    data = []
    if os.path.exists(HANDOFF_FILE):
        try:
            data = json.load(open(HANDOFF_FILE, "r", encoding="utf-8"))
        except Exception:
            data = []
    data.append(row)
    with open(HANDOFF_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    # envía a Google Sheets si está configurado
    if SHEETS_WEBHOOK_URL:
        try:
            requests.post(SHEETS_WEBHOOK_URL, json=row, timeout=15)
        except Exception as e:
            print("Sheets webhook error:", e)

def make_ticket(from_number: str, message: str, nombre="", tema="", contacto="", horario="", etiqueta="NochGPT"):
    ts = int(time.time())
    row = {
        "ts": ts,
        "label": etiqueta,
        "from": from_number,
        "nombre": nombre or "",
        "tema": tema or "",
        "contacto": contacto or from_number,
        "horario": horario or "",
        "mensaje": message or "",
    }
    save_ticket(row)
    return row

# --- LLM helpers ---
def llm_answer(user_text: str, user_lang: str) -> str:
    sys = SYSTEM_PROMPT + f" Respond in: {user_lang}."
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMP,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def llm_vision_answer(user_text: str, image_url: str, user_lang: str) -> str:
    sys = SYSTEM_PROMPT + f" Respond in: {user_lang}."
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMP,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_text or "Analiza la imagen clínica y da sugerencias."},
                 {"type": "image_url", "image_url": {"url": image_url}},
             ]},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

def transcribe_audio_bytes(b: bytes, fname: str = "audio.ogg") -> str:
    # guarda temporal y envía a Whisper
    tmp = f"{DATA_DIR}/{int(time.time()*1000)}_{fname}"
    with open(tmp, "wb") as f:
        f.write(b)
    with open(tmp, "rb") as f:
        tr = client.audio.transcriptions.create(model="whisper-1", file=f)
    try:
        os.remove(tmp)
    except Exception:
        pass
    return (tr.text or "").strip()

# --- Webhook WhatsApp ---
@app.get("/_debug/health")
def health():
    cfg = {
        "openai": bool(OPENAI_API_KEY),
        "wa_token": bool(WA_TOKEN),
        "wa_phone_id": bool(WA_PHONE_ID),
        "model": OPENAI_MODEL,
        "sheet_webhook": bool(SHEETS_WEBHOOK_URL),
    }
    return {"ok": True, "cfg": cfg}

@app.get("/")
def root():
    return HTMLResponse("<b>Dental-LLM corriendo ✅</b>")

@app.get("/tickets")
def tickets():
    if os.path.exists(HANDOFF_FILE):
        return JSONResponse(json.load(open(HANDOFF_FILE, "r", encoding="utf-8")))
    return JSONResponse([])

@app.get("/handoff")
def handoff():
    return tickets()

@app.get("/panel")
def panel():
    rows = []
    if os.path.exists(HANDOFF_FILE):
        rows = json.load(open(HANDOFF_FILE, "r", encoding="utf-8"))
    html = [
        "<html><head><meta charset='utf-8'><title>Tickets – NochGPT</title>",
        "<style>body{font-family:system-ui;padding:16px;background:#0b1220;color:#e5e7eb}",
        "table{width:100%;border-collapse:collapse} th,td{padding:10px} ",
        "th{background:#111827} tr:nth-child(even){background:#11182780}",
        ".pill{background:#10b981;padding:6px 10px;border-radius:9999px;color:#052e2b;font-weight:700}",
        "</style></head><body>",
        f"<h2>Tickets – NochGPT <span class='pill'>{len(rows)} activos</span></h2>",
        "<p>Se actualiza al refrescar la página · Origen: /tmp/handoff.json</p>",
        "<table><tr><th>Fecha/Hora</th><th>Número</th><th>Nombre</th><th>Tema</th><th>Contacto</th><th>Mensaje</th></tr>"
    ]
    for r in rows[::-1]:
        dt = datetime.fromtimestamp(r.get("ts", 0), tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")
        html.append(f"<tr><td>{dt}</td><td>{r.get('from','')}</td><td>{r.get('nombre','')}</td>"
                    f"<td>{r.get('tema','')}</td><td>{r.get('contacto','')}</td><td>{r.get('mensaje','')}</td></tr>")
    html.append("</table></body></html>")
    return HTMLResponse("".join(html))

# Verificación GET (Meta)
@app.get("/webhook")
def verify(request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123"):
        return PlainTextResponse(challenge or "")
    return PlainTextResponse("not verified", status_code=403)

# Recepción POST
@app.post("/webhook")
def webhook(req: Request):
    body = req.json() if hasattr(req, "json") else None
    if not body:
        body = json.loads(requests.Request.body or "{}")
    entry = body.get("entry", [])
    # WhatsApp status callbacks no necesitan respuesta
    for e in entry:
        changes = e.get("changes", [])
        for ch in changes:
            value = ch.get("value", {})
            if value.get("messages"):
                handle_messages(value)
    return JSONResponse({"ok": True})

def handle_messages(value: dict):
    contacts = value.get("contacts", [{}])
    messages = value.get("messages", [])
    from_number = contacts[0].get("wa_id") or value.get("from") or messages[0].get("from")

    for msg in messages:
        mtype = msg.get("type")
        # Interactive reply (botones)
        if mtype == "interactive":
            lang = "es"  # por simplicidad muestra confirmación en español; se podría guardar estado/idioma por user
            payload = msg.get("interactive", {})
            id_sel = ""
            if payload.get("type") == "button_reply":
                id_sel = payload["button_reply"].get("id", "")
            elif payload.get("type") == "list_reply":
                id_sel = payload["list_reply"].get("id", "")
            if id_sel == "human":
                # Levanta ticket
                make_ticket(from_number, "Hablar con humano")
                wa_send_text(from_number, tr(lang, "ticket_thanks"))
            elif id_sel == "prices":
                # Envía lista simple de planes
                text = (
                    "🧾 *Planes*\n"
                    "• IA con tu Logo — $500\n"
                    "• Exocad experto — $500\n"
                    "• Meshmixer — $50\n\n"
                    "¿Cuál te interesa?"
                )
                wa_send_text(from_number, text)
            elif id_sel == "quote":
                wa_send_text(from_number, "Escribe tu caso (pieza, material, plazo) para cotizar.")
            else:
                wa_send_text(from_number, "✓")
            continue

        # Texto
        if mtype == "text":
            user_text = msg["text"]["body"]
            lang = detect_lang(user_text)

            # Si es saludo → botones
            if re.search(HELLO_PATTERNS.get(lang, ""), user_text.lower()):
                wa_send_buttons(from_number, lang)
                return

            # Si pide humano explícitamente → ticket
            if user_text.strip().lower() in ["hablar con humano", "human", "asesor", "operator", "soporte"]:
                make_ticket(from_number, user_text)
                wa_send_text(from_number, tr(lang, "ticket_thanks"))
                return

            answer = llm_answer(user_text, lang)
            wa_send_text(from_number, answer)
            return

        # Imágenes
        if mtype == "image":
            lang = detect_lang(value.get("metadata", {}).get("display_phone_number", "")) or "es"
            wa_send_text(from_number, tr(lang, "image_receipt"))
            media_id = msg["image"]["id"]
            # Descargar binario y subir a hosting temporal de Meta → construir URL directa
            # Para visión en OpenAI basta con URL pública; WhatsApp requiere doble paso.
            # Truco: usar el endpoint de media url directamente (no re-hosteamos).
            r = requests.get(wa_url(media_id), headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=30)
            r.raise_for_status()
            media_url = r.json().get("url")
            answer = llm_vision_answer("Analiza la imagen clínica y sugiere pasos.", media_url, lang)
            wa_send_text(from_number, answer)
            return

        # Audios
        if mtype == "audio":
            # Recibo audio → descargo y transcribo
            audio_id = msg["audio"]["id"]
            lang_guess = "es"
            wa_send_text(from_number, tr(lang_guess, "audio_receipt"))
            ct, bts, ext = wa_download_media(audio_id)
            text = ""
            try:
                text = transcribe_audio_bytes(bts, f"audio{ext or '.ogg'}")
            except Exception as e:
                print("Transcribe error:", e)
                wa_send_text(from_number, "No pude transcribir el audio, ¿puedes repetirlo más claro?")
                return
            lang = detect_lang(text or "")
            if not text:
                wa_send_text(from_number, {"es":"Parece que no dijiste nada. ¿Puedes repetir?","en":"I didn't hear anything, could you repeat?"}.get(lang,"Ponlo de nuevo, por favor."))
                return
            answer = llm_answer(text, lang)
            wa_send_text(from_number, answer)
            return

        # Documentos (PDF) → resumir
        if mtype == "document":
            lang = "es"
            fname = msg["document"].get("filename", "documento.pdf")
            wa_send_text(from_number, f"📄 Recibí *{fname}*. Por ahora puedo darte un resumen si me indicas el tema clave.")
            return

# ---------------------------------------------------------------------------
# Fin del archivo
# ---------------------------------------------------------------------------

