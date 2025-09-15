# server/main.py — Hotfix 4 en 1 (idioma + botones + audio + fotos + tickets)
# ---------------------------------------------------------------------------
# Qué corrige:
# 1) Auto-detección de idioma por MENSAJE (no global) y menú en el idioma del usuario.
# 2) Transcripción de audios (WhatsApp OGG/MP3) con Whisper y respuesta en el mismo idioma.
# 3) Análisis básico de fotos con GPT-4o (visión) y respuesta en el mismo idioma.
# 4) Envío de ticket a Google Sheets vía Apps Script (si SHEET_WEBHOOK_URL está configurada).
#
# Requisitos de entorno (Render → Environment):
#   OPENAI_API_KEY           (obligatorio)
#   WA_TOKEN                 (obligatorio – token de la app de Meta)
#   WA_PHONE_ID              (obligatorio – phone number id de WhatsApp)
#   SHEET_WEBHOOK_URL        (opcional – URL del Apps Script para registrar tickets)
#   OPENAI_MODEL=gpt-4o-mini (por defecto)
#   OPENAI_TEMP=0.2          (por defecto)
#
# Endpoints útiles:
#   GET  /_debug/health   -> estado y variables clave
#   GET  /handoff         -> tickets locales (respaldo en /tmp/handoff.json)
#   POST /webhook         -> webhook de Meta
# ---------------------------------------------------------------------------

from __future__ import annotations
import os, io, time, json, base64, re, typing, mimetypes, requests, pathlib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Dental-LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Config ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WA_TOKEN        = os.getenv("WA_TOKEN", "")
WA_PHONE_ID     = os.getenv("WA_PHONE_ID", "")
SHEET_WEBHOOK   = os.getenv("SHEET_WEBHOOK_URL", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMP     = float(os.getenv("OPENAI_TEMP", "0.2"))

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "You are NochGPT, a helpful dental laboratory assistant. "
    "Focus strictly on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials). "
    "Be concise and practical. Always reply in the same language as the user."
)

# ===== Utilidades de idioma =====
def detect_lang(t: str) -> str:
    s = (t or "").strip().lower()
    # Atajos por palabras comunes
    if re.search(r"[áéíóúñ¿¡]|(hola|gracias|buenas|cotizar|implante|zirconia)", s): return "es"
    if re.search(r"[çãõ]|(olá|obrigado)", s): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]|(bonjour|merci)", s): return "fr"
    if re.search(r"[\u0400-\u04FF]|(спасибо|привет)", s): return "ru"  # cirílico
    if re.search(r"[\u0900-\u097F]", s): return "hi"                   # devanagari (hindi)
    if re.search(r"[\u0600-\u06FF]", s): return "ar"                   # árabe
    if re.search(r"[\u4e00-\u9fff]", s): return "zh"                   # chino
    if re.search(r"[\u3040-\u30ff]", s): return "ja"                   # japonés
    return "en"

# Textos por idioma
TEXTS = {
    "es": {
        "hi": "¡Hola! ¿En qué puedo ayudarte hoy en el ámbito dental?\n\nElige una opción:",
        "menu": ["Planes y precios", "Tiempos de entrega", "Hablar con humano"],
        "audio_wait": "🎙️ Recibí tu audio, lo estoy transcribiendo…",
        "audio_fail": "No pude transcribir el audio. ¿Podrías escribir un breve resumen?",
        "img_wait": "🖼️ Recibí tu imagen, la estoy analizando…",
        "img_fail": "Lo siento, no pude analizar la imagen. ¿Puedes describirme qué deseas revisar?",
        "handoff_ask": ("👤 Te conecto con un asesor. Comparte por favor:\n"
                        "• Nombre\n• Tema (implante, zirconia, urgencia)\n"
                        "• Horario preferido y teléfono si es otro"),
        "handoff_ok": "✅ Gracias. Tu solicitud fue registrada; un asesor te contactará pronto.",
    },
    "en": {
        "hi": "Hi! How can I help you today with dental topics?\n\nChoose an option:",
        "menu": ["Plans & pricing", "Turnaround times", "Talk to a human"],
        "audio_wait": "🎙️ I received your audio, transcribing…",
        "audio_fail": "I couldn’t transcribe the audio. Could you type a short summary?",
        "img_wait": "🖼️ I received your image, analyzing…",
        "img_fail": "Sorry, I couldn't analyze the image. Can you describe what you need?",
        "handoff_ask": ("👤 I’ll connect you with an agent. Please share:\n"
                        "• Name\n• Topic (implant, zirconia, urgent)\n"
                        "• Preferred time and phone if different"),
        "handoff_ok": "✅ Thanks. Your request was recorded; an agent will contact you soon.",
    },
    "pt": {
        "hi": "Olá! Como posso ajudar hoje com temas dentários?\n\nEscolha uma opção:",
        "menu": ["Planos e preços", "Prazos", "Falar com humano"],
        "audio_wait": "🎙️ Recebi seu áudio, transcrevendo…",
        "audio_fail": "Não consegui transcrever o áudio. Pode escrever um resumo?",
        "img_wait": "🖼️ Recebi sua imagem, analisando…",
        "img_fail": "Desculpe, não consegui analisar a imagem. Pode descrever?",
        "handoff_ask": ("👤 Vou conectá-lo a um atendente. Envie:\n"
                        "• Nome\n• Tema (implante, zircônia, urgência)\n"
                        "• Horário preferido e telefone se outro"),
        "handoff_ok": "✅ Obrigado. Sua solicitação foi registrada; entraremos em contato.",
    },
    "fr": {
        "hi": "Bonjour ! Comment puis-je vous aider aujourd’hui en dentaire ?\n\nChoisissez une option :",
        "menu": ["Offres & tarifs", "Délais", "Parler à un humain"],
        "audio_wait": "🎙️ J’ai reçu votre audio, transcription…",
        "audio_fail": "Je n’ai pas pu transcrire l’audio. Pouvez-vous écrire un résumé ?",
        "img_wait": "🖼️ Image reçue, analyse…",
        "img_fail": "Désolé, je n’ai pas pu analyser l’image. Décrivez-moi ce que vous voulez.",
        "handoff_ask": ("👤 Je vous mets en relation avec un conseiller. Donnez :\n"
                        "• Nom\n• Sujet (implant, zircone, urgent)\n"
                        "• Horaire préféré et téléphone si différent"),
        "handoff_ok": "✅ Merci. Votre demande a été enregistrée ; un conseiller vous contactera.",
    },
    "ru": {
        "hi": "Привет! Чем могу помочь по стоматологии?\n\nВыберите опцию:",
        "menu": ["Планы и цены", "Сроки", "Связаться с человеком"],
        "audio_wait": "🎙️ Получил аудио, расшифровываю…",
        "audio_fail": "Не удалось расшифровать аудио. Напишите кратко текстом?",
        "img_wait": "🖼️ Изображение получено, анализ…",
        "img_fail": "Не удалось проанализировать изображение. Опишите задачу текстом?",
        "handoff_ask": "👤 Соединю с оператором. Укажите имя, тему и удобное время.",
        "handoff_ok": "✅ Заявка принята. Мы свяжемся с вами.",
    },
    "hi": {
        "hi": "नमस्ते! दंत विषयों में आज मैं कैसे मदद कर सकता हूँ?\n\nएक विकल्प चुनें:",
        "menu": ["प्लान और कीमत", "टर्नअराउंड टाइम", "मानव से बात करें"],
        "audio_wait": "🎙️ आपका ऑडियो मिला, लिप्यंतरण कर रहा हूँ…",
        "audio_fail": "ऑडियो लिप्यंतरण नहीं हो सका. कृपया एक छोटा सार लिखें।",
        "img_wait": "🖼️ आपकी छवि मिली, विश्लेषण कर रहा हूँ…",
        "img_fail": "छवि का विश्लेषण नहीं हो सका. कृपया बताएं क्या चाहिए।",
        "handoff_ask": "👤 मैं आपको एजेंट से जोड़ूँगा. नाम, विषय, पसंदीदा समय भेजें।",
        "handoff_ok": "✅ अनुरोध दर्ज हो गया. हम आपसे संपर्क करेंगे।",
    },
    "ar": {
        "hi": "مرحبًا! كيف أساعدك اليوم في مواضيع طب الأسنان؟\n\nاختر خيارًا:",
        "menu": ["الخطط والأسعار", "أوقات التسليم", "التحدث إلى موظف"],
        "audio_wait": "🎙️ استلمت المقطع الصوتي وأقوم بكتابته…",
        "audio_fail": "تعذر تفريغ الصوت. هل يمكنك كتابة ملخص قصير؟",
        "img_wait": "🖼️ استلمت الصورة وأقوم بتحليلها…",
        "img_fail": "عذرًا، لم أتمكن من تحليل الصورة. صف ما تريد فحصه.",
        "handoff_ask": "👤 سأوصلك بمستشار. أرسل الاسم والموضوع والوقت المفضل.",
        "handoff_ok": "✅ تم تسجيل طلبك، سيتواصل معك مستشار قريبًا.",
    },
    "zh": {
        "hi": "你好！今天在牙科方面我能帮你什么？\n\n请选择：",
        "menu": ["方案与价格", "交付时间", "联系人工"],
        "audio_wait": "🎙️ 收到你的语音，正在转写…",
        "audio_fail": "无法转写语音。请简要用文字说明。",
        "img_wait": "🖼️ 收到你的图片，正在分析…",
        "img_fail": "抱歉，无法分析图片。请用文字描述你的需求。",
        "handoff_ask": "👤 我将为你联系人工。请提供姓名、主题和方便时间。",
        "handoff_ok": "✅ 已登记你的请求，稍后会有工作人员联系你。",
    },
    "ja": {
        "hi": "こんにちは！歯科分野で今日は何をお手伝いできますか？\n\nオプションを選んでください：",
        "menu": ["プランと料金", "納期", "担当者と話す"],
        "audio_wait": "🎙️ 音声を受信、文字起こし中…",
        "audio_fail": "文字起こしに失敗しました。要点をテキストで送ってください。",
        "img_wait": "🖼️ 画像を受信、解析中…",
        "img_fail": "画像を解析できませんでした。内容をテキストで教えてください。",
        "handoff_ask": "👤 担当者におつなぎします。お名前・内容・希望時間を送ってください。",
        "handoff_ok": "✅ 受付しました。担当者よりご連絡します。",
    },
}

def T(lang: str, key: str) -> str:
    d = TEXTS.get(lang) or TEXTS["en"]
    return d.get(key, TEXTS["en"][key])

# ===== WhatsApp helpers =====
def wa_send_json(payload: dict):
    url = f"https://graph.facebook.com/v21.0/{WA_PHONE_ID}/messages"
    h = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type":"application/json"}
    r = requests.post(url, headers=h, json=payload, timeout=30)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}

def wa_send_text(to: str, text: str):
    return wa_send_json({"messaging_product":"whatsapp","to":to,"type":"text","text":{"body":text}})

def wa_send_menu(to: str, lang: str):
    # Botones localizados
    a,b,c = TEXTS.get(lang, TEXTS["en"])["menu"]
    body = T(lang,"hi")
    payload = {
        "messaging_product":"whatsapp","to":to,"type":"interactive",
        "interactive":{
            "type":"button",
            "body":{"text": body},
            "action":{"buttons":[
                {"type":"reply","reply":{"id":"plans","title":a}},
                {"type":"reply","reply":{"id":"tat","title":b}},
                {"type":"reply","reply":{"id":"human","title":c}},
            ]}
        }
    }
    return wa_send_json(payload)

def wa_media_url(media_id: str) -> str:
    url = f"https://graph.facebook.com/v21.0/{media_id}"
    h = {"Authorization": f"Bearer {WA_TOKEN}"}
    r = requests.get(url, headers=h, timeout=30)
    r.raise_for_status()
    return r.json()["url"]

def wa_media_bytes(signed_url: str) -> bytes:
    h = {"Authorization": f"Bearer {WA_TOKEN}"}
    r = requests.get(signed_url, headers=h, timeout=60)
    r.raise_for_status()
    return r.content

# ===== Tickets (Google Sheets) =====
HANDOFF_FILE = "/tmp/handoff.json"
def save_local_ticket(item: dict):
    data = []
    if pathlib.Path(HANDOFF_FILE).exists():
        try:
            data = json.loads(pathlib.Path(HANDOFF_FILE).read_text("utf-8"))
        except Exception:
            data = []
    data.append(item)
    pathlib.Path(HANDOFF_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def send_ticket_sheet(item: dict):
    if not SHEET_WEBHOOK:
        return False, "no SHEET_WEBHOOK_URL"
    try:
        r = requests.post(SHEET_WEBHOOK, json=item, timeout=30)
        ok = (200 <= r.status_code < 300)
        return ok, r.text
    except Exception as e:
        return False, str(e)

# ===== Flow helpers =====
def analyze_text(user_text: str, lang: str) -> str:
    # Respuesta breve, dental-only, en el idioma detectado
    msg = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":user_text}
    ]
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=OPENAI_TEMP, messages=msg
    )
    return resp.choices[0].message.content.strip()

def analyze_image(img_bytes: bytes, lang: str) -> str:
    b64 = base64.b64encode(img_bytes).decode()
    msg = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":[
            {"type":"text","text":"Describe brevemente la situación clínica en la imagen y sugiere pasos/precauciones. Responde en "+lang},
            {"type":"input_image","image_data": b64}
        ]}
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", temperature=OPENAI_TEMP, messages=msg)
    return resp.choices[0].message.content.strip()

def transcribe_audio(audio_bytes: bytes) -> str:
    # Whisper espera un archivo. Enviamos bytes como archivo temporal en memoria.
    # Render usa python3.11, openai==1.x lo soporta.
    fileobj = io.BytesIO(audio_bytes); fileobj.name = "audio.ogg"
    tr = client.audio.transcriptions.create(model="whisper-1", file=fileobj)
    return tr.text.strip()

# ===== Estados simples por usuario =====
STATE = {}  # from -> "waiting_human"

def handle_text(from_number: str, body: str):
    lang = detect_lang(body)
    txt = body.strip().lower()

    # palabra de arranque -> mostrar menú
    if txt in {"hola","menu","hi","hello","inicio","start"}:
        wa_send_menu(from_number, lang)
        return True

    # pedir humano
    if txt in {"humano","hablar con humano","asesor","agente","human"}:
        STATE[from_number] = "waiting_human"
        wa_send_text(from_number, T(lang,"handoff_ask"))
        return True

    # completar ticket si estaba esperando
    if STATE.get(from_number) == "waiting_human":
        item = {
            "ts": int(time.time()), "label":"NochGPT",
            "from": from_number, "nombre":"", "tema":"", "contacto": from_number,
            "horario":"", "mensaje": body
        }
        save_local_ticket(item)
        send_ticket_sheet(item)
        wa_send_text(from_number, T(lang,"handoff_ok"))
        STATE[from_number] = "done"
        return True

    # Respuesta LLM normal
    answer = analyze_text(body, lang)
    wa_send_text(from_number, answer)
    return True

# ===== FastAPI =====
@app.get("/_debug/health")
def health():
    return {"ok": True, "cfg":{
        "openai": bool(OPENAI_API_KEY), "wa_token": bool(WA_TOKEN),
        "wa_phone_id": bool(WA_PHONE_ID), "model": OPENAI_MODEL,
        "sheet_webhook": bool(SHEET_WEBHOOK)
    }}

@app.get("/handoff")
def handoff_list():
    if not pathlib.Path(HANDOFF_FILE).exists():
        return []
    return json.loads(pathlib.Path(HANDOFF_FILE).read_text("utf-8"))

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    try:
        entry = data.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value  = change.get("value", {})
        msgs   = value.get("messages", [])
        if not msgs:
            return JSONResponse({"status":"ok"})  # status updates, etc.

        msg = msgs[0]
        from_number = msg.get("from")
        msg_type = msg.get("type")

        # --- TEXTO ---
        if msg_type == "text":
            body = msg["text"]["body"]
            handle_text(from_number, body)
            return JSONResponse({"status":"ok"})

        # --- BOTONES (interactive) ---
        if msg_type == "interactive":
            lang = "es"  # pequeño truco: si vienen de botón, mantenemos ES por defecto
            btn = msg.get("interactive", {}).get("button_reply") or {}
            btn_id = btn.get("id","")
            if btn_id == "plans":
                # Tu mensaje/plantilla de planes:
                wa_send_text(from_number, "💳 Planes: Básico $50, Pro $150, Enterprise $500.\nEscríbeme qué necesitas y te cotizo.")
            elif btn_id == "tat":
                wa_send_text(from_number, "⏱️ Tiempos típicos: Zirconia 2–4 días, Implantes 5–7 días. Consulta disponibilidad.")
            elif btn_id == "human":
                STATE[from_number] = "waiting_human"
                wa_send_text(from_number, T(lang,"handoff_ask"))
            return JSONResponse({"status":"ok"})

        # --- AUDIO ---
        if msg_type == "audio":
            wa_send_text(from_number, T(detect_lang("hola"), "audio_wait"))
            media_id = msg["audio"]["id"]
            url = wa_media_url(media_id)
            blob = wa_media_bytes(url)
            try:
                text = transcribe_audio(blob)
                handle_text(from_number, text)  # reutiliza lógica (idioma del texto)
            except Exception:
                wa_send_text(from_number, T(detect_lang("en"), "audio_fail"))
            return JSONResponse({"status":"ok"})

        # --- IMAGEN ---
        if msg_type == "image":
            lang = detect_lang("hola")  # por defecto español si no tenemos pista
            wa_send_text(from_number, T(lang, "img_wait"))
            media_id = msg["image"]["id"]
            url = wa_media_url(media_id)
            blob = wa_media_bytes(url)
            try:
                out = analyze_image(blob, lang)
                wa_send_text(from_number, out)
            except Exception:
                wa_send_text(from_number, T(lang, "img_fail"))
            return JSONResponse({"status":"ok"})

        # --- DOCUMENTO (PDF u otros) -> por ahora: texto de recibo (resumen después)
        if msg_type == "document":
            wa_send_text(from_number, "📄 Recibí tu archivo. Puedo darte un resumen de texto si me dices qué parte te interesa.")
            return JSONResponse({"status":"ok"})

        # otros tipos
        wa_send_text(from_number, "I received your message. For now I understand text best 😉")
        return JSONResponse({"status":"ok"})

    except Exception as e:
        return JSONResponse({"status":"error","detail":str(e)})

@app.get("/")
def home():
    return HTMLResponse("<b>Dental-LLM corriendo ✅</b>")
