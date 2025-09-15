# -*- coding: utf-8 -*-
# NochGPT WhatsApp – “todo en uno” estable (idiomas, botones, audio, imagen, tickets)
# ------------------------------------------------------------------------------------
# Endpoints:
#   GET  /                         -> ok
#   GET  /_debug/health            -> config sanity
#   GET  /webhook (verify)         -> verificación con Meta
#   POST /webhook                  -> mensajes entrantes
#
# Requiere ENV en Render:
#   OPENAI_API_KEY, WA_TOKEN, WA_PHONE_ID, APPS_SCRIPT_URL, (opcional) OPENAI_MODEL=gpt-4o-mini
#
# Notas:
# - Botones traducidos al idioma detectado (es, en, pt, fr, hi, ar, ru, ja, zh).
# - Audio: transcribe con Whisper.
# - Imagen: analiza con GPT-4o-mini (visión) vía base64 (no URLs firmadas).
# - Tickets a Google Sheets: cuando el usuario elige “Hablar con humano”.
# ------------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, base64, re, typing, mimetypes
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI(title="NochGPT WhatsApp")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ----------------- Config -----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
WA_TOKEN       = os.getenv("WA_TOKEN","")
WA_PHONE_ID    = os.getenv("WA_PHONE_ID","")
APPS_SCRIPT_URL= os.getenv("APPS_SCRIPT_URL","")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL","gpt-4o-mini")

if not OPENAI_API_KEY: print("⚠️ Falta OPENAI_API_KEY")
if not WA_TOKEN:       print("⚠️ Falta WA_TOKEN")
if not WA_PHONE_ID:    print("⚠️ Falta WA_PHONE_ID")
if not APPS_SCRIPT_URL:print("⚠️ Falta APPS_SCRIPT_URL")

client = OpenAI(api_key=OPENAI_API_KEY)

# Memorias simples
USER_LANG: dict[str,str] = {}          # +123.. -> 'es' / 'en' ...
USER_STATE: dict[str,str] = {}         # +123.. -> 'handoff_wait'
DEDUP: set[str] = set()                # mid vistos

# ----------------- Idiomas -----------------
# detección ligera por texto
def detect_lang(s: str) -> str:
    t = (s or "").lower()
    # atajos evidentes
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõç]", t):      return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    # saludos
    if re.search(r"\b(hola|buenas|qué tal)\b", t): return "es"
    if re.search(r"\b(hi|hello|hey)\b", t):        return "en"
    if re.search(r"\b(नमस्ते|नमस्कार)\b", t):       return "hi"
    if re.search(r"[أ-ي]", t):                     return "ar"
    if re.search(r"[А-Яа-я]", t):                  return "ru"
    if re.search(r"[ぁ-んァ-ン一-龯]", t):           return "ja"
    if re.search(r"[一-龥]", t):                    return "zh"
    return "en"

# traducciones básicas
TXT = {
    "hi": {
        "es": "¡Hola! ¿En qué puedo ayudarte hoy en temas dentales?\n\nElige una opción:",
        "en": "Hi! How can I help you today with dental topics?\n\nChoose an option:",
        "pt": "Olá! Como posso ajudar hoje com temas odontológicos?\n\nEscolha uma opção:",
        "fr": "Salut ! Comment puis-je vous aider aujourd’hui en dentaire ?\n\nChoisissez une option :",
        "hi": "नमस्ते! मैं दंत विषयों में आज आपकी कैसे मदद कर सकता हूँ?\n\nएक विकल्प चुनें:",
        "ar": "مرحبًا! كيف يمكنني مساعدتك اليوم في مواضيع طب الأسنان؟\n\nاختر خيارًا:",
        "ru": "Привет! Чем могу помочь по стоматологии сегодня?\n\nВыберите вариант:",
        "ja": "こんにちは！歯科のことで今日はどうお手伝いできますか？\n\nオプションを選んでください：",
        "zh": "你好！今天我如何在牙科方面帮助你？\n\n请选择：",
    },
    "buttons": {
        "es": ["Planes y precios","Tiempos","Hablar con humano"],
        "en": ["Plans & pricing","Turnaround times","Talk to a human"],
        "pt": ["Planos e preços","Prazos","Falar com humano"],
        "fr": ["Forfaits et prix","Délais","Parler à un humain"],
        "hi": ["प्लान और मूल्य","समय","मानव से बात"],
        "ar": ["الخطط والأسعار","المدد","التحدث إلى موظف"],
        "ru": ["Тарифы и цены","Сроки","Связаться с человеком"],
        "ja": ["プランと料金","納期","担当者と話す"],
        "zh": ["方案与价格","交付时间","转人工"],
    },
    "handoff_prompt": {
        "es": "👤 Te conecto con un asesor. Envía por favor:\n• Nombre\n• Tema (implante, zirconia, urgencia)\n• Horario preferido y teléfono si es otro",
        "en": "👤 I’ll connect you with a human. Please send:\n• Name\n• Topic (implant, zirconia, urgent)\n• Preferred time and phone if different",
        "pt": "👤 Vou te conectar com um atendente. Envie:\n• Nome\n• Tema (implante, zircônia, urgência)\n• Horário preferido e telefone se for outro",
        "fr": "👤 Je vous mets en relation avec un conseiller. Envoyez :\n• Nom\n• Sujet (implant, zircone, urgence)\n• Horaire préféré et téléphone si différent",
        "hi": "👤 मैं आपको मानव एजेंट से जोड़ रहा हूँ। कृपया भेजें:\n• नाम\n• विषय (इम्प्लांट, ज़िरकोनिया, आपात)\n• पसंदीदा समय और फोन (यदि अलग हो)",
        "ar": "👤 سأوصلك بمستشار بشري. أرسل من فضلك:\n• الاسم\n• الموضوع (زرعات، زركونيا، طارئ)\n• الوقت المفضل ورقم الهاتف إن كان مختلفًا",
        "ru": "👤 Соединяю с оператором. Отправьте:\n• Имя\n• Тема (имплант, цирконий, срочно)\n• Удобное время и телефон, если другой",
        "ja": "👤 担当者へお繋ぎします。以下を送ってください：\n• お名前\n• 内容（インプラント、ジルコニア、至急 など）\n• 希望時間と別連絡先があれば番号",
        "zh": "👤 我将为你转人工。请发送：\n• 姓名\n• 主题（种植体、氧化锆、紧急等）\n• 方便时间和如果不同的联系电话",
    },
    "handoff_ok": {
        "es": "✅ Tu solicitud fue registrada; un asesor te contactará pronto.",
        "en": "✅ Thanks. Your request was recorded; an agent will contact you soon.",
        "pt": "✅ Obrigado. Sua solicitação foi registrada; um atendente entrará em contato.",
        "fr": "✅ Merci. Votre demande a été enregistrée ; un conseiller vous contactera bientôt.",
        "hi": "✅ धन्यवाद। आपकी अनुरोध दर्ज हो गया है; एजेंट जल्द संपर्क करेगा।",
        "ar": "✅ شكرًا. تم تسجيل طلبك، سيتواصل معك مستشار قريبًا.",
        "ru": "✅ Готово. Запрос сохранён; с вами свяжется оператор.",
        "ja": "✅ 受付しました。担当者からご連絡します。",
        "zh": "✅ 已登记请求；客服稍后联系你。",
    },
    "audio_wait": {
        "es": "🎙️ Recibí tu audio, lo estoy transcribiendo…",
        "en": "🎙️ Got your audio, transcribing…",
        "pt": "🎙️ Recebi seu áudio, transcrevendo…",
        "fr": "🎙️ Audio reçu, transcription en cours…",
        "hi": "🎙️ आपका ऑडियो मिला, ट्रांसक्राइब कर रहा हूँ…",
        "ar": "🎙️ تم استلام المقطع الصوتي، جارٍ التفريغ…",
        "ru": "🎙️ Получил аудио, делаю расшифровку…",
        "ja": "🎙️ 音声を受け取りました。文字起こし中…",
        "zh": "🎙️ 收到你的语音，正在转写…",
    },
    "img_wait": {
        "es": "🖼️ Recibí tu imagen, la estoy analizando…",
        "en": "🖼️ Got your image, analyzing…",
        "pt": "🖼️ Recebi sua imagem, analisando…",
        "fr": "🖼️ Image reçue, analyse en cours…",
        "hi": "🖼️ आपकी छवि मिली, विश्लेषण कर रहा हूँ…",
        "ar": "🖼️ تم استلام الصورة، جارٍ التحليل…",
        "ru": "🖼️ Изображение получено, анализирую…",
        "ja": "🖼️ 画像を受け取りました。分析中…",
        "zh": "🖼️ 收到你的图片，正在分析…",
    }
}

def tr(key: str, lang: str) -> str | list[str]:
    lang = lang if lang in TXT[key] else "en"
    return TXT[key][lang]

# ----------------- WhatsApp helpers -----------------
WA_API = "https://graph.facebook.com/v20.0"

def wa_send_text(to: str, text: str):
    url = f"{WA_API}/{WA_PHONE_ID}/messages"
    payload = {"messaging_product":"whatsapp","to":to,"text":{"body":text}}
    r = requests.post(url, json=payload, headers={"Authorization":f"Bearer {WA_TOKEN}"}, timeout=20)
    r.raise_for_status(); return r.json()

def wa_send_buttons(to: str, lang: str):
    labels = tr("buttons", lang)
    url = f"{WA_API}/{WA_PHONE_ID}/messages"
    payload = {
        "messaging_product":"whatsapp",
        "to":to,
        "type":"interactive",
        "interactive":{
            "type":"button",
            "body":{"text": tr("hi", lang)},
            "action":{"buttons":[
                {"type":"reply","reply":{"id":"plans","title":labels[0]}},
                {"type":"reply","reply":{"id":"times","title":labels[1]}},
                {"type":"reply","reply":{"id":"human","title":labels[2]}},
            ]}
        }
    }
    r = requests.post(url, json=payload, headers={"Authorization":f"Bearer {WA_TOKEN}"}, timeout=20)
    r.raise_for_status(); return r.json()

def wa_media_url(media_id: str) -> str:
    # 1) obtener URL firmada
    r = requests.get(f"{WA_API}/{media_id}", headers={"Authorization":f"Bearer {WA_TOKEN}"}, timeout=20)
    r.raise_for_status()
    return r.json()["url"]

def wa_media_bytes(signed_url: str) -> bytes:
    # 2) descargar con el mismo token
    r = requests.get(signed_url, headers={"Authorization":f"Bearer {WA_TOKEN}"}, timeout=60)
    r.raise_for_status()
    return r.content

# ----------------- Google Sheets (Apps Script) -----------------
def save_ticket(payload: dict):
    if not APPS_SCRIPT_URL: return
    try:
        requests.post(APPS_SCRIPT_URL, json=payload, timeout=15)
    except Exception as e:
        print("Sheets error:", e)

# ----------------- OpenAI helpers -----------------
def transcribe_audio(b: bytes) -> str:
    # guardamos temporal y mandamos a Whisper
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=True) as tmp:
        tmp.write(b); tmp.flush()
        with open(tmp.name, "rb") as f:
            res = client.audio.transcriptions.create(model="whisper-1", file=f)
        return (res.text or "").strip()

def vision_caption(img_b64: str, lang: str) -> str:
    # img_b64: "data:image/jpeg;base64,...."
    prompt = {
        "es":"Eres un asistente dental. Resume en 3–5 puntos clínicos lo más relevante de la imagen.",
        "en":"You are a dental assistant. Summarize the most relevant clinical points from the image in 3–5 bullets.",
        "pt":"Você é um assistente odontológico. Resuma 3–5 pontos clínicos relevantes da imagem.",
        "fr":"Assistant dentaire : résume 3–5 points cliniques pertinents de l’image.",
        "hi":"आप एक डेंटल सहायक हैं। छवि से 3–5 मुख्य क्लिनिकल बिंदु बताएँ।",
        "ar":"أنت مساعد أسنان. لخّص 3–5 نقاط سريرية مهمة من الصورة.",
        "ru":"Вы стоматологический ассистент. Опишите 3–5 ключевых клинических пунктов на изображении.",
        "ja":"歯科アシスタントとして、画像から重要な臨床ポイントを3–5個にまとめてください。",
        "zh":"你是牙科助理。用3–5条总结这张图片的关键临床要点。"
    }.get(lang,"en")
    chat = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{
            "role":"user",
            "content":[
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":img_b64}}
            ]
        }],
        temperature=0.2
    )
    return chat.choices[0].message.content.strip()

# ----------------- Ruteo -----------------
@app.get("/")
def root(): return PlainTextResponse("Dental-LLM listo ✅")

@app.get("/_debug/health")
def health():
    cfg = {
        "openai": bool(OPENAI_API_KEY),
        "wa_token": bool(WA_TOKEN),
        "wa_phone_id": bool(WA_PHONE_ID),
        "model": OPENAI_MODEL,
        "sheet_webhook": bool(APPS_SCRIPT_URL),
    }
    return {"ok": all(cfg.values()) or cfg["openai"] and cfg["wa_token"] and cfg["wa_phone_id"], "cfg": cfg}

# Webhook verify (GET)
@app.get("/webhook")
def verify(mode: str | None = None, challenge: str | None = None, token: str | None = None):
    # usa el token que pegaste en Meta → Verify token (pon lo mismo aquí si quieres)
    VERIFY_TOKEN = os.getenv("VERIFY_TOKEN","nochgpt")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge or "")
    raise HTTPException(status_code=403, detail="forbidden")

# Webhook POST
@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    # WhatsApp envuelve en entry->changes->value->messages
    entry = (data.get("entry") or [{}])[0]
    changes = (entry.get("changes") or [{}])[0]
    value = changes.get("value") or {}
    messages = value.get("messages") or []
    if not messages:
        return {"ok": True}  # Solo status updates, etc.

    for m in messages:
        mid = m.get("id")
        if mid in DEDUP: 
            continue
        DEDUP.add(mid)

        frm = (m.get("from") or "").strip()   # número del usuario
        mtype = m.get("type")

        # idioma
        guess_text = ""
        if mtype == "text":    guess_text = m["text"]["body"]
        elif mtype == "button":guess_text = m["button"]["text"]
        elif mtype == "interactive":
            if "button_reply" in m["interactive"]:
                guess_text = m["interactive"]["button_reply"]["title"]
            elif "list_reply" in m["interactive"]:
                guess_text = m["interactive"]["list_reply"]["title"]
        lang = USER_LANG.get(frm) or detect_lang(guess_text)
        USER_LANG[frm] = lang

        # --- routing por tipo ---
        try:
            if mtype == "text":
                text = (m["text"]["body"] or "").strip()
                await handle_text(frm, text, lang)
            elif mtype == "audio":
                await handle_audio(frm, m["audio"]["id"], lang)
            elif mtype == "image":
                await handle_image(frm, m["image"]["id"], m["image"].get("mime_type"), lang)
            elif mtype in ("button","interactive"):
                await handle_interactive(frm, m, lang)
            elif mtype == "document":
                # Solo confirmación simple; luego podemos resumir PDF
                wa_send_text(frm, reply_lang("Recibí tu archivo PDF. Puedo resumirlo si lo deseas.", lang))
            else:
                wa_send_text(frm, reply_lang("Recibí tu mensaje. Por ahora entiendo mejor texto 😊", lang))
        except Exception as e:
            print("ERROR handling message:", e)
            wa_send_text(frm, reply_lang("Hubo un error temporal. Inténtalo de nuevo, por favor.", lang))

    return {"ok": True}

# Helpers de idioma para respuestas cortas
def reply_lang(es_text: str, lang: str) -> str:
    table = {
        "es": es_text,
        "en": "I received your message. For now I understand text best 😊",
        "pt": "Recebi sua mensagem. Por enquanto entendo melhor texto 😊",
        "fr": "J’ai bien reçu votre message. Pour l’instant je comprends mieux le texte 😊",
        "hi": "मुझे आपका संदेश मिला। अभी के लिए मैं पाठ बेहतर समझता हूँ 😊",
        "ar": "تلقيت رسالتك. حاليًا أفهم النصوص بشكل أفضل 😊",
        "ru": "Сообщение получил. Пока лучше всего понимаю текст 😊",
        "ja": "メッセージを受け取りました。今のところ文章が一番得意です 😊",
        "zh": "我收到了你的消息。目前我最擅长处理文字 😊",
    }
    return table.get(lang,"en" in table and table["en"] or es_text)

# ----------------- Handlers -----------------
async def handle_text(to: str, text: str, lang: str):
    # atajos: hola/saludos -> botones
    if re.search(r"\b(hola|hi|hello|oi|bonjour|नमस्ते)\b", text.lower()):
        wa_send_buttons(to, lang)
        return

    # si está esperando datos de handoff, guardamos ticket
    if USER_STATE.get(to) == "handoff_wait":
        save_ticket({
            "ts": int(time.time()),
            "from": to,
            "nombre": "",
            "tema": "",
            "contacto": to,
            "horario": "",
            "mensaje": text,
            "label": "NochGPT"
        })
        USER_STATE[to] = ""
        wa_send_text(to, tr("handoff_ok", lang))
        return

    # texto normal -> respuesta breve y dental-focus
    sys = ("You are NochGPT, a helpful dental laboratory assistant. "
           "Be concise and practical. Always respond in the user's language.")
    chat = client.chat.completions.create(
        model=OPENAI_MODEL, temperature=0.2,
        messages=[
            {"role":"system","content":sys},
            {"role":"user","content":f"[lang={lang}] {text}"}
        ]
    )
    wa_send_text(to, chat.choices[0].message.content.strip())

async def handle_audio(to: str, media_id: str, lang: str):
    wa_send_text(to, tr("audio_wait", lang))
    url  = wa_media_url(media_id)
    data = wa_media_bytes(url)
    try:
        text = transcribe_audio(data)
        if not text:
            wa_send_text(to, reply_lang("No pude transcribir el audio. ¿Puedes escribir un resumen?", lang))
            return
        await handle_text(to, text, lang)
    except Exception as e:
        print("audio error:", e)
        wa_send_text(to, reply_lang("No pude transcribir el audio. ¿Puedes escribir un resumen?", lang))

async def handle_image(to: str, media_id: str, mime: str | None, lang: str):
    wa_send_text(to, tr("img_wait", lang))
    url  = wa_media_url(media_id)
    data = wa_media_bytes(url)
    try:
        mt = mime or "image/jpeg"
        b64 = base64.b64encode(data).decode("utf-8")
        img_b64 = f"data:{mt};base64,{b64}"
        summary = vision_caption(img_b64, lang)
        wa_send_text(to, summary)
    except Exception as e:
        print("image error:", e)
        wa_send_text(to, reply_lang("Lo siento, hubo un problema analizando la imagen.", lang))

async def handle_interactive(to: str, m: dict, lang: str):
    # botones
    btn_id = None
    if "interactive" in m and "button_reply" in m["interactive"]:
        btn_id = m["interactive"]["button_reply"]["id"]
    elif "button" in m:
        btn_id = m["button"]["payload"]
    if not btn_id: 
        wa_send_text(to, reply_lang("Recibí tu selección.", lang))
        return

    if btn_id == "plans":
        # puedes poner aquí tus planes reales
        msg = {
            "es": "💳 *Planes*: Básico $50/mes · Pro $99/mes · Enterprise $299/mes.\n¿Deseas más detalles?",
            "en": "💳 *Plans*: Basic $50/mo · Pro $99/mo · Enterprise $299/mo.\nWant more details?",
            "pt": "💳 *Planos*: Básico $50/mês · Pro $99/mês · Enterprise $299/mês.\nQuer mais detalhes?",
        }.get(lang, "💳 Plans: Basic $50/mo · Pro $99/mo · Enterprise $299/mo.")
        wa_send_text(to, msg)
    elif btn_id == "times":
        msg = {
            "es":"⏱️ *Tiempos típicos*: Zirconia 3–5 días, Metal-cerámica 5–7, Implantes 7–10.",
            "en":"⏱️ *Typical times*: Zirconia 3–5 days, PFM 5–7, Implant cases 7–10.",
            "pt":"⏱️ *Prazos típicos*: Zircônia 3–5 dias, Metalocerâmica 5–7, Implantes 7–10.",
        }.get(lang, "⏱️ Typical times: Zirconia 3–5d, PFM 5–7d, Implants 7–10d.")
        wa_send_text(to, msg)
    elif btn_id == "human":
        USER_STATE[to] = "handoff_wait"
        wa_send_text(to, tr("handoff_prompt", lang))
    else:
        wa_send_text(to, reply_lang("Recibí tu selección.", lang))
