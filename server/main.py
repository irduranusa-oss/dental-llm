# server/main.py — NochGPT WhatsApp (ES/EN/PT/FR/RU/HI/AR/JA/ZH) + audio + imágenes + tickets
# -------------------------------------------------------------------------------------------
from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os, time, json, re, requests, base64, mimetypes

# ======= Config =======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WA_TOKEN       = os.getenv("WA_TOKEN", "")
WA_PHONE_ID    = os.getenv("WA_PHONE_ID", "")  # << MUY IMPORTANTE
VERIFY_TOKEN   = os.getenv("VERIFY_TOKEN", "nochgpt-verify")
GAS_TICKET_URL = os.getenv("GAS_TICKET_URL", "")  # Apps Script Web App URL

# Endpoints Meta (CORRECTO: incluye {WA_PHONE_ID})
WA_GRAPH_BASE  = "https://graph.facebook.com/v20.0"
WA_MSG_URL     = f"{WA_GRAPH_BASE}/{WA_PHONE_ID}/messages"
WA_MEDIA_URL   = f"{WA_GRAPH_BASE}/{{media_id}}"  # luego formateamos

# OpenAI
from openai import OpenAI
oa = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

# ======= FastAPI =======
app = FastAPI(title="NochGPT – WhatsApp Dental LLM")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ======= Utilidades idioma =======
EMOJI_RE = re.compile(r"[𐀀-􏿿]", flags=re.UNICODE)

def detect_lang(text: str) -> str:
    """Detección muy simple por caracteres/palabras. Devuelve código de idioma WhatsApp compatible."""
    t = (text or "").lower()
    # Español / portugués / francés / árabe / hindi / ruso / japonés / chino / inglés
    if re.search(r"[áéíóúñ¿¡]", t): return "es"      # Español
    if re.search(r"[ãõáéíóúç]", t):  return "pt"      # Portugués
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    if re.search(r"[اأإآء-ي]", t):    return "ar"
    if re.search(r"[अ-ह]", t):        return "hi"
    if re.search(r"[а-яё]", t):       return "ru"
    if re.search(r"[ぁ-んァ-ン一-龯]", t): return "ja"
    if re.search(r"[一-龥]", t):       return "zh"
    return "en"

# textos por idioma
T = {
    "es": {
        "hi": "¡Hola! ¿En qué puedo ayudarte hoy en el ámbito dental?\n\nElige una opción:",
        "plans": "Planes y precios",
        "times": "Tiempos de entrega",
        "human": "Hablar con humano",
        "audio_rx": "🎙️ Recibí tu audio, lo estoy transcribiendo…",
        "audio_fail": "No pude transcribir el audio. ¿Podrías escribir un resumen breve?",
        "img_rx": "🖼️ Recibí tu imagen, la estoy analizando…",
        "img_fail": "Lo siento, no pude analizar la imagen. Si quieres, cuéntame el caso.",
        "ticket_ack": "✅ Gracias. Tu solicitud fue registrada; un asesor te contactará pronto.",
        "handoff_ask": "👤 Te conecto con un asesor. Comparte:\n• Nombre\n• Tema (implante, zirconia, urgencia)\n• Horario y teléfono si es otro",
    },
    "en": {
        "hi": "Hi! How can I help you today with dental topics?\n\nChoose an option:",
        "plans": "Plans & pricing",
        "times": "Turnaround times",
        "human": "Talk to a human",
        "audio_rx": "🎙️ I received your audio, transcribing…",
        "audio_fail": "I couldn’t transcribe the audio. Could you type a short summary?",
        "img_rx": "🖼️ I received your image, analyzing…",
        "img_fail": "Sorry, I couldn’t analyze the image. If you want, describe the case.",
        "ticket_ack": "✅ Thanks. Your request was recorded; an agent will contact you soon.",
        "handoff_ask": "👤 I’ll connect you with an agent. Please share:\n• Name\n• Topic (implant, zirconia, urgency)\n• Preferred time and phone if different",
    },
    "pt": {"hi":"Oi! Como posso ajudar hoje com temas dentários?\n\nEscolha uma opção:",
           "plans":"Planos e preços","times":"Prazos","human":"Falar com humano",
           "audio_rx":"🎙️ Recebi seu áudio, transcrevendo…","audio_fail":"Não consegui transcrever. Pode digitar um resumo?",
           "img_rx":"🖼️ Recebi sua imagem, analisando…","img_fail":"Não consegui analisar a imagem.",
           "ticket_ack":"✅ Obrigado. Sua solicitação foi registrada.",
           "handoff_ask":"👤 Vou conectar você a um agente. Envie: nome, tema e horário."},
    "fr": {"hi":"Salut ! Comment puis-je t’aider aujourd’hui (dentaire) ?\n\nChoisis une option :",
           "plans":"Offres & tarifs","times":"Délais","human":"Parler à un humain",
           "audio_rx":"🎙️ Audio reçu, transcription en cours…","audio_fail":"Impossible de transcrire. Peux-tu écrire un résumé ?",
           "img_rx":"🖼️ Image reçue, analyse en cours…","img_fail":"Désolé, je n’ai pas pu analyser l’image.",
           "ticket_ack":"✅ Merci. Ta demande a été enregistrée.",
           "handoff_ask":"👤 Je te connecte à un conseiller. Indique : nom, sujet, horaire."},
    "ru": {"hi":"Привет! Чем могу помочь по стоматологии?\n\nВыберите вариант:",
           "plans":"Тарифы","times":"Сроки","human":"Связаться с оператором",
           "audio_rx":"🎙️ Получил аудио, расшифровываю…","audio_fail":"Не удалось расшифровать. Напишите кратко, пожалуйста.",
           "img_rx":"🖼️ Получил изображение, анализирую…","img_fail":"Не удалось проанализировать изображение.",
           "ticket_ack":"✅ Заявка зарегистрирована. Мы свяжемся с вами.",
           "handoff_ask":"👤 Соединю с оператором. Напишите имя, тему и удобное время."},
    "hi": {"hi":"नमस्ते! दंत विषयों में आज मैं आपकी कैसे मदद कर सकता हूँ?\n\nकृपया विकल्प चुनें:",
           "plans":"प्लान व कीमत","times":"टर्नअराउंड समय","human":"मानव से बात करें",
           "audio_rx":"🎙️ आपका ऑडियो मिला, ट्रांसक्राइब कर रहा हूँ…","audio_fail":"ऑडियो ट्रांसक्राइब नहीं कर सका। कृपया संक्षेप लिखें।",
           "img_rx":"🖼️ आपकी तस्वीर मिली, विश्लेषण कर रहा हूँ…","img_fail":"तस्वीर का विश्लेषण नहीं हो सका।",
           "ticket_ack":"✅ आपकी रिक्वेस्ट दर्ज हो गई है।",
           "handoff_ask":"👤 एजेंट से जोड़ रहा हूँ: नाम, विषय, समय बताएं."},
    "ar": {"hi":"مرحبًا! كيف أستطيع مساعدتك اليوم في مواضيع الأسنان؟\n\nاختر خيارًا:",
           "plans":"الخطط والأسعار","times":"أوقات التسليم","human":"التحدث إلى موظف",
           "audio_rx":"🎙️ استلمت المقطع الصوتي وأقوم بتفريغه…","audio_fail":"تعذر التفريغ. هل تكتب ملخصًا قصيرًا؟",
           "img_rx":"🖼️ استلمت الصورة وأقوم بتحليلها…","img_fail":"عذرًا، تعذر تحليل الصورة.",
           "ticket_ack":"✅ تم تسجيل طلبك وسنتواصل معك قريبًا.",
           "handoff_ask":"👤 سأوصلك بموظف. اذكر الاسم والموضوع والوقت المناسب."},
    "ja": {"hi":"こんにちは！歯科関連で今日は何をお手伝いできますか？\n\nオプションを選んでください：",
           "plans":"プランと料金","times":"納期","human":"担当者と話す",
           "audio_rx":"🎙️ 音声を受信、文字起こし中…","audio_fail":"文字起こしできませんでした。要点を送ってください。",
           "img_rx":"🖼️ 画像を受信、分析中…","img_fail":"画像を分析できませんでした。",
           "ticket_ack":"✅ 受付しました。担当者から連絡します。",
           "handoff_ask":"👤 担当者へ接続します。お名前・トピック・時間帯を教えてください。"},
    "zh": {"hi":"你好！关于牙科我今天可以如何帮助你？\n\n请选择：",
           "plans":"方案与价格","times":"制作周期","human":"人工客服",
           "audio_rx":"🎙️ 已收到语音，正在转写…","audio_fail":"无法转写语音，请简单文字说明。",
           "img_rx":"🖼️ 已收到图片，正在分析…","img_fail":"抱歉，无法分析该图片。",
           "ticket_ack":"✅ 已登记，稍后将联系你。",
           "handoff_ask":"👤 将为你接入客服。请提供：姓名、主题、时间。"},
}

def Tget(lang: str, key: str) -> str:
    base = T.get(lang) or T["en"]
    return base.get(key) or T["en"].get(key, "")

# ======= WhatsApp helpers =======
def wa_headers():
    return {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}

def wa_send_text(to: str, body: str):
    payload = {"messaging_product":"whatsapp","to":to,"type":"text","text":{"body":body}}
    r = requests.post(WA_MSG_URL, headers=wa_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        print("wa_send_text error:", r.status_code, r.text)
    return r.ok

def wa_send_typing(to: str, on=True):
    payload = {"messaging_product":"whatsapp","to":to,"type":"typing","typing":{"duration":"short" if on else "stop"}}
    # Si falla, ignoramos (no todos los clientes lo muestran)
    try: requests.post(WA_MSG_URL, headers=wa_headers(), json=payload, timeout=10)
    except: pass

def wa_send_menu(to: str, lang: str):
    # Botones "interactive" (reply buttons)
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": Tget(lang, "hi")},
            "action": {
                "buttons": [
                    {"type":"reply","reply":{"id":"plans","title":Tget(lang,"plans")}},
                    {"type":"reply","reply":{"id":"times","title":Tget(lang,"times")}},
                    {"type":"reply","reply":{"id":"human","title":Tget(lang,"human")}},
                ]
            }
        }
    }
    r = requests.post(WA_MSG_URL, headers=wa_headers(), json=payload, timeout=30)
    if r.status_code >= 400:
        print("wa_send_menu error:", r.status_code, r.text)
    return r.ok

def wa_get_media_url(media_id: str):
    # 1) obtener URL firmada
    r = requests.get(WA_MEDIA_URL.format(media_id=media_id), headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=30)
    if r.status_code >= 400:
        print("get_media meta error:", r.status_code, r.text); return None
    return r.json().get("url")

def wa_download(url: str):
    r = requests.get(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=60)
    if r.status_code >= 400:
        print("download media error:", r.status_code, r.text); return None, None
    ct = r.headers.get("Content-Type") or "application/octet-stream"
    return r.content, ct

# ======= Tickets (Google Apps Script) =======
def send_ticket_to_sheet(payload: dict):
    if not GAS_TICKET_URL: return False
    try:
        r = requests.post(GAS_TICKET_URL, data=json.dumps(payload), headers={"Content-Type":"application/json"}, timeout=20)
        if r.status_code >= 400:
            print("GAS ticket error:", r.status_code, r.text); return False
        return True
    except Exception as e:
        print("GAS ticket exception:", e); return False

# ======= Webhook verification (GET) =======
@app.get("/webhook")
def verify(mode: str = "", challenge: str = "", token: str = ""):
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(content=challenge or "OK", status_code=200)
    return PlainTextResponse("error", status_code=403)

# ======= Health (debug) =======
@app.get("/_debug/health")
def health():
    cfg = {"openai": bool(OPENAI_API_KEY), "wa_token": bool(WA_TOKEN), "wa_phone_id": bool(WA_PHONE_ID),
           "model": OPENAI_MODEL, "sheet_webhook": bool(GAS_TICKET_URL)}
    return {"ok": True, "cfg": cfg}

# ======= Webhook (POST) =======
@app.post("/webhook")
async def webhook(request: Request):
    body = await request.json()
    entry = (body.get("entry") or [{}])[0]
    changes = (entry.get("changes") or [{}])[0]
    value = changes.get("value") or {}
    messages = value.get("messages") or []
    statuses = value.get("statuses") or []

    # ACK de status (entregados, leídos) – opcional
    if statuses:
        return JSONResponse({"ok": True})

    if not messages:
        return JSONResponse({"ok": True})

    msg = messages[0]
    from_number = msg.get("from")
    msg_type = msg.get("type")

    # Texto: autodetecta idioma y responde con menú si dice "hola" o saluda
    if msg_type == "text":
        user_text = (msg.get("text") or {}).get("body","")
        lang = detect_lang(user_text)
        t = user_text.strip().lower()
        if t in ["hola","hola!","hi","hello","oi","bonjour","مرحبا","привет","नमस्ते","こんにちは","你好"]:
            wa_send_menu(from_number, lang)
            return JSONResponse({"ok": True})
        # Handoff: si el usuario escribe después de pedir humano, crea ticket
        if t.startswith("hablar con humano") or t.startswith("human") or t.startswith("asesor"):
            wa_send_text(from_number, Tget(lang,"handoff_ask"))
            return JSONResponse({"ok": True})
        # Si envía datos tipo "Nacho, zirconia 7-9, 6232310578 …" guardamos como ticket simple
        if any(k in t for k in ["implante","zircon","urgenc","whatsapp","tel","telefono","phone"]):
            ticket = {
                "ts": int(time.time()),
                "from": from_number,
                "nombre": "",
                "tema": user_text[:120],
                "contacto": from_number,
                "horario": "",
                "mensaje": user_text[:500],
                "label": "NochGPT"
            }
            send_ticket_to_sheet(ticket)
            wa_send_text(from_number, Tget(lang, "ticket_ack"))
            return JSONResponse({"ok": True})

        # Respuesta LLM breve y foco dental
        prompt = (
            "You are NochGPT, a helpful dental laboratory assistant. "
            "Answer briefly in the user's language. Stay on dental topics (prosthetics, zirconia, implants, CAD/CAM)."
        )
        resp = oa.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":prompt},
                      {"role":"user","content":user_text}],
            temperature=0.3
        )
        wa_send_text(from_number, resp.choices[0].message.content.strip())
        return JSONResponse({"ok": True})

    # Botones (interactive)
    if msg_type == "interactive":
        itype = (msg.get("interactive") or {}).get("type")
        if itype == "button_reply":
            reply = (msg["interactive"].get("button_reply") or {})
            choice = reply.get("id")
        elif itype == "list_reply":
            reply = (msg["interactive"].get("list_reply") or {})
            choice = reply.get("id")
        else:
            choice = ""

        lang = "en"
        # intentamos recuperar texto original para idioma
        if value.get("messages") and value["messages"][0].get("context",{}).get("body"):
            lang = detect_lang(value["messages"][0]["context"]["body"])

        if choice == "plans":
            wa_send_text(from_number,
                {"es":"Planes: IA con logo $500, Exocad experto $500, Meshmixer $50.",
                 "en":"Plans: AI with your logo $500, Exocad expert $500, Meshmixer $50.",
                 "pt":"Planos: IA com logo $500, Exocad $500, Meshmixer $50.",
                 "fr":"Offres: IA avec logo 500$, Exocad 500$, Meshmixer 50$.",
                 "ru":"Тарифы: ИИ с логотипом $500, Exocad $500, Meshmixer $50.",
                 "hi":"प्लान: लोगो सहित AI $500, Exocad $500, Meshmixer $50.",
                 "ar":"الخطط: ذكاء اصطناعي مع شعار $500، Exocad $500، Meshmixer $50.",
                 "ja":"プラン: ロゴ入りAI $500、Exocad $500、Meshmixer $50。",
                 "zh":"方案：含Logo的AI $500，Exocad $500，Meshmixer $50。"
                }.get(lang, "Plans: AI with logo $500, Exocad $500, Meshmixer $50.")
            )
        elif choice == "times":
            wa_send_text(from_number,
                {"es":"Tiempos típicos: zirconia 2-3 días, implante 4-6 días, urgencias el mismo día si hay espacio.",
                 "en":"Typical times: zirconia 2-3 days, implant 4-6 days, rush same-day if available."}.get(lang,
                 "Typical times: zirconia 2-3 days, implant 4-6 days, rush same-day if available.")
            )
        elif choice == "human":
            wa_send_text(from_number, Tget(lang,"handoff_ask"))
        else:
            wa_send_menu(from_number, lang)

        return JSONResponse({"ok": True})

    # Audio → transcripción
    if msg_type == "audio":
        lang = "en"
        wa_send_text(from_number, Tget(lang,"audio_rx"))
        media_id = (msg.get("audio") or {}).get("id")
        url = wa_get_media_url(media_id)
        if not url:
            wa_send_text(from_number, Tget(lang,"audio_fail")); return JSONResponse({"ok": True})
        data, ct = wa_download(url)
        if not data:
            wa_send_text(from_number, Tget(lang,"audio_fail")); return JSONResponse({"ok": True})

        # Enviar a Whisper
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
                f.write(data); f.flush()
                tr = oa.audio.transcriptions.create(model=WHISPER_MODEL, file=open(f.name, "rb"))
            text = tr.text.strip()
            lang = detect_lang(text)
            # Pasar al modelo para respuesta dental corta
            resp = oa.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":
                           "You are NochGPT, dental assistant. Reply concisely in user's language."},
                          {"role":"user","content":text}],
                temperature=0.3
            )
            wa_send_text(from_number, resp.choices[0].message.content.strip())
        except Exception as e:
            print("whisper error:", e)
            wa_send_text(from_number, Tget(lang,"audio_fail"))
        return JSONResponse({"ok": True})

    # Imágenes → análisis
    if msg_type == "image":
        lang = "en"
        wa_send_text(from_number, Tget(lang,"img_rx"))
        media_id = (msg.get("image") or {}).get("id")
        url = wa_get_media_url(media_id)
        if not url:
            wa_send_text(from_number, Tget(lang,"img_fail")); return JSONResponse({"ok": True})
        # GPT-4o con image_url
        try:
            resp = oa.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":"Analyze the dental photo. Describe relevant clinical/technical aspects briefly."},
                        {"type":"image_url","image_url":{"url":url}}
                    ]
                }],
                temperature=0.2
            )
            txt = resp.choices[0].message.content.strip()
            lang = detect_lang(txt)  # muy básico; si quieres fuerza a español con detect from_number context
            wa_send_text(from_number, txt)
        except Exception as e:
            print("vision error:", e); wa_send_text(from_number, Tget(lang,"img_fail"))
        return JSONResponse({"ok": True})

    # Documentos (PDF) → dejamos aviso corto (puedes ampliar luego)
    if msg_type == "document":
        wa_send_text(from_number,
            "📄 Recibí tu archivo. En esta versión analizo PDFs simples más tarde; por ahora dime qué necesitas y te ayudo."
        )
        return JSONResponse({"ok": True})

    # Cualquier otro tipo
    wa_send_text(from_number, "👍")
    return JSONResponse({"ok": True})
