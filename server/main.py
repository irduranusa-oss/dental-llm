# server/main.py — NochGPT v3 (WhatsApp + Handoff + Planes/Precios + LLM + Panel)
# ---------------------------------------------------------------------------------
# ✅ Webhook Meta (GET verify + POST mensajes)
# ✅ Botones "Planes/Precios", "Sitio web", "Hablar con humano"
# ✅ Flujo "Hablar con humano": guarda tickets en /tmp/handoff.json y confirma
# ✅ Dedupe por message_id (evita respuestas dobles)
# ✅ Healthcheck /_debug/health, listado /tickets, JSON /handoff y panel /panel
# ✅ LLM fallback (OpenAI) que responde en el idioma del usuario
# ✅ Detección simple de idioma (es/en/pt/fr/ar/hi/ru/zh/ja)
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, re, pathlib, mimetypes, base64, typing, requests
from collections import deque
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel

# ---------- Config ----------
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL     = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WA_TOKEN         = os.getenv("WHATSAPP_TOKEN", "")
WA_PHONE_ID      = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN= os.getenv("META_VERIFY_TOKEN", "")
PLANS_URL        = os.getenv("PLANS_URL", "https://www.dentodo.com/plans-pricing")

HANDOFF_FILE     = "/tmp/handoff.json"
SEEN_MSGS        = deque(maxlen=500)   # de-dup message_id
USER_STATE: dict[str,str] = {}         # estado por número (waiting_handoff/done)
LAST_LANG: dict[str,str] = {}          # último idioma detectado por número

# ---------- App ----------
app = FastAPI(title="Dental-LLM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Utils ----------
def now_ts() -> int: return int(time.time())

def read_handoff() -> list[dict]:
    if not os.path.exists(HANDOFF_FILE): return []
    try:
        with open(HANDOFF_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return []

def append_handoff(record: dict):
    data = read_handoff()
    data.append(record)
    try:
        with open(HANDOFF_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("⚠️ Error guardando handoff:", e)

def detect_lang(s: str) -> str:
    s = (s or "").strip()
    # bloques Unicode básicos (muy simple pero útil)
    if re.search(r"[\u0600-\u06FF]", s): return "ar"  # árabe
    if re.search(r"[\u0900-\u097F]", s): return "hi"  # hindi
    if re.search(r"[\u0400-\u04FF]", s): return "ru"  # cirílico
    if re.search(r"[\u4E00-\u9FFF]", s): return "zh"  # chino
    if re.search(r"[\u3040-\u30FF]", s): return "ja"  # japonés
    sL = s.lower()
    if re.search(r"[áéíóúñ¿¡]", sL): return "es"
    if re.search(r"[ãõáéíóúç]", sL): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", sL): return "fr"
    return "en"

def same_lang_reply(lang: str, text_map: dict) -> str:
    return text_map.get(lang, text_map.get("en"))

# ---------- WhatsApp senders ----------
def wa_api_url() -> str:
    if not WA_PHONE_ID: return ""
    return f"https://graph.facebook.com/v19.0/{WA_PHONE_ID}/messages"

def wa_send(payload: dict):
    if not (WA_TOKEN and WA_PHONE_ID):
        print("⚠️ Falta WHATSAPP_TOKEN o WHATSAPP_PHONE_ID")
        return
    try:
        r = requests.post(
            wa_api_url(),
            headers={"Authorization": f"Bearer {WA_TOKEN}",
                     "Content-Type": "application/json"},
            json=payload, timeout=20
        )
        print("WA SEND:", r.status_code, r.text[:300])
    except Exception as e:
        print("WA error:", e)

def wa_send_text(to: str, body: str):
    wa_send({"messaging_product":"whatsapp","to":to,"type":"text","text":{"body":body}})

def wa_send_buttons_prices(to: str, lang: str):
    T = {
        "es": {"body":"¿Qué te gustaría hacer?","b1":"Ver planes","b2":"Sitio web","b3":"Hablar con humano"},
        "en": {"body":"What would you like to do?","b1":"See plans","b2":"Website","b3":"Talk to human"},
        "pt": {"body":"O que você gostaria de fazer?","b1":"Ver planos","b2":"Site","b3":"Falar com humano"},
        "fr": {"body":"Que veux-tu faire ?","b1":"Voir les plans","b2":"Site web","b3":"Parler à humain"},
        "ru": {"body":"Что вы хотите сделать?","b1":"Тарифы","b2":"Сайт","b3":"Оператор"},
        "ar": {"body":"ماذا تريد أن تفعل؟","b1":"الخطط","b2":"الموقع","b3":"التحدث لشخص"},
        "hi": {"body":"आप क्या करना चाहेंगे?","b1":"प्लान देखें","b2":"वेबसाइट","b3":"मानव से बात"},
        "zh": {"body":"你想做什么？","b1":"查看方案","b2":"网站","b3":"人工客服"},
        "ja": {"body":"何をしますか？","b1":"プランを見る","b2":"サイト","b3":"人と話す"},
    }
    t = T.get(lang, T["en"])
    payload = {
        "messaging_product":"whatsapp","to":to,"type":"interactive",
        "interactive":{
            "type":"button","body":{"text":t["body"]},
            "action":{"buttons":[
                {"type":"reply","reply":{"id":"planes","title":t["b1"]}},
                {"type":"reply","reply":{"id":"web","title":t["b2"]}},
                {"type":"reply","reply":{"id":"humano","title":t["b3"]}},
            ]}
        }
    }
    wa_send(payload)

# ---------- LLM (OpenAI) ----------
def llm_answer(user_text: str, lang: str) -> str:
    if not OPENAI_API_KEY:
        return same_lang_reply(lang, {
            "es":"Configuración incompleta: falta OPENAI_API_KEY.",
            "en":"Setup incomplete: missing OPENAI_API_KEY.",
            "pt":"Falta OPENAI_API_KEY.",
        })
    import openai  # sdk oficial retro-compatible
    openai.api_key = OPENAI_API_KEY
    system = (
        "You are NochGPT, a helpful dental laboratory assistant. "
        "Focus strictly on dental topics (prosthetics, implants, zirconia, CAD/CAM). "
        "Always reply in the user's language."
    )
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":user_text}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("LLM error:", e)
        return same_lang_reply(lang, {
            "es":"No pude generar respuesta ahora.",
            "en":"I couldn’t generate a reply right now.",
            "pt":"Não consegui responder agora.",
        })

# ---------- Interceptores de texto ----------
PRICE_WORDS = r"\b(precio|precios|plan(es)?|cotizar|tienda|prices?|plans?)\b"

def handle_text(from_num: str, text: str) -> bool:
    """Devuelve True si ya respondió y NO debe seguir al LLM."""
    lang = detect_lang(text)
    LAST_LANG[from_num] = lang
    t = text.strip().lower()

    # Botón: planes/precios
    if re.search(PRICE_WORDS, t):
        wa_send_buttons_prices(from_num, lang)
        return True

    # Handoff: pedir humano
    if "hablar con humano" in t or "human" in t or "humano" in t:
        USER_STATE[from_num] = "waiting_handoff"
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"👤 Te conecto con un asesor. Comparte: nombre, tema y horario (y teléfono si es otro).",
            "en":"👤 I’ll connect you with a human. Please send: name, topic and preferred time (and phone if different).",
            "pt":"👤 Vou te conectar a um assessor. Envie: nome, tema e horário (e telefone, se outro).",
            "fr":"👤 Je te mets en contact. Envoie : nom, sujet et horaire (téléphone si différent).",
            "ru":"👤 Соединю с консультантом. Напишите: имя, тема и время (и телефон, если другой).",
            "ar":"👤 سأوصلك بمستشار. أرسل: الاسم، الموضوع والوقت المفضل (ورقم مختلف إن وُجد).",
            "hi":"👤 मैं आपको सलाहकार से जोड़ूँगा। भेजें: नाम, विषय और समय (दूसरा फोन हो तो).",
            "zh":"👤 我将为你转人工。请发送：姓名、主题和时间（若有其他电话也请写）。",
            "ja":"👤 担当者にお繋ぎします。お名前・用件・希望時間（別番号があれば）を送ってください。",
        }))
        return True

    # Si usuario está completando datos del ticket
    if USER_STATE.get(from_num) == "waiting_handoff":
        record = {
            "ts": now_ts(), "label":"NochGPT",
            "from": from_num,
            "nombre":"", "tema": text[:120], "contacto": from_num,
            "horario":"", "mensaje": text
        }
        append_handoff(record)
        USER_STATE[from_num] = "done"
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"✅ Gracias. Tu solicitud fue registrada; un asesor te contactará en breve.",
            "en":"✅ Thanks. Your request was recorded; an agent will contact you soon.",
            "pt":"✅ Obrigado. Seu pedido foi registrado; entraremos em contato.",
            "fr":"✅ Merci. Votre demande a été enregistrée ; un agent vous contactera bientôt.",
            "ru":"✅ Спасибо. Заявка записана; с вами свяжется специалист.",
            "ar":"✅ شكرًا. تم تسجيل طلبك وسيتواصل معك مستشار قريبًا.",
            "hi":"✅ धन्यवाद. आपका अनुरोध दर्ज हो गया; कोई प्रतिनिधि आपसे संपर्क करेगा।",
            "zh":"✅ 感谢。已登记请求；稍后会有同事联系你。",
            "ja":"✅ ありがとうございます。担当よりご連絡します。",
        }))
        return True

    return False  # que siga al LLM

# ---------- Rutas públicas ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return "<b>Dental-LLM corriendo ✅</b>"

@app.get("/_debug/health")
def health():
    cfg = {
        "openai": bool(OPENAI_API_KEY),
        "wa_token": bool(WA_TOKEN),
        "wa_phone_id": bool(WA_PHONE_ID),
        "model": OPENAI_MODEL,
        "sheet_webhook": False,
        "switch_number": False
    }
    return {"ok": True, "cfg": cfg}

@app.get("/handoff")
def get_handoff():
    return read_handoff()

@app.get("/tickets")
def get_tickets():
    return read_handoff()

@app.get("/panel", response_class=HTMLResponse)
def panel():
    rows = read_handoff()[::-1]
    html = [
        "<html><head><meta charset='utf-8'><title>Tickets – NochGPT</title>",
        "<style>body{font-family:system-ui;background:#0b1b3f;color:#fff}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #234}th{background:#123}tr:hover{background:#112}</style>",
        "</head><body>",
        "<h2>Tickets – NochGPT <span style='background:#1b5fff;padding:6px 10px;border-radius:10px'>"
        f"{len(rows)} activos</span></h2>",
        "<small>Se actualiza al refrescar · Origen: /tmp/handoff.json</small>",
        "<table><thead><tr><th>Fecha/Hora</th><th>Número</th><th>Nombre</th><th>Tema</th><th>Contacto</th></tr></thead><tbody>"
    ]
    for r in rows:
        dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("ts", now_ts())))
        html.append(f"<tr><td>{dt}</td><td>{r.get('from','')}</td><td>{r.get('nombre','')}</td>"
                    f"<td>{r.get('tema','')}</td><td>{r.get('contacto','')}</td></tr>")
    html.append("</tbody></table></body></html>")
    return "\n".join(html)

# ---------- Webhook Meta (GET verify + POST inbound) ----------
@app.get("/webhook")
def webhook_verify(request: Request):
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    if mode == "subscribe" and token == META_VERIFY_TOKEN:
        return PlainTextResponse(challenge or "")
    raise HTTPException(status_code=403, detail="Forbidden")

@app.post("/webhook")
async def webhook_post(req: Request):
    data = await req.json()
    print("Payload recibido:", str(data)[:800])

    entry = (data.get("entry") or [{}])[0]
    changes = (entry.get("changes") or [{}])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [])
    if not messages:
        return {"status":"ok"}  # acks de entrega, etc.

    for m in messages:
        msg_id = m.get("id")
        if msg_id in SEEN_MSGS:
            print("👀 Duplicado, ignorado:", msg_id)
            continue
        SEEN_MSGS.append(msg_id)

        from_num = m.get("from")
        msg_type = m.get("type")

        # 1) Respuesta a botones
        if msg_type == "interactive":
            inter = m.get("interactive", {})
            if inter.get("type") == "button_reply":
                bid = inter["button_reply"]["id"]
                lang = LAST_LANG.get(from_num, "es")
                if bid == "planes":
                    wa_send_text(from_num, same_lang_reply(lang, {
                        "es": f"Aquí puedes ver los planes y precios:\n{PLANS_URL}",
                        "en": f"See plans & pricing here:\n{PLANS_URL}",
                        "pt": f"Aqui estão os planos e preços:\n{PLANS_URL}",
                        "fr": f"Consultez les tarifs ici :\n{PLANS_URL}",
                        "ru": f"Тарифы здесь:\n{PLANS_URL}",
                        "ar": f"الخطط والأسعار هنا:\n{PLANS_URL}",
                        "hi": f"प्लान और कीमतें यहाँ देखें:\n{PLANS_URL}",
                        "zh": f"方案与价格：\n{PLANS_URL}",
                        "ja": f"料金プランはこちら：\n{PLANS_URL}",
                    }))
                    continue
                if bid == "web":
                    wa_send_text(from_num, PLANS_URL)
                    continue
                if bid == "humano":
                    USER_STATE[from_num] = "waiting_handoff"
                    wa_send_text(from_num, same_lang_reply(lang, {
                        "es":"👤 Te conecto con un asesor. Comparte: nombre, tema y horario.",
                        "en":"👤 I’ll connect you with a human. Please send: name, topic, time.",
                        "pt":"👤 Vou te conectar a um assessor. Envie: nome, tema, horário.",
                    }))
                    continue

        # 2) Mensaje de texto normal
        if msg_type == "text":
            user_text = m.get("text",{}).get("body","")
            # interceptores
            handled = handle_text(from_num, user_text)
            if handled:
                continue
            # si no fue interceptado → LLM
            lang = detect_lang(user_text)
            answer = llm_answer(user_text, lang)
            wa_send_text(from_num, answer)
            continue

        # 3) Otros tipos (audio, imagen, etc.) → simple aviso
        lang = LAST_LANG.get(from_num, "es")
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"Recibí tu mensaje. Por ahora entiendo mejor texto 😉",
            "en":"I received your message. For now I understand text best 😉",
            "pt":"Recebi sua mensagem. No momento entendo melhor texto 😉",
        }))

    return {"status":"ok"}
