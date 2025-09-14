# server/main.py — NochGPT v3.1 (menú automático + precios + humano)
# ---------------------------------------------------------------------------------
# ✅ Webhook Meta (GET verify + POST)
# ✅ Menú con botones: "Ver planes", "Sitio web", "Hablar con humano"
# ✅ Muestra botones: (1) primer mensaje de un número, (2) saludos/menú, (3) cuando piden precios/planes
# ✅ Handoff humano: guarda ticket en /tmp/handoff.json y confirma
# ✅ De-dup por message_id, healthcheck, /tickets, /handoff, /panel
# ✅ LLM fallback (OpenAI) responde en el idioma detectado
# ---------------------------------------------------------------------------------

from __future__ import annotations
import os, json, time, re, typing, requests
from collections import deque
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse

# ---------- Config ----------
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
WA_TOKEN          = os.getenv("WHATSAPP_TOKEN", "")
WA_PHONE_ID       = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
PLANS_URL         = os.getenv("PLANS_URL", "https://www.dentodo.com/plans-pricing")

HANDOFF_FILE      = "/tmp/handoff.json"
SEEN_MSGS         = deque(maxlen=500)   # de-dup message_id
USER_STATE: dict[str,str] = {}          # waiting_handoff/done
LAST_LANG: dict[str,str] = {}           # idioma por número
FIRST_SEEN: set[str] = set()            # números que ya recibieron menú

# ---------- App ----------
app = FastAPI(title="Dental-LLM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------- Utilidades ----------
def now_ts() -> int: return int(time.time())

def read_handoff() -> list[dict]:
    if not os.path.exists(HANDOFF_FILE): return []
    try:
        with open(HANDOFF_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

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
    if re.search(r"[\u0600-\u06FF]", s): return "ar"
    if re.search(r"[\u0900-\u097F]", s): return "hi"
    if re.search(r"[\u0400-\u04FF]", s): return "ru"
    if re.search(r"[\u4E00-\u9FFF]", s): return "zh"
    if re.search(r"[\u3040-\u30FF]", s): return "ja"
    sl = s.lower()
    if re.search(r"[áéíóúñ¿¡]", sl): return "es"
    if re.search(r"[ãõáéíóúç]", sl): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", sl): return "fr"
    return "en"

def same_lang_reply(lang: str, text_map: dict) -> str:
    return text_map.get(lang, text_map.get("en"))

# ---------- WhatsApp helpers ----------
def wa_api_url() -> str:
    return f"https://graph.facebook.com/v19.0/{WA_PHONE_ID}/messages" if WA_PHONE_ID else ""

def wa_send(payload: dict):
    if not (WA_TOKEN and WA_PHONE_ID): 
        print("⚠️ Falta WHATSAPP_TOKEN o WHATSAPP_PHONE_ID")
        return
    try:
        r = requests.post(
            wa_api_url(),
            headers={"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"},
            json=payload, timeout=20
        )
        print("WA SEND:", r.status_code, r.text[:300])
    except Exception as e:
        print("WA error:", e)

def wa_send_text(to: str, body: str):
    wa_send({"messaging_product":"whatsapp","to":to,"type":"text","text":{"body":body}})

def wa_send_buttons_menu(to: str, lang: str):
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
    wa_send({
        "messaging_product":"whatsapp","to":to,"type":"interactive",
        "interactive":{
            "type":"button","body":{"text":t["body"]},
            "action":{"buttons":[
                {"type":"reply","reply":{"id":"planes","title":t["b1"]}},
                {"type":"reply","reply":{"id":"web","title":t["b2"]}},
                {"type":"reply","reply":{"id":"humano","title":t["b3"]}},
            ]}
        }
    })

# ---------- LLM ----------
def llm_answer(user_text: str, lang: str) -> str:
    if not OPENAI_API_KEY:
        return same_lang_reply(lang, {
            "es":"Falta OPENAI_API_KEY en el servidor.",
            "en":"Missing OPENAI_API_KEY on server.",
            "pt":"Falta OPENAI_API_KEY no servidor."
        })
    import openai
    openai.api_key = OPENAI_API_KEY
    system = ("You are NochGPT, a helpful dental laboratory assistant. "
              "Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM). "
              "Always reply in the user's language.")
    try:
        resp = openai.chat.completions.create(
            model=OPENAI_MODEL, temperature=0.2,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user_text}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("LLM error:", e)
        return same_lang_reply(lang, {
            "es":"No pude generar respuesta ahora.",
            "en":"I couldn’t generate a reply right now.",
            "pt":"Não consegui responder agora."
        })

# ---------- Reglas de activación ----------
PRICE_WORDS = r"(precio|precios|plan|planes|cotizar|tienda|price|prices|plan|plans)"
MENU_WORDS  = r"(hola|buenas|men[uú]|opciones|menu|start|inicio|help|ayuda|hi|hello)"
HUMAN_WORDS = r"(humano|asesor|soporte|operator|agent|agente|hablar con humano|speak to human|help me)"

def handle_text(from_num: str, text: str) -> bool:
    """True = ya respondí; False = que siga al LLM."""
    lang = detect_lang(text)
    LAST_LANG[from_num] = lang
    t = text.strip().lower()

    # 0) Primer mensaje de este número → mostrar menú
    if from_num not in FIRST_SEEN:
        FIRST_SEEN.add(from_num)
        wa_send_buttons_menu(from_num, lang)
        # Mensaje de bienvenida corto
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"¡Hola! ¿En qué puedo ayudarte hoy en el ámbito dental?",
            "en":"Hi! How can I help you today with dental topics?",
            "pt":"Olá! Como posso ajudar você hoje em temas dentários?",
            "fr":"Salut ! Comment puis-je t’aider sur les sujets dentaires ?",
            "ru":"Привет! Чем помочь по стоматологии?",
            "ar":"مرحبًا! كيف أساعدك اليوم بموضوعات طب الأسنان؟",
            "hi":"नमस्ते! दंत विषयों में आज मैं आपकी कैसे मदद करूँ?",
            "zh":"你好！今天我可以如何在牙科方面帮助你？",
            "ja":"こんにちは！歯科に関して今日は何をお手伝いできますか？",
        }))
        return True

    # 1) Menú por saludo/“menú”
    if re.search(MENU_WORDS, t):
        wa_send_buttons_menu(from_num, lang)
        return True

    # 2) Menú por intención de precios/planes
    if re.search(PRICE_WORDS, t):
        wa_send_buttons_menu(from_num, lang)
        return True

    # 3) Handoff humano por palabra clave/sinónimo
    if re.search(HUMAN_WORDS, t):
        USER_STATE[from_num] = "waiting_handoff"
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"👤 Te conecto con un asesor. Escribe: nombre, tema y horario (y teléfono si es otro).",
            "en":"👤 I’ll connect you with a human. Send: name, topic and preferred time (phone if different).",
            "pt":"👤 Vou te conectar a um assessor. Envie: nome, tema e horário (telefone se for outro)."
        }))
        return True

    # 4) Si está completando ticket
    if USER_STATE.get(from_num) == "waiting_handoff":
        append_handoff({
            "ts": now_ts(), "label":"NochGPT", "from": from_num,
            "nombre":"", "tema": text[:120], "contacto": from_num,
            "horario":"", "mensaje": text
        })
        USER_STATE[from_num] = "done"
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"✅ Gracias. Tu solicitud fue registrada; un asesor te contactará en breve.",
            "en":"✅ Thanks. Your request was recorded; an agent will contact you soon.",
            "pt":"✅ Obrigado. Seu pedido foi registrado; entraremos em contato."
        }))
        return True

    return False

# ---------- Rutas ----------
@app.get("/", response_class=HTMLResponse)
def root():
    return "<b>Dental-LLM corriendo ✅</b>"

@app.get("/_debug/health")
def health():
    return {"ok": True, "cfg":{
        "openai": bool(OPENAI_API_KEY),
        "wa_token": bool(WA_TOKEN),
        "wa_phone_id": bool(WA_PHONE_ID),
        "model": OPENAI_MODEL,
        "sheet_webhook": False,
        "switch_number": False
    }}

@app.get("/handoff")
def get_handoff(): return read_handoff()

@app.get("/tickets")
def get_tickets(): return read_handoff()

@app.get("/panel", response_class=HTMLResponse)
def panel():
    rows = read_handoff()[::-1]
    html = [
        "<html><head><meta charset='utf-8'><title>Tickets – NochGPT</title>",
        "<style>body{font-family:system-ui;background:#0b1b3f;color:#fff}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #234}th{background:#123}tr:hover{background:#112}</style>",
        "</head><body>",
        f"<h2>Tickets – NochGPT <span style='background:#1b5fff;padding:6px 10px;border-radius:10px'>{len(rows)} activos</span></h2>",
        "<small>Se actualiza al refrescar · Origen: /tmp/handoff.json</small>",
        "<table><thead><tr><th>Fecha/Hora</th><th>Número</th><th>Nombre</th><th>Tema</th><th>Contacto</th></tr></thead><tbody>"
    ]
    for r in rows:
        dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("ts", now_ts())))
        html.append(f"<tr><td>{dt}</td><td>{r.get('from','')}</td><td>{r.get('nombre','')}</td>"
                    f"<td>{r.get('tema','')}</td><td>{r.get('contacto','')}</td></tr>")
    html.append("</tbody></table></body></html>")
    return "\n".join(html)

# ---------- Webhook Meta ----------
@app.get("/webhook")
def webhook_verify(request: Request):
    qp = request.query_params
    if qp.get("hub.mode") == "subscribe" and qp.get("hub.verify_token") == META_VERIFY_TOKEN:
        return PlainTextResponse(qp.get("hub.challenge",""))
    raise HTTPException(status_code=403, detail="Forbidden")

@app.post("/webhook")
async def webhook_post(req: Request):
    data = await req.json()
    print("Payload:", str(data)[:800])

    entry = (data.get("entry") or [{}])[0]
    changes = (entry.get("changes") or [{}])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [])
    if not messages:
        return {"status":"ok"}

    for m in messages:
        msg_id = m.get("id")
        if msg_id in SEEN_MSGS:
            print("👀 Duplicado:", msg_id)
            continue
        SEEN_MSGS.append(msg_id)

        from_num = m.get("from")
        mtype = m.get("type")

        # Botón pulsado
        if mtype == "interactive":
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
                    wa_send_text(from_num, PLANS_URL); continue
                if bid == "humano":
                    USER_STATE[from_num] = "waiting_handoff"
                    wa_send_text(from_num, same_lang_reply(lang, {
                        "es":"👤 Te conecto con un asesor. Escribe: nombre, tema y horario.",
                        "en":"👤 I’ll connect you with a human. Send: name, topic, time.",
                        "pt":"👤 Vou te conectar a um assessor. Envie: nome, tema e horário.",
                    }))
                    continue

        # Texto normal
        if mtype == "text":
            body = m.get("text",{}).get("body","")
            if handle_text(from_num, body):
                continue
            # Si no gatilló nada, respondemos con LLM
            lang = detect_lang(body)
            wa_send_text(from_num, llm_answer(body, lang))
            return {"status":"ok"}

        # Otros tipos (audio, imagen…)
        lang = LAST_LANG.get(from_num, "es")
        wa_send_text(from_num, same_lang_reply(lang, {
            "es":"Recibí tu mensaje. Por ahora entiendo mejor el texto 😉",
            "en":"I received your message. For now I understand text best 😉",
            "pt":"Recebi sua mensagem. No momento entendo melhor texto 😉",
        }))

    return {"status":"ok"}
