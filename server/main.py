# server/main.py — NochGPT v3 (botones + handoff + sheets + autodetección)
from __future__ import annotations
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from openai import OpenAI
import os, time, json, re, requests, mimetypes, pathlib
from typing import Any, Dict, Tuple, List

# =================== Config básica ===================
app = FastAPI(title="Dental-LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TEMP    = float(os.getenv("OPENAI_TEMP", "0.2"))

WA_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
WA_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

SHEET_WEBHOOK = os.getenv("SHEET_WEBHOOK", "")  # URL de tu Apps Script (opcional)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = (
    "You are NochGPT, a helpful dental laboratory assistant.\n"
    "- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).\n"
    "- Be concise, practical, and give ranges when relevant.\n"
    "- Always reply in the same language as the user's question.\n"
    "- If the topic is not dental, politely say you are focused on dental topics."
)

# =================== Utilidades idioma ===================
def detect_lang(text: str) -> str:
    if not text:
        return "en"
    t = text.strip().lower()

    if re.search(r"[\u0600-\u06FF]", t):   return "ar"   # árabe
    if re.search(r"[\u0900-\u097F]", t):   return "hi"   # hindi
    if re.search(r"[\u4E00-\u9FFF]", t):   return "zh"   # chino
    if re.search(r"[\u3040-\u30FF\u31F0-\u31FF]", t): return "ja"  # japonés
    if re.search(r"[\u0400-\u04FF]", t):   return "ru"   # cirílico
    if re.search(r"[\uAC00-\uD7AF]", t):   return "ko"   # coreano
    if re.search(r"[áéíóúñ¿¡]", t):        return "es"   # español
    if re.search(r"[ãõáéíóúç]", t):        return "pt"   # portugués
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"   # francés

    if re.search(r"\b(hola|buenos dias|buenas tardes|precio|presupuesto|implante|carilla|zirconia|corona|puente)\b", t):
        return "es"
    if re.search(r"\b(ol[áa]|bom dia|boa tarde|orçamento|implante|faceta|zirc[oô]nia)\b", t):
        return "pt"
    if re.search(r"\b(bonjour|bonsoir|salut|implant|facette|zircone)\b", t):
        return "fr"
    if re.search(r"\b(hello|hi|good morning|good afternoon|implant|veneer|zirconia)\b", t):
        return "en"
    return "en"

def is_greeting(t: str) -> bool:
    t = (t or "").strip().lower()
    return bool(re.search(
        r"\b(hola|buenos dias|buenas tardes|buenas noches|que tal|saludos|hi|hello|hey|bonjour|salut|ol[áa]|namaste|privet|السلام|مرحبا)\b",
        t
    ))

# =================== WhatsApp helpers ===================
def _wa_graph(path: str, payload: dict):
    if not WA_TOKEN or not WA_PHONE_ID:
        return {"ok": False, "error": "WA env missing"}
    url = f"https://graph.facebook.com/v20.0{path}"
    hdr = {"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=hdr, json=payload, timeout=15)
        ok = 200 <= r.status_code < 300
        return {"ok": ok, "status": r.status_code, "resp": r.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def wa_send_text(to_number: str, text: str):
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text},
    }
    return _wa_graph(f"/{WA_PHONE_ID}/messages", payload)

def wa_send_template(to_number: str, template_name: str, lang_code: str = "es_MX"):
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "template",
        "template": {"name": template_name, "language": {"code": lang_code}},
    }
    return _wa_graph(f"/{WA_PHONE_ID}/messages", payload)

def wa_send_buttons(to_number: str, body_text: str, btn1: str, btn2: str, btn3: str):
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {"text": body_text},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": "BTN_PRECIOS", "title": btn1}},
                    {"type": "reply", "reply": {"id": "BTN_PLANES",  "title": btn2}},
                    {"type": "reply", "reply": {"id": "BTN_HUMANO",  "title": btn3}},
                ]
            },
        },
    }
    return _wa_graph(f"/{WA_PHONE_ID}/messages", payload)

def send_welcome_menu(to_number: str, lang: str = "es"):
    lang_map = {
        "es": "es_MX", "en": "en_US", "pt": "pt_BR", "fr": "fr_FR",
        "ru": "ru", "ar": "ar", "hi": "hi", "zh": "zh_CN", "ja": "ja", "ko": "ko"
    }
    # 1) intentar plantilla llamada "nochgpt"
    try:
        r = wa_send_template(to_number, "nochgpt", lang_map.get(lang, "es_MX"))
        if isinstance(r, dict) and r.get("ok"):
            return
    except Exception:
        pass
    # 2) fallback botones localizados
    textos = {
        "es": ("Hola 👋, ¿qué necesitas?", "Precios", "Planes", "Hablar con humano"),
        "en": ("Hi 👋, how can I help?", "Prices", "Plans", "Talk to a human"),
        "pt": ("Oi 👋, como posso ajudar?", "Preços", "Planos", "Falar com humano"),
        "fr": ("Salut 👋, comment puis-je aider ?", "Tarifs", "Offres", "Parler à un humain"),
        "ru": ("Привет 👋 Чем помочь?", "Цены", "Тарифы", "Связаться с человеком"),
        "ar": ("مرحباً 👋 كيف أساعدك؟", "الأسعار", "الباقات", "التحدث مع شخص"),
        "hi": ("नमस्ते 👋 कैसे मदद करूँ?", "कीमतें", "योजनाएँ", "इंसान से बात"),
        "zh": ("你好 👋 需要什么帮助？", "价格", "方案", "人工客服"),
        "ja": ("こんにちは 👋 何をお手伝いできますか？", "料金", "プラン", "担当者に相談"),
        "ko": ("안녕하세요 👋 무엇을 도와드릴까요?", "가격", "플랜", "상담원 연결"),
    }
    body, b1, b2, b3 = textos.get(lang, textos["es"])
    wa_send_buttons(to_number, body, b1, b2, b3)

# =================== LLM ===================
def call_openai(user_text: str, lang_hint: str = "en") -> str:
    if not client:
        return "Configura OPENAI_API_KEY."
    try:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text}
        ]
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=OPENAI_TEMP,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error de modelo: {e}"

# =================== Handoff a humano + Sheets ===================
HANDOFF_FILE = "/tmp/handoff.json"

def _load_handoff() -> List[Dict[str, Any]]:
    if not os.path.exists(HANDOFF_FILE):
        return []
    try:
        with open(HANDOFF_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_handoff(data: List[Dict[str, Any]]):
    try:
        with open(HANDOFF_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("⚠️ Error guardando handoff:", e)

def add_ticket(label: str, from_num: str, nombre: str, tema: str, contacto: str, horario: str, mensaje: str):
    row = {
        "ts": int(time.time()),
        "label": label,
        "from": from_num,
        "nombre": nombre or "",
        "tema": tema or "",
        "contacto": contacto or from_num,
        "horario": horario or "",
        "mensaje": mensaje or "",
    }
    data = _load_handoff()
    data.append(row)
    _save_handoff(data)

    # enviar a Google Sheets si está configurado
    if SHEET_WEBHOOK:
        try:
            requests.post(SHEET_WEBHOOK, json=row, timeout=10)
        except Exception as e:
            print("⚠️ No se pudo enviar a Sheets:", e)

# =================== Extractor de mensaje ===================
def _extract_user_text_and_kind(msg: dict) -> Tuple[str, str]:
    mtype = msg.get("type")
    if mtype == "text":
        return ( (msg.get("text") or {}).get("body", "").strip(), "text" )
    if mtype == "button":
        return ( (msg.get("button") or {}).get("text", "").strip(), "button_reply" )
    if mtype == "interactive":
        inter = msg.get("interactive") or {}
        brep = (inter.get("button_reply") or {})
        return ((brep.get("title") or "").strip(), "interactive")
    return ("", mtype or "unknown")

# =================== Rutas útiles ===================
@app.get("/", response_class=PlainTextResponse)
def root():
    return "Dental-LLM corriendo ✅"

@app.get("/_debug/health")
def health():
    return {
        "ok": True,
        "cfg": {
            "openai": bool(OPENAI_API_KEY),
            "wa_token": bool(WA_TOKEN),
            "wa_phone_id": bool(WA_PHONE_ID),
            "model": OPENAI_MODEL,
            "sheet_webhook": bool(SHEET_WEBHOOK),
        }
    }

@app.get("/handoff")
def list_handoff():
    return _load_handoff()

@app.get("/tickets", response_class=JSONResponse)
def tickets():
    return _load_handoff()

@app.get("/panel", response_class=HTMLResponse)
def panel():
    rows = _load_handoff()
    rows_sorted = sorted(rows, key=lambda x: x["ts"], reverse=True)
    html_rows = []
    for r in rows_sorted:
        dt = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        html_rows.append(
            f"<tr><td>{dt}</td><td>{r['from']}</td><td>{r.get('nombre','')}</td>"
            f"<td>{r.get('tema','')}</td><td>{r.get('contacto','')}</td><td>{r.get('mensaje','')}</td></tr>"
        )
    html = f"""
    <html><head>
    <meta charset="utf-8"/>
    <title>Tickets – NochGPT</title>
    <style>
      body{{font-family:ui-sans-serif,system-ui;}}
      table{{border-collapse:collapse;width:100%}}
      th,td{{border:1px solid #ccc;padding:8px}}
      th{{background:#0b5fff;color:#fff;text-align:left}}
      .badge{{display:inline-block;background:#37C871;color:#083;color:#052; padding:4px 8px;border-radius:8px;margin-left:8px}}
    </style>
    </head><body>
      <h2>Tickets – NochGPT <span class="badge">{len(rows)} activos</span></h2>
      <p>Se actualiza al refrescar la página · Origen: /tmp/handoff.json</p>
      <table>
        <thead><tr><th>Fecha/Hora</th><th>Número</th><th>Nombre</th><th>Tema</th><th>Contacto</th><th>Mensaje</th></tr></thead>
        <tbody>
          {''.join(html_rows)}
        </tbody>
      </table>
    </body></html>
    """
    return HTMLResponse(html)

# =================== WEBHOOK WhatsApp ===================
@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Maneja mensajes entrantes de WhatsApp Cloud API.
    - Saludo => plantilla o botones de bienvenida
    - Botones: Precios / Planes / Hablar con humano
    - Texto normal => LLM
    """
    try:
        body = await request.json()
    except Exception:
        return {"ok": False, "error": "invalid json"}

    # Verificación (GET) la maneja Meta por otro endpoint; aquí solo POST inbound
    entry = (body.get("entry") or [{}])[0]
    changes = (entry.get("changes") or [{}])[0]
    value = changes.get("value") or {}
    messages = value.get("messages") or []
    if not messages:
        return {"ok": True, "info": "no messages"}

    msg = messages[0]
    from_number = msg.get("from")  # número del usuario
    user_text, kind = _extract_user_text_and_kind(msg)
    lang = detect_lang(user_text)

    # 1) saludo => menú
    if is_greeting(user_text):
        send_welcome_menu(from_number, lang)
        return {"ok": True, "action": "welcome"}

    # 2) botones
    if kind in ("button_reply", "interactive"):
        t = user_text.lower()

        # PRECIOS
        if t in ("precios", "prices", "preços", "tarifs", "цены", "الأسعار", "कीमतें", "价格", "料金", "가격"):
            wa_send_text(from_number, {
                "es": "Aquí tienes nuestros precios: https://www.dentodo.com/plans-pricing",
                "en": "Our prices: https://www.dentodo.com/plans-pricing",
                "pt": "Nossos preços: https://www.dentodo.com/plans-pricing",
                "fr": "Nos tarifs : https://www.dentodo.com/plans-pricing"
            }.get(lang, "Prices: https://www.dentodo.com/plans-pricing"))
            return {"ok": True, "action": "prices"}

        # PLANES
        if t in ("planes", "plans", "planos", "offres", "тарифы", "الباقات", "योजनाएँ", "方案", "プラン", "플랜"):
            wa_send_text(from_number, {
                "es": "Estos son nuestros planes: https://www.dentodo.com/plans-pricing",
                "en": "These are our plans: https://www.dentodo.com/plans-pricing",
                "pt": "Nossos planos: https://www.dentodo.com/plans-pricing",
                "fr": "Nos offres : https://www.dentodo.com/plans-pricing"
            }.get(lang, "Plans: https://www.dentodo.com/plans-pricing"))
            return {"ok": True, "action": "plans"}

        # HABLAR CON HUMANO
        if any(x in t for x in ["humano","human","человек","شخص","इंसान","人工","担当","상담"]):
            prompt = {
                "es": "👤 Te conecto con un asesor. Comparte por favor:\n• Nombre\n• Tema (implante, zirconia, urgencia)\n• Horario preferido y teléfono si es otro\nTe contactamos enseguida.",
                "en": "👤 I'll connect you with a specialist. Please share:\n• Name\n• Topic (implant, zirconia, urgency)\n• Preferred time and phone (if different)\nWe'll contact you shortly.",
            }.get(lang, "👤 Please share your name, topic, and preferred time. We'll contact you shortly.")
            wa_send_text(from_number, prompt)
            # Marca al usuario para el siguiente mensaje como parte del ticket
            PENDING_HUMAN[from_number] = True
            return {"ok": True, "action": "human_request"}

    # 3) si usuario está completando handoff
    if PENDING_HUMAN.get(from_number):
        add_ticket("NochGPT", from_number, "", user_text, from_number, "", user_text)
        wa_send_text(from_number, {
            "es": "✅ Tu solicitud fue registrada y la atiende un asesor. Normalmente respondemos el mismo día hábil.",
            "en": "✅ Your request was recorded. A specialist will contact you soon.",
            "pt": "✅ Seu pedido foi registrado. Um especialista entrará em contato em breve."
        }.get(lang, "✅ Request recorded. A specialist will contact you soon."))
        PENDING_HUMAN.pop(from_number, None)
        return {"ok": True, "action": "human_saved"}

    # 4) Normal → LLM dental
    answer = call_openai(user_text, lang_hint=lang)
    wa_send_text(from_number, answer)
    return {"ok": True, "action": "llm"}

# memoria simple para el estado de “hablar con humano”
PENDING_HUMAN: Dict[str, bool] = {}
