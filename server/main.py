# server/main.py — NochGPT WhatsApp v3.4 (idioma/audio/foto/pdf/sheets OK)
# -------------------------------------------------------------------------
# ✅ Botones localizados según el IDIOMA del MENSAJE del usuario
# ✅ Recuerda el idioma por usuario (last_lang) para todas las respuestas
# ✅ Audio (voice/ogg/aac/mp4): descarga y transcribe con gpt-4o-mini-transcribe
# ✅ Imágenes: descarga, convierte a data:URL y analiza con gpt-4o-mini (visión)
# ✅ PDF y docs: descarga; si es PDF intenta extraer texto simple y resumir;
#    si no logra extraer, avisa y pide detalles (no se cae)
# ✅ Handoff “Hablar con humano”: guarda en /tmp/handoff.json y POST a Google Sheets
# ✅ Webhook robusto: responde 200 rápido y procesa en background
# -------------------------------------------------------------------------

from __future__ import annotations
import os, re, json, time, base64, tempfile, typing
import requests

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from openai import OpenAI

# ====== ENV ======
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
OPENAI_VISION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_STT_MODEL    = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

WA_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
WA_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")

G_SHEET_WEBHOOK = os.getenv("G_SHEET_WEBHOOK", "")   # URL del Apps Script (opcional)

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Dental-LLM API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ====== ESTADO ======
processed_ids: set[str] = set()
user_state: dict[str, str] = {}         # ej. { "+52162...": "waiting_handoff" }
last_lang:  dict[str, str] = {}         # idioma recordado por usuario
last_sheet_status: dict[str, typing.Any] = {}

# ====== UTILS ======
EMOJI_RE = re.compile(r"[𐀀-􏿿]", flags=re.UNICODE)

def sanitize_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\r", " ").strip()
    s = EMOJI_RE.sub("", s)
    return s[:4000]

def detect_lang_from_text(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõáéíóúç]", t): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    if re.search(r"[\u0900-\u097F]", t): return "hi"   # hindi
    if re.search(r"[\u0400-\u04FF]", t): return "ru"   # ruso
    if re.search(r"[\u0600-\u06FF]", t): return "ar"   # árabe
    if re.search(r"[\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9F]", t): return "ja" # japonés
    if re.search(r"[\u4E00-\u9FFF]", t): return "zh"   # chino
    return "en"

L = {
    "es": {
        "hi": "¡Hola! ¿En qué puedo ayudarte hoy en el ámbito dental?",
        "menu": "Selecciona una opción:",
        "btn_prices": "Planes y precios",
        "btn_times": "Tiempos de entrega",
        "btn_human": "Hablar con humano",
        "handoff_ask": ("👤 Te conecto con un asesor. Comparte:\n"
                        "• Nombre\n• Tema (implante, zirconia, urgencia)\n"
                        "• Horario preferido y teléfono si es otro"),
        "handoff_ok": "✅ Gracias. Tu solicitud fue registrada y la atiende un asesor.",
        "audio_rcv": "🎙 Recibí tu audio, lo estoy transcribiendo…",
        "audio_fail": "No pude transcribir el audio. ¿Puedes escribir un breve resumen?",
        "img_rcv": "🖼️ Recibí tu imagen, la estoy analizando…",
        "img_fail": "No pude analizar la imagen. ¿Puedes describir lo que necesitas?",
        "doc_rcv": "📄 Recibí tu archivo, lo reviso…",
        "pdf_fail": "No pude leer el PDF. Envíame el punto clave o una foto de la hoja importante."
    },
    "en": {
        "hi": "Hi! How can I help you today with dental topics?",
        "menu": "Choose an option:",
        "btn_prices": "Plans & pricing",
        "btn_times": "Turnaround times",
        "btn_human": "Talk to a human",
        "handoff_ask": "👤 I’ll connect you with an agent. Share name, topic and preferred time.",
        "handoff_ok": "✅ Thanks. Your request was recorded; an agent will contact you soon.",
        "audio_rcv": "🎙 I received your audio, transcribing…",
        "audio_fail": "I couldn’t transcribe the audio. Could you type a short summary?",
        "img_rcv": "🖼️ I received your image, analyzing…",
        "img_fail": "I couldn’t analyze the image. Please describe your need.",
        "doc_rcv": "📄 I received your file, reviewing…",
        "pdf_fail": "I couldn’t read that PDF. Please share the key point or a photo of the page."
    },
}
def T(lang: str, key: str) -> str:
    if lang not in L: lang = "es"
    return L[lang].get(key, L["es"].get(key, ""))

SYSTEM_PROMPT = (
    "You are NochGPT, a helpful dental laboratory assistant.\n"
    "- Focus on prosthetics, implants, zirconia, CAD/CAM, workflows, materials.\n"
    "- Be concise and practical. Always reply in the user's language."
)

def wa_url(path: str) -> str:
    return f"https://graph.facebook.com/v19.0/{path}"

def wa_send(payload: dict):
    try:
        r = requests.post(
            wa_url(f"{WA_PHONE_ID}/messages"),
            headers={"Authorization": f"Bearer {WA_TOKEN}", "Content-Type": "application/json"},
            json=payload, timeout=20
        )
        if r.status_code >= 400:
            print("WA send error:", r.status_code, r.text)
    except Exception as e:
        print("WA send ex:", e)

def wa_send_text(to: str, text: str):
    wa_send({"messaging_product":"whatsapp","to":to,"type":"text","text":{"body": text[:4096]}})

def wa_send_buttons(to: str, body: str, lang: str):
    wa_send({
        "messaging_product":"whatsapp",
        "to": to,
        "type":"interactive",
        "interactive":{
            "type":"button",
            "body":{"text": body[:1024]},
            "action":{"buttons":[
                {"type":"reply","reply":{"id":"btn_prices","title":T(lang,'btn_prices')}},
                {"type":"reply","reply":{"id":"btn_times","title":T(lang,'btn_times')}},
                {"type":"reply","reply":{"id":"btn_human","title":T(lang,'btn_human')}},
            ]}
        }
    })

def wa_get_media_download_url(media_id: str) -> str | None:
    r = requests.get(wa_url(media_id), headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=20)
    if r.status_code != 200: 
        print("media meta error:", r.text); return None
    return r.json().get("url")

def wa_download_to_tmp(media_id: str) -> tuple[str|None, str|None]:
    url = wa_get_media_download_url(media_id)
    if not url: return None, None
    r = requests.get(url, headers={"Authorization": f"Bearer {WA_TOKEN}"}, timeout=30)
    if r.status_code != 200:
        print("media dl error:", r.text); return None, None
    ct = r.headers.get("Content-Type","application/octet-stream")
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(r.content)
    return path, ct

# ====== TICKETS ======
HANDOFF_FILE = "/tmp/handoff.json"
def save_ticket(rec: dict):
    try:
        data = []
        if os.path.exists(HANDOFF_FILE):
            with open(HANDOFF_FILE,"r",encoding="utf-8") as f: data = json.load(f)
        data.append(rec)
        with open(HANDOFF_FILE,"w",encoding="utf-8") as f: json.dump(data,f,indent=2,ensure_ascii=False)
    except Exception as e:
        print("save_ticket:", e)

def push_sheet(rec: dict):
    if not G_SHEET_WEBHOOK: 
        last_sheet_status["ok"]=False; last_sheet_status["why"]="G_SHEET_WEBHOOK missing"; return
    try:
        r = requests.post(G_SHEET_WEBHOOK, json=rec, timeout=15)
        last_sheet_status["ok"] = r.status_code==200
        last_sheet_status["code"]=r.status_code
        last_sheet_status["text"]=r.text[:300]
    except Exception as e:
        last_sheet_status["ok"]=False; last_sheet_status["err"]=str(e)

@app.get("/handoff")
def list_handoff():
    if not os.path.exists(HANDOFF_FILE): return []
    with open(HANDOFF_FILE,"r",encoding="utf-8") as f: return json.load(f)

@app.get("/_debug/health")
def health():
    return {"ok": True, "cfg":{
        "openai": bool(OPENAI_API_KEY),
        "wa_token": bool(WA_TOKEN),
        "wa_phone_id": bool(WA_PHONE_ID),
        "model": OPENAI_VISION_MODEL,
        "stt": OPENAI_STT_MODEL,
        "sheet_webhook": bool(G_SHEET_WEBHOOK),
        "last_sheet": last_sheet_status or {}
    }}

# ====== LLM ======
def llm_reply(user_text: str, lang: str) -> str:
    try:
        rsp = client.chat.completions.create(
            model=OPENAI_VISION_MODEL, temperature=0.2,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_text}
            ]
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("llm_reply:", e)
        return T(lang,"audio_fail") if user_text.startswith("[AUDIO]") else "Un momento, por favor. Hubo un problema temporal."

def llm_vision_analyze(prompt: str, data_url: str, lang: str) -> str:
    try:
        rsp = client.chat.completions.create(
            model=OPENAI_VISION_MODEL, temperature=0.2,
            messages=[{
                "role":"system","content":SYSTEM_PROMPT
            },{
                "role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":data_url}}
                ]
            }]
        )
        return rsp.choices[0].message.content.strip()
    except Exception as e:
        print("vision:", e)
        return T(lang,"img_fail")

def stt_transcribe(file_path: str) -> str|None:
    try:
        with open(file_path,"rb") as f:
            tr = client.audio.transcriptions.create(model=OPENAI_STT_MODEL, file=f)
        return getattr(tr,"text",None)
    except Exception as e:
        print("stt:", e); return None

# ====== MENÚ / HANDOFF ======
def send_menu(to: str, lang: str):
    body = f"{T(lang,'hi')}\n\n{T(lang,'menu')}"
    wa_send_buttons(to, body, lang)

def start_handoff(to: str, lang: str):
    user_state[to] = "waiting_handoff"
    wa_send_text(to, T(lang,"handoff_ask"))

def complete_handoff(to: str, lang: str, text: str):
    rec = {
        "ts": int(time.time()),
        "label": "NochGPT",
        "from": to,
        "nombre": "",
        "tema": text,
        "contacto": to,
        "horario": "",
        "mensaje": text,
    }
    save_ticket(rec)
    push_sheet(rec)
    user_state[to] = "done"
    wa_send_text(to, T(lang,"handoff_ok"))

# ====== BACKGROUND ======
def process_value(value: dict):
    try:
        messages = value.get("messages", [])
        if not messages: return
        msg = messages[0]
        msg_id = msg.get("id")
        if msg_id in processed_ids: return
        processed_ids.add(msg_id)

        from_number = msg.get("from")
        mtype = msg.get("type")

        # Determinar idioma desde el CONTENIDO (no del JSON completo)
        lang = last_lang.get(from_number, "es")
        if mtype == "text":
            lang = detect_lang_from_text(msg.get("text",{}).get("body",""))
        elif mtype == "interactive":
            # conserva el último
            pass
        elif mtype == "audio":
            # sin texto aún, usa último o español
            pass
        elif mtype == "image":
            # usa caption si existe
            cap = msg.get("image",{}).get("caption","")
            if cap: lang = detect_lang_from_text(cap)
        elif mtype == "document":
            cap = msg.get("document",{}).get("caption","") or msg.get("document",{}).get("filename","")
            if cap: lang = detect_lang_from_text(cap)
        last_lang[from_number] = lang

        # === BOTONES (interactive) ===
        if mtype == "interactive":
            data = msg.get("interactive",{})
            if data.get("type") == "button_reply":
                btn = data.get("button_reply",{})
                bid = btn.get("id","")
                if bid == "btn_human": start_handoff(from_number, lang); return
                if bid == "btn_prices":
                    wa_send_text(from_number,
                                 "💳 " + ( "Planes y precios: https://www.dentodo.com/plans-pricing" if lang=="es"
                                           else "Plans & pricing: https://www.dentodo.com/plans-pricing"))
                    return
                if bid == "btn_times":
                    wa_send_text(from_number,
                                 "⏱️ " + ( "Tiempos típicos: Zirconia 24–48h; Implante 3–5 días."
                                           if lang=="es" else
                                           "Typical turnaround: Zirconia 24–48h; Implant 3–5 days." ))
                    return

        # === TEXTO ===
        if mtype == "text":
            text = sanitize_text(msg.get("text",{}).get("body",""))
            # saludo → menú localizado
            if re.search(r"\b(hola|hello|hi|oi|salut|مرحبا|नमस्ते|привет|こんにちは|你好)\b", text.lower()):
                send_menu(from_number, detect_lang_from_text(text)); return
            # handoff por texto libre
            if re.search(r"hablar\s+con\s+humano|asesor|human|agent|operator", text.lower()):
                start_handoff(from_number, lang); return
            # completar handoff si está esperando
            if user_state.get(from_number) == "waiting_handoff":
                complete_handoff(from_number, lang, text); return
            # respuesta LLM
            wa_send_text(from_number, llm_reply(text, lang)); return

        # === AUDIO ===
        if mtype == "audio":
            wa_send_text(from_number, T(lang,"audio_rcv"))
            media_id = msg.get("audio",{}).get("id")
            fpath, ctype = wa_download_to_tmp(media_id) if media_id else (None, None)
            if fpath:
                text = stt_transcribe(fpath)
                try: os.remove(fpath)
                except: pass
                if text and text.strip():
                    lang2 = detect_lang_from_text(text); last_lang[from_number]=lang2
                    wa_send_text(from_number, llm_reply(text, lang2))
                else:
                    wa_send_text(from_number, T(lang,"audio_fail"))
            else:
                wa_send_text(from_number, T(lang,"audio_fail"))
            return

        # === IMAGEN ===
        if mtype == "image":
            wa_send_text(from_number, T(lang,"img_rcv"))
            media_id = msg.get("image",{}).get("id")
            fpath, ctype = wa_download_to_tmp(media_id) if media_id else (None, None)
            if fpath:
                try:
                    with open(fpath,"rb") as f: b64 = base64.b64encode(f.read()).decode()
                    mime = ctype or "image/jpeg"
                    data_url = f"data:{mime};base64,{b64}"
                    prompt = "Describe brevemente la situación clínica y sugiere 1–3 pasos prácticos." if lang=="es" else \
                             "Briefly describe the clinical situation and suggest 1–3 practical steps."
                    out = llm_vision_analyze(prompt, data_url, lang)
                    wa_send_text(from_number, out)
                except Exception as e:
                    print("img analyze:", e); wa_send_text(from_number, T(lang,"img_fail"))
                try: os.remove(fpath)
                except: pass
            else:
                wa_send_text(from_number, T(lang,"img_fail"))
            return

        # === DOCUMENTO (PDF) ===
        if mtype == "document":
            wa_send_text(from_number, T(lang,"doc_rcv"))
            doc = msg.get("document",{})
            media_id = doc.get("id")
            fname = doc.get("filename","file")
            fpath, ctype = wa_download_to_tmp(media_id) if media_id else (None, None)
            if fpath and (ctype or "").startswith("application/pdf") or fname.lower().endswith(".pdf"):
                # Intento simple: leer bytes y pedir a LLM que dé puntos clave (no OCR perfecto)
                try:
                    with open(fpath,"rb") as f:
                        raw = f.read(120_000) # primeros ~120KB para no pasar límite
                    sample_b64 = base64.b64encode(raw).decode()[:150000]
                    prompt = (
                        f"Archivo PDF '{fname}'. A partir de este fragmento base64 de su contenido, "
                        f"extrae en {lang} 5 puntos clave clínicos o técnicos. "
                        "Si la señal es insuficiente, di que necesitas una foto o el texto."
                    )
                    text_hint = f"[PDF_BASE64_FRAGMENT]{sample_b64}"
                    wa_send_text(from_number, llm_reply(prompt + "\n\n" + text_hint, lang))
                except Exception as e:
                    print("pdf summarize:", e); wa_send_text(from_number, T(lang,"pdf_fail"))
                try: os.remove(fpath)
                except: pass
                return
            else:
                # otro tipo de doc: acuse
                wa_send_text(from_number, T(lang,"doc_rcv"))
                if fpath:
                    try: os.remove(fpath)
                    except: pass
                return

        # Otros → menú para guiar
        send_menu(from_number, last_lang.get(from_number,"es"))

    except Exception as e:
        print("process_value ex:", e)

# ====== ROUTES ======
@app.get("/", response_class=HTMLResponse)
def root():
    return "Dental-LLM corriendo ✅"

@app.get("/webhook")
def verify(mode: str="", challenge: str="", verify_token: str=""):
    return PlainTextResponse(challenge or "")

@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks):
    try:
        body = await request.json()
    except Exception:
        return {"status":"ok"}
    entry = body.get("entry") or []
    if not entry: return {"status":"ok"}
    changes = entry[0].get("changes") or []
    if not changes: return {"status":"ok"}
    value = changes[0].get("value") or {}
    background.add_task(process_value, value)
    return {"status":"ok"}
