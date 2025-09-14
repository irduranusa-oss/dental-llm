# server/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import os, time, re, requests, mimetypes, base64, pathlib

# --- IMPORTA LA CAJITA (cache en memoria) ---
# Nota: requiere server/__init__.py para import absoluto
from server.cache import get_from_cache, save_to_cache

app = FastAPI(title="Dental-LLM API")

# ----------------------------
# CORS (en pruebas = "*")
# Luego fija tu dominio (p. ej. "https://www.dentodo.com")
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

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # chat + visión
OPENAI_TEMP = float(os.getenv("OPENAI_TEMP", "0.2"))

SYSTEM_PROMPT = """You are NochGPT, a helpful dental laboratory assistant.
- Focus on dental topics (prosthetics, implants, zirconia, CAD/CAM, workflows, materials, sintering, etc.).
- Be concise, practical, and provide ranges (e.g., temperatures or times) when relevant.
- If the question is not dental-related, politely say you are focused on dental topics and offer a helpful redirection.
- IMPORTANT: Always reply in the same language as the user's question.
"""

class ChatIn(BaseModel):
    pregunta: str

# Historial simple en memoria
HIST = []      # cada item: {"t": timestamp, "pregunta": ..., "respuesta": ...}
MAX_HIST = 200

def detect_lang(text: str) -> str:
    t = text.lower()
    if re.search(r"[áéíóúñ¿¡]", t): return "es"
    if re.search(r"[ãõáéíóúç]", t): return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t): return "fr"
    return "en"

def call_openai(question: str) -> str:
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=OPENAI_TEMP,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="Error con el modelo")

# ===== Helpers Visión / PDF =====
def _mime_from_path(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"

def _to_data_url(path: str) -> str:
    mime = _mime_from_path(path)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def analyze_image_with_openai(image_path: str, extra_prompt: str = "") -> str:
    """
    Envía la imagen como dataURL a OpenAI visión y devuelve un análisis breve,
    orientado a odontología si aplica.
    """
    data_url = _to_data_url(image_path)
    user_msg = [
        {"type": "text", "text": (
            "Analiza brevemente esta imagen desde el punto de vista dental. "
            "Si no es odontológica, describe en términos generales. "
            "Sé conciso y práctico."
            + (f"\nContexto del usuario: {extra_prompt}" if extra_prompt else "")
        )},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=OPENAI_TEMP,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        print("Vision error:", e)
        return "Recibí tu imagen, pero no pude analizarla en este momento."

def extract_pdf_text(pdf_path: str, max_chars: int = 20000) -> str:
    """
    Extrae texto de PDF con PyPDF2 si está disponible.
    Devuelve texto recortado a max_chars.
    """
    try:
        import PyPDF2  # requiere PyPDF2==3.0.1 en requirements.txt
    except Exception as e:
        print("PyPDF2 no disponible:", e)
        return ""

    try:
        out = []
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text() or ""
                out.append(t)
                if sum(len(s) for s in out) >= max_chars:
                    break
        text = "\n".join(out)
        return text[:max_chars]
    except Exception as e:
        print("Error extrayendo PDF:", e)
        return ""

def summarize_document_with_openai(raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    prompt = (
        "Resume el siguiente documento de forma clara y accionable para un técnico dental. "
        "Incluye puntos clave, medidas/valores si existen y recomendaciones:\n\n" + raw_text
    )
    try:
        return call_openai(prompt)
    except Exception as e:
        print("Summarize error:", e)
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

# Diagnóstico rápido de PyPDF2
@app.get("/check_pypdf2")
def check_pypdf2():
    try:
        import PyPDF2
        return {"ok": True, "version": getattr(PyPDF2, "__version__", "unknown")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/chat")
def chat(body: ChatIn):
    q = (body.pregunta or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Falta 'pregunta'")
    lang = detect_lang(q)
    cached = get_from_cache(q, lang)
    if cached is not None:
        return {"respuesta": cached, "cached": True}
    a = call_openai(q)
    save_to_cache(q, lang, a)
    HIST.append({"t": time.time(), "pregunta": q, "respuesta": a})
    if len(HIST) > MAX_HIST:
        del HIST[: len(HIST) - MAX_HIST]
    return {"respuesta": a, "cached": False}

@app.post("/chat_multi")
def chat_multi(body: ChatIn):
    return chat(body)

@app.get("/history")
def history(q: str = "", limit: int = 10):
    q = (q or "").strip().lower()
    out = []
    for item in reversed(HIST):
        if q and q not in item["pregunta"].lower():
            continue
        out.append(item)
        if len(out) >= max(1, min(limit, 50)):
            break
    return list(reversed(out))

# ======================================================================
#                         WHATSAPP WEBHOOK
# ======================================================================

WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "")
META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "nochgpt-verify-123")

FB_API = "https://graph.facebook.com/v20.0"

def _e164_no_plus(num: str) -> str:
    num = (num or "").strip().replace(" ", "").replace("-", "")
    return num[1:] if num.startswith("+") else num

def _wa_base_url() -> str:
    return f"{FB_API}/{WHATSAPP_PHONE_ID}/messages"

def wa_send_text(to_number: str, body: str):
    if not (WHATSAPP_TOKEN and WHATSAPP_PHONE_ID):
        print("⚠️ Falta WHATSAPP_TOKEN o WHATSAPP_PHONE_ID")
        return {"ok": False, "error": "missing_credentials"}

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    data = {
        "messaging_product": "whatsapp",
        "to": _e164_no_plus(to_number),
        "type": "text",
        "text": {"preview_url": False, "body": body[:3900]},
    }
    try:
        r = requests.post(_wa_base_url(), headers=headers, json=data, timeout=20)
        j = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        return {"ok": r.ok, "status": r.status_code, "resp": j}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def wa_get_media_url(media_id: str) -> str:
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(f"{FB_API}/{media_id}", headers=headers, timeout=15)
    r.raise_for_status()
    return (r.json() or {}).get("url", "")

def wa_download_media(signed_url: str, dest_prefix: str = "/tmp/wa_media/") -> tuple[str, str]:
    # Asegura carpeta
    pathlib.Path(dest_prefix).mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    r = requests.get(signed_url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    mime = r.headers.get("Content-Type", "application/octet-stream")
    ext = mimetypes.guess_extension(mime) or ""
    path = os.path.join(dest_prefix, f"{int(time.time())}{ext}")
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path, mime

# --- VERIFICACIÓN (GET) ---
@app.get("/webhook")
async def verify_webhook(request: Request):
    mode      = request.query_params.get("hub.mode", "")
    token     = request.query_params.get("hub.verify_token", "")
    challenge = request.query_params.get("hub.challenge", "")
    print("WEBHOOK VERIFY =>", {"mode": mode, "token": token, "challenge": challenge})
    if mode == "subscribe" and token == META_VERIFY_TOKEN and challenge:
        return PlainTextResponse(content=challenge, status_code=200)
    return PlainTextResponse(content="forbidden", status_code=403)

# --- RECEPCIÓN DE MENSAJES (POST) ---
@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"received": False, "error": "invalid_json"})

    print("📩 Payload recibido:", data)

    try:
        entry   = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        value   = changes.get("value") or {}

        # A) Mensajes nuevos
        msgs = value.get("messages") or []
        if msgs:
            msg         = msgs[0]
            from_number = msg.get("from")
            mtype       = msg.get("type")

            # 1) Texto / botón
            if mtype == "text":
                user_text = (msg.get("text") or {}).get("body", "").strip()
            elif mtype == "button":
                user_text = (msg.get("button") or {}).get("text", "").strip()
            else:
                user_text = ""

            if user_text:
                try:
                    answer = call_openai(user_text)
                except Exception:
                    answer = "Lo siento, tuve un problema procesando tu mensaje."
                if from_number:
                    wa_send_text(from_number, answer)
                return {"status": "ok"}

            # 2) Imagen
            if mtype == "image":
                img = msg.get("image") or {}
                media_id = img.get("id")
                caption  = (img.get("caption") or "").strip()
                if media_id and from_number:
                    try:
                        url = wa_get_media_url(media_id)
                        path, mime = wa_download_media(url)
                        print(f"🖼️ Imagen guardada en {path} ({mime})")
                        analysis = analyze_image_with_openai(path, caption)
                        wa_send_text(from_number, f"🖼️ Análisis breve:\n{analysis}")
                    except Exception as e:
                        print("Error imagen:", e)
                        wa_send_text(from_number, "No pude analizar la imagen. ¿Puedes intentar de nuevo?")
                return {"status": "ok"}

            # 3) Documento (PDF)
            if mtype == "document":
                doc = msg.get("document") or {}
                media_id = doc.get("id")
                filename = doc.get("filename") or "documento.pdf"
                if media_id and from_number:
                    try:
                        url = wa_get_media_url(media_id)
                        path, mime = wa_download_media(url)
                        print(f"📄 Documento guardado en {path} ({mime})")
                        if "pdf" in mime or filename.lower().endswith(".pdf"):
                            raw = extract_pdf_text(path, max_chars=20000)
                            if raw:
                                summary = summarize_document_with_openai(raw)
                                wa_send_text(from_number, f"📄 Resumen de *{filename}*:\n{summary}")
                            else:
                                wa_send_text(
                                    from_number,
                                    "Recibí tu PDF pero no pude leerlo aquí. "
                                    "Agrega *PyPDF2==3.0.1* a requirements.txt y vuelvo a intentarlo."
                                )
                        else:
                            wa_send_text(
                                from_number,
                                f"Recibí *{filename}*. Por ahora analizo PDFs; si puedes convertirlo a PDF, te lo resumo."
                            )
                    except Exception as e:
                        print("Error documento:", e)
                        wa_send_text(from_number, "No pude procesar el documento. ¿Puedes intentar de nuevo?")
                return {"status": "ok"}

            # 4) Otros tipos
            if from_number:
                wa_send_text(
                    from_number,
                    "Recibí tu mensaje. Por ahora manejo texto, imágenes y PDFs. "
                    "Si necesitas algo con audio/video, avísame."
                )
            return {"status": "ok"}

        # B) Status (entregado/leído, etc.) — responder 200 OK
        if value.get("statuses"):
            return {"status": "status_ok"}

        return {"status": "no_message"}

    except Exception as e:
        print("❌ Error en webhook:", e)
        return {"status": "error", "detail": str(e)}
