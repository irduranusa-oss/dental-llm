# ========= Idiomas =========
LANG_NAME = {"es": "Spanish", "en": "English", "pt": "Portuguese", "fr": "French"}

def detect_lang(text: str) -> str:
    """
    Heurística rápida y robusta para ES/EN/PT/FR.
    - Acentos/¿¡ => ES
    - Tildes/ç portuguesas => PT
    - Diacríticos FR => FR
    - Si no, palabras muy frecuentes para resolver empates.
    """
    t = (text or "").strip().lower()

    # Señales fuertes
    if re.search(r"[áéíóúñ¿¡]", t):
        return "es"
    if re.search(r"[ãõáéíóúç]", t):
        return "pt"
    if re.search(r"[àâçéèêëîïôùûüÿœ]", t):
        return "fr"

    # Palabras frecuentes
    es_hits = len(re.findall(r"\b(el|la|de|que|y|para|con|qué|cómo|cuál|dónde)\b", t))
    en_hits = len(re.findall(r"\b(the|and|for|what|how|where|with|you|your)\b", t))
    pt_hits = len(re.findall(r"\b(o|a|de|que|e|para|com|você|seu)\b", t))
    fr_hits = len(re.findall(r"\b(le|la|de|et|pour|avec|vous|votre|quoi|comment)\b", t))

    scores = {"es": es_hits, "en": en_hits, "pt": pt_hits, "fr": fr_hits}
    # Si hay empate, preferimos EN como fallback
    return max(scores, key=lambda k: (scores[k], 1 if k == "en" else 0)) or "en"

def call_openai(question: str, lang_hint: str | None = None) -> str:
    """
    Llama al modelo con system prompt e INSTRUCCIÓN DURA de idioma.
    """
    sys = SYSTEM_PROMPT
    if lang_hint in LANG_NAME:
        # Instrucción explícita y contundente para evitar respuestas en inglés.
        sys += f"\nResponde SIEMPRE en {LANG_NAME[lang_hint]}. Si el usuario cambia de idioma, adapta la respuesta al nuevo idioma."

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": question},
            ],
            temperature=OPENAI_TEMP,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI error:", e)
        raise HTTPException(status_code=500, detail="Error con el modelo")

# ========= Tickets a Google Sheet =========
SHEETS_WEBHOOK_URL = os.getenv("SHEET_WEBHOOK", "") or os.getenv("SHEETS_WEBHOOK_URL", "")

def send_ticket(numero: str, mensaje: str, respuesta: str, etiqueta: str = "NochGPT"):
    """
    Envía un ticket simple al Apps Script (si está configurado).
    El Apps Script actual espera: fecha, numero, mensaje, respuesta, etiqueta.
    """
    if not SHEETS_WEBHOOK_URL:
        print("Falta SHEETS_WEBHOOK_URL (no se envía ticket)")
        return

    payload = {
        "fecha": datetime.utcnow().isoformat(timespec="seconds"),
        "numero": str(numero or ""),
        "mensaje": mensaje or "",
        "respuesta": respuesta or "",
        "etiqueta": etiqueta or "NochGPT",
    }
    try:
        r = requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=15)
        ok = (r.status_code // 100) == 2
        print("Ticket sheet =>", r.status_code, r.text[:200])
        return ok
    except Exception as e:
        print("Ticket error:", e)
        return False

# ========= Webhook WhatsApp =========
@app.post("/webhook")
async def webhook_handler(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"received": False, "error": "invalid_json"})

    print("📩 Payload recibido:", data)

    try:
        entry = (data.get("entry") or [{}])[0]
        changes = (entry.get("changes") or [{}])[0]
        value = changes.get("value") or {}

        # A) Mensajes nuevos
        msgs = value.get("messages") or []
        if not msgs:
            # B) Status u otros
            if value.get("statuses"):
                return {"status": "status_ok"}
            return {"status": "no_message"}

        msg = msgs[0]
        from_number = msg.get("from")
        mtype = msg.get("type")

        # ---- TEXTO ----
        if mtype == "text":
            user_text = (msg.get("text") or {}).get("body", "").strip()
            if not user_text:
                return {"status": "ok"}

            lang = detect_lang(user_text)
            try:
                answer = call_openai(user_text, lang_hint=lang)
            except Exception:
                # fallback mínimamente en el mismo idioma
                answer = "Lo siento, tuve un problema procesando tu mensaje." if lang == "es" else \
                         "Desculpe, tive um problema ao processar sua mensagem." if lang == "pt" else \
                         "Désolé, j'ai eu un problème en traitant votre message." if lang == "fr" else \
                         "Sorry, I had trouble processing your message."

            if from_number:
                wa_send_text(from_number, answer)
                # Ticket
                send_ticket(from_number, user_text, answer)

            return {"status": "ok"}

        # ---- AUDIO (nota de voz) ----
        if mtype == "audio":
            aud = msg.get("audio") or {}
            media_id = aud.get("id")
            if media_id and from_number:
                try:
                    url = wa_get_media_url(media_id)
                    path, mime = wa_download_media(url)
                    print(f"🎧 Audio guardado en {path} ({mime})")

                    transcript = transcribe_audio_with_openai(path)
                    if transcript:
                        lang = detect_lang(transcript)
                        answer = call_openai(
                            f"Transcripción del audio del usuario:\n\"\"\"{transcript}\"\"\"\n\n"
                            "Responde de forma útil, breve y enfocada en odontología cuando aplique.",
                            lang_hint=lang,
                        )
                        wa_send_text(from_number, f"🗣️ *Transcripción*:\n{transcript}\n\n💬 *Respuesta*:\n{answer}")
                        # Ticket
                        send_ticket(from_number, f"[AUDIO] {transcript}", answer)
                    else:
                        msg_txt = "No pude transcribir el audio. ¿Puedes intentar otra nota de voz?"
                        wa_send_text(from_number, msg_txt)
                        send_ticket(from_number, "[AUDIO] (sin transcripción)", msg_txt)

                except Exception as e:
                    print("Error audio:", e)
                    msg_txt = "No pude procesar el audio. ¿Puedes intentar de nuevo?"
                    wa_send_text(from_number, msg_txt)
                    send_ticket(from_number, "[AUDIO] error", msg_txt)

            return {"status": "ok"}

        # ---- Otros tipos (opcionalmente ignoramos) ----
        if from_number:
            wa_send_text(
                from_number,
                "Recibí tu mensaje. Por ahora manejo *texto* y *notas de voz*."
            )
            send_ticket(from_number, "[OTRO TIPO]", "Mensaje no soportado (solo texto y audio).")

        return {"status": "ok"}

    except Exception as e:
        print("❌ Error en webhook:", e)
        return {"status": "error"}
