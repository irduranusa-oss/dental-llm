# ─────────────────────────────────────────────────────────────────────────────
# Archivo: app/main.py
# Descripción: API FastAPI que consulta Wikipedia (MediaWiki API), limpia texto,
# cachea en SQLite, trocea en chunks y expone endpoints para RAG/LLM.
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
import json
import math
import sqlite3
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import requests

WIKI_ENDPOINT = "https://{lang}.wikipedia.org/w/api.php"
DEFAULT_LANG = os.getenv("WIKI_LANG", "es")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24h
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "1200"))       # tamaño de chunk aprox ~800-1000 tokens
MIN_CHUNK_CHARS = 400

DB_PATH = os.getenv("DB_PATH", "cache.db")

app = FastAPI(title="Wikipedia RAG Tool", version="1.0.0", description="Servicio de integración con Wikipedia para LLMs (RAG).")

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de caché (SQLite)
# ─────────────────────────────────────────────────────────────────────────────
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            lang TEXT,
            title TEXT,
            content TEXT,
            fetched_at INTEGER,
            PRIMARY KEY (lang, title)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            lang TEXT,
            query TEXT,
            results TEXT,
            fetched_at INTEGER,
            PRIMARY KEY (lang, query)
        )
    """)
    return conn

def cache_get_page(lang: str, title: str) -> Optional[str]:
    conn = db()
    cur = conn.execute("SELECT content, fetched_at FROM pages WHERE lang=? AND title=?", (lang, title))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    content, fetched = row
    if time.time() - fetched > CACHE_TTL_SECONDS:
        return None
    return content

def cache_put_page(lang: str, title: str, content: str):
    conn = db()
    conn.execute(
        "INSERT OR REPLACE INTO pages (lang, title, content, fetched_at) VALUES (?, ?, ?, ?)",
        (lang, title, content, int(time.time()))
    )
    conn.commit()
    conn.close()

def cache_get_search(lang: str, query: str) -> Optional[List[Dict[str, Any]]]:
    conn = db()
    cur = conn.execute("SELECT results, fetched_at FROM search_cache WHERE lang=? AND query=?", (lang, query))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    results_json, fetched = row
    if time.time() - fetched > CACHE_TTL_SECONDS:
        return None
    return json.loads(results_json)

def cache_put_search(lang: str, query: str, results: List[Dict[str, Any]]):
    conn = db()
    conn.execute(
        "INSERT OR REPLACE INTO search_cache (lang, query, results, fetched_at) VALUES (?, ?, ?, ?)",
        (lang, query, json.dumps(results), int(time.time()))
    )
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# MediaWiki API Wrappers
# ─────────────────────────────────────────────────────────────────────────────
def wiki_search(query: str, lang: str, limit: int = 5) -> List[Dict[str, Any]]:
    # intenta caché
    cached = cache_get_search(lang, query)
    if cached:
        return cached

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1
    }
    url = WIKI_ENDPOINT.format(lang=lang)
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error consultando Wikipedia search: {r.text}")
    data = r.json()
    results = []
    for item in data.get("query", {}).get("search", []):
        results.append({
            "title": item.get("title"),
            "snippet": item.get("snippet"),
            "pageid": item.get("pageid")
        })
    cache_put_search(lang, query, results)
    return results

def wiki_get_plain_text(title: str, lang: str) -> str:
    # intenta caché
    cached = cache_get_page(lang, title)
    if cached:
        return cached

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
        "utf8": 1
    }
    url = WIKI_ENDPOINT.format(lang=lang)
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Error consultando Wikipedia page: {r.text}")
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise HTTPException(status_code=404, detail="Página no encontrada en Wikipedia.")
    # pages es un dict con clave pageid
    page = next(iter(pages.values()))
    content = page.get("extract", "")
    if not content:
        raise HTTPException(status_code=404, detail="No hay extracto de texto para ese título.")
    # guarda en caché
    cache_put_page(lang, title, content)
    return content

def simple_chunk(text: str, max_len: int = MAX_CHUNK_CHARS, overlap: int = 150) -> List[str]:
    text = text.strip()
    if len(text) <= max_len:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_len, len(text))
        chunk = text[i:end].strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        i = end - overlap
        if i < 0:
            i = end
    return chunks if chunks else [text]

# ─────────────────────────────────────────────────────────────────────────────
# Modelos de request/response
# ─────────────────────────────────────────────────────────────────────────────
class SearchResponse(BaseModel):
    title: str
    snippet: str
    pageid: int

class PageResponse(BaseModel):
    title: str
    language: str
    content_chars: int
    chunks: List[str]

class ToolCallRequest(BaseModel):
    # Entrada genérica para "function/tool calling"
    query: str
    lang: Optional[str] = None
    top_k: int = 3
    max_chars: int = MAX_CHUNK_CHARS

class ToolCallResponse(BaseModel):
    query: str
    language: str
    candidates: List[Dict[str, Any]]  # [{title, snippet, chunks:[...] }]

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "service": "wikipedia-rag", "version": "1.0.0"}

@app.get("/search", response_model=List[SearchResponse])
def search(q: str = Query(..., description="Consulta a buscar en Wikipedia"), lang: str = DEFAULT_LANG, limit: int = 5):
    results = wiki_search(q, lang, limit=limit)
    return [SearchResponse(title=r["title"], snippet=r["snippet"], pageid=r["pageid"]) for r in results]

@app.get("/page", response_model=PageResponse)
def page(title: str = Query(..., description="Título exacto de la página"), lang: str = DEFAULT_LANG, max_chars: int = MAX_CHUNK_CHARS):
    content = wiki_get_plain_text(title, lang)
    chunks = simple_chunk(content, max_len=max_chars)
    return PageResponse(title=title, language=lang, content_chars=len(content), chunks=chunks)

@app.get("/chunks", response_model=PageResponse)
def chunks(q: str = Query(..., description="Consulta; se toma el primer resultado"), lang: str = DEFAULT_LANG, max_chars: int = MAX_CHUNK_CHARS):
    results = wiki_search(q, lang, limit=1)
    if not results:
        raise HTTPException(status_code=404, detail="Sin resultados.")
    title = results[0]["title"]
    content = wiki_get_plain_text(title, lang)
    chunks = simple_chunk(content, max_len=max_chars)
    return PageResponse(title=title, language=lang, content_chars=len(content), chunks=chunks)

@app.post("/tool", response_model=ToolCallResponse)
def tool(req: ToolCallRequest):
    lang = req.lang or DEFAULT_LANG
    results = wiki_search(req.query, lang, limit=req.top_k)
    candidates = []
    for r in results:
        title = r["title"]
        try:
            content = wiki_get_plain_text(title, lang)
            chunks = simple_chunk(content, max_len=req.max_chars)
        except HTTPException:
            chunks = []
        candidates.append({
            "title": title,
            "snippet": r.get("snippet", ""),
            "chunks": chunks
        })
    return ToolCallResponse(query=req.query, language=lang, candidates=candidates)

# ─────────────────────────────────────────────────────────────────────────────
# Spec de herramienta para LLMs con function calling (puedes pegarlo en tu orquestador)
# ─────────────────────────────────────────────────────────────────────────────
TOOL_SPEC = {
    "name": "wikipedia_tool",
    "description": "Busca artículos en Wikipedia y devuelve texto limpio troceado para RAG.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Tema o consulta para buscar en Wikipedia."},
            "lang": {"type": "string", "description": "Código de idioma ISO, ej. 'es', 'en'."},
            "top_k": {"type": "integer", "description": "Número de artículos candidatos.", "default": 3},
            "max_chars": {"type": "integer", "description": "Tamaño de cada chunk en caracteres.", "default": MAX_CHUNK_CHARS}
        },
        "required": ["query"]
    }
}

@app.get("/tool/spec")
def tool_spec():
    return TOOL_SPEC


# ─────────────────────────────────────────────────────────────────────────────
# Archivo: requirements.txt
# ─────────────────────────────────────────────────────────────────────────────
# fastapi y uvicorn para la API, requests para HTTP.
# (sqlite3 viene en la stdlib)
# ─ Guardar este bloque en requirements.txt ─
# fastapi
# uvicorn
# requests

# ─────────────────────────────────────────────────────────────────────────────
# Archivo: Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# syntax=docker/dockerfile:1
# Imagen base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# Exponer puerto para Render
EXPOSE 10000
# Uvicorn sirve en 0.0.0.0:10000 para Render
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]

# ─────────────────────────────────────────────────────────────────────────────
# Archivo: render.yaml
# ─────────────────────────────────────────────────────────────────────────────
# Servicio web en Render. Ajusta el nombre a tu preferencia.
# Guardar como render.yaml en la raíz del repo.
# 
# Nota: Render detecta automáticamente Dockerfile.
#
services:
  - type: web
    name: wikipedia-rag-tool
    env: docker
    plan: free
    autoDeploy: true
    runtime: docker
    healthCheckPath: /health
    envVars:
      - key: WIKI_LANG
        value: es
      - key: CACHE_TTL_SECONDS
        value: "86400"
      - key: MAX_CHUNK_CHARS
        value: "1200"

# ─────────────────────────────────────────────────────────────────────────────
# Archivo: README.md (mini)
# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia RAG Tool
#
# 1) Clona/crea repo con estos archivos.
# 2) Conecta el repo a Render → New Web Service → Docker → despliega.
# 3) Endpoints clave:
#    - GET  /health
#    - GET  /search?q=odontología&lang=es
#    - GET  /page?title=Prostodoncia&lang=es
#    - GET  /chunks?q=prótesis%20dental&lang=es
#    - POST /tool   (body JSON: {"query":"prótesis dental","lang":"es","top_k":3})
#    - GET  /tool/spec
#
# Ejemplos cURL:
#   curl "$HOST/health"
#   curl "$HOST/search?q=protesis%20dental&lang=es"
#   curl "$HOST/page?title=Pr%C3%B3tesis%20dental&lang=es"
#   curl "$HOST/chunks?q=corona%20dental&lang=es"
#   curl -X POST "$HOST/tool" -H "Content-Type: application/json" \
#        -d '{"query":"cerámica dental","lang":"es","top_k":2}'
#
# Integración con tu LLM:
# - Si usas function-calling (OpenAI, etc.), registra /tool/spec y, cuando el modelo pida la herramienta,
#   haz POST a /tool y alimenta los "chunks" como contexto del mensaje del sistema o del usuario.
#
# Integración desde Wix (frontend):
#   import {fetch} from 'wix-fetch'; // o fetch nativo
#   export function getWikiChunks(query) {
#     const url = 'https://TU_DOMINIO_RENDER/tool';
#     return fetch(url, {
#       method: 'POST',
#       headers: {'Content-Type': 'application/json'},
#       body: JSON.stringify({ query, lang: 'es', top_k: 3, max_chars: 1200 })
#     }).then(r => r.json());
#   }
#   // Luego pasas candidates[*].chunks a tu llamada del LLM como contexto.
#
# Sugerencias pro:
# - Ajusta MAX_CHUNK_CHARS (800-1500) según el contexto que aguante tu LLM.
# - Implementa un simple re-ranker (opcional) por longitud o por TF-IDF antes de devolver chunks.
# - Activa Cloudflare/Render CDN si esperas mucho tráfico.
# - Si necesitas varios idiomas, setea lang dinámico por consulta.
