# cache.py
import time, hashlib

# Tiempo que guardamos cada respuesta (1 hora)
CACHE_TTL = 60 * 60

# Aquí está la caja (un diccionario de Python en memoria)
_cache = {}

def _normalize(text: str) -> str:
    """Convierte la pregunta en una forma simple para comparar mejor."""
    return " ".join(text.lower().strip().split())

def _key(pregunta: str, lang: str) -> str:
    """Crea una llave única con pregunta+idioma."""
    base = f"{lang}|{_normalize(pregunta)}"
    return hashlib.sha256(base.encode()).hexdigest()

def get_from_cache(pregunta: str, lang: str):
    """Busca si ya tenemos la respuesta en la caja."""
    k = _key(pregunta, lang)
    item = _cache.get(k)
    if not item:
        return None
    exp, val = item
    if exp < time.time():
        # ya caducó
        _cache.pop(k, None)
        return None
    return val

def save_to_cache(pregunta: str, lang: str, respuesta: str):
    """Guarda una respuesta nueva en la caja."""
    k = _key(pregunta, lang)
    exp = time.time() + CACHE_TTL
    _cache[k] = (exp, respuesta)
