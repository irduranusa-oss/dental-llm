"""Multi-provider LLM router with classified failover.

PRIMARY=gemini → SECONDARY=openrouter → TERTIARY=openai
Business logic (profiles, language, Wikipedia) stays outside this module.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import quote

import requests

PROVIDER_GEMINI = "gemini"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_OPENAI = "openai"
KNOWN_PROVIDERS = (PROVIDER_GEMINI, PROVIDER_OPENROUTER, PROVIDER_OPENAI)

CATEGORY_FAILOVER = "failover"
CATEGORY_USER = "user"
CATEGORY_CONFIG = "config"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 75
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_MAX_INPUT_CHARS = 16000
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_PROVIDER_RETRIES = 1

GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class ProviderError(Exception):
    def __init__(self, provider: str, reason: str, category: str, retryable: bool = False):
        self.provider = provider
        self.reason = reason
        self.category = category
        self.retryable = retryable
        super().__init__(f"{provider}:{reason}")


class UserRequestError(ProviderError):
    def __init__(self, provider: str, reason: str = "invalid_request"):
        super().__init__(provider, reason, CATEGORY_USER, retryable=False)


class FailoverError(ProviderError):
    def __init__(self, provider: str, reason: str, retryable: bool = True):
        super().__init__(provider, reason, CATEGORY_FAILOVER, retryable=retryable)


class ConfigError(ProviderError):
    def __init__(self, provider: str, reason: str = "missing_key"):
        super().__init__(provider, reason, CATEGORY_CONFIG, retryable=False)


@dataclass
class LLMConfig:
    primary: str = PROVIDER_GEMINI
    secondary: str = PROVIDER_OPENROUTER
    tertiary: str = PROVIDER_OPENAI
    gemini_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    openrouter_api_key: str = ""
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    openai_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    temperature: float = DEFAULT_TEMPERATURE
    max_provider_retries: int = DEFAULT_MAX_PROVIDER_RETRIES

    def ordered_providers(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for name in (self.primary, self.secondary, self.tertiary):
            key = (name or "").strip().lower()
            if key in KNOWN_PROVIDERS and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def key_for(self, provider: str) -> str:
        if provider == PROVIDER_GEMINI:
            return (self.gemini_api_key or "").strip()
        if provider == PROVIDER_OPENROUTER:
            return (self.openrouter_api_key or "").strip()
        if provider == PROVIDER_OPENAI:
            return (self.openai_api_key or "").strip()
        return ""

    def model_for(self, provider: str) -> str:
        if provider == PROVIDER_GEMINI:
            return (self.gemini_model or DEFAULT_GEMINI_MODEL).strip()
        if provider == PROVIDER_OPENROUTER:
            return (self.openrouter_model or DEFAULT_OPENROUTER_MODEL).strip()
        if provider == PROVIDER_OPENAI:
            return (self.openai_model or DEFAULT_OPENAI_MODEL).strip()
        return ""

    def is_configured(self, provider: str) -> bool:
        return bool(self.key_for(provider))

    def any_configured(self) -> bool:
        return any(self.is_configured(p) for p in self.ordered_providers())


@dataclass
class ProviderCallResult:
    text: str
    provider: str
    failover_reason: Optional[str] = None


@dataclass
class RuntimeStatus:
    last_provider_used: Optional[str] = None
    last_failover_reason: Optional[str] = None
    last_success: bool = False
    attempts: list[str] = field(default_factory=list)


_STATUS = RuntimeStatus()
_HTTP = requests.Session()


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.getenv(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(lo, min(hi, value))


def load_config() -> LLMConfig:
    gemini_key = (
        os.getenv("GEMINI_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
    )
    return LLMConfig(
        primary=(os.getenv("LLM_PROVIDER_PRIMARY", PROVIDER_GEMINI) or PROVIDER_GEMINI).strip().lower(),
        secondary=(os.getenv("LLM_PROVIDER_SECONDARY", PROVIDER_OPENROUTER) or PROVIDER_OPENROUTER).strip().lower(),
        tertiary=(os.getenv("LLM_PROVIDER_TERTIARY", PROVIDER_OPENAI) or PROVIDER_OPENAI).strip().lower(),
        gemini_api_key=gemini_key,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip()
        or DEFAULT_OPENROUTER_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL,
        timeout_seconds=_env_float("LLM_REQUEST_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS, 5.0, 180.0),
        max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS, 64, 4096),
        max_input_chars=_env_int("LLM_MAX_INPUT_CHARS", DEFAULT_MAX_INPUT_CHARS, 500, 100000),
        temperature=_env_float("OPENAI_TEMP", DEFAULT_TEMPERATURE, 0.0, 2.0),
        max_provider_retries=_env_int("MAX_PROVIDER_RETRIES", DEFAULT_MAX_PROVIDER_RETRIES, 0, 2),
    )


def clip_input(text: str, max_chars: int) -> str:
    raw = text or ""
    if len(raw) <= max_chars:
        return raw
    return raw[: max(0, max_chars - 24)].rstrip() + "\n[truncated]"


def classify_http_error(provider: str, status_code: int, body_text: str = "") -> ProviderError:
    blob = (body_text or "").lower()
    if status_code == 429 or "insufficient_quota" in blob or "credit_balance" in blob:
        return FailoverError(provider, "quota", retryable=True)
    if "quota" in blob or "billing" in blob or "resource_exhausted" in blob:
        return FailoverError(provider, "quota", retryable=True)
    if "api key not valid" in blob or "api_key_invalid" in blob or "invalid api key" in blob:
        return FailoverError(provider, "auth", retryable=False)
    if status_code in (408, 409, 423, 425, 449) or 500 <= status_code <= 599:
        return FailoverError(provider, f"http_{status_code}", retryable=True)
    if status_code in (401, 403):
        return FailoverError(provider, "auth", retryable=False)
    if status_code == 404:
        return FailoverError(provider, "not_found", retryable=False)
    if status_code == 400 or "invalid_request" in blob or "invalid argument" in blob:
        if "safety" in blob or "blocked" in blob or "content" in blob:
            return UserRequestError(provider, "content_validation")
        return UserRequestError(provider, "invalid_request")
    if status_code == 422:
        return UserRequestError(provider, "invalid_request")
    return FailoverError(provider, f"http_{status_code}", retryable=True)


def classify_exception(provider: str, exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, requests.Timeout):
        return FailoverError(provider, "timeout", retryable=True)
    if isinstance(exc, (requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
        return FailoverError(provider, "connection", retryable=True)
    if isinstance(exc, (TypeError, AttributeError, ValueError, KeyError)):
        return UserRequestError(provider, "programming")
    return FailoverError(provider, "temporary", retryable=True)


def _safe_log(event: str, provider: str, reason: str = "") -> None:
    if reason:
        print(f"{event}={provider} reason={reason}")
    else:
        print(f"{event}={provider}")


def get_runtime_status() -> RuntimeStatus:
    return _STATUS


def reset_runtime_status() -> None:
    _STATUS.last_provider_used = None
    _STATUS.last_failover_reason = None
    _STATUS.last_success = False
    _STATUS.attempts = []


def public_llm_status(config: Optional[LLMConfig] = None) -> dict:
    cfg = config or load_config()
    return {
        "primary": cfg.primary,
        "secondary": cfg.secondary,
        "tertiary": cfg.tertiary,
        "primary_configured": cfg.is_configured(cfg.primary),
        "secondary_configured": cfg.is_configured(cfg.secondary),
        "tertiary_configured": cfg.is_configured(cfg.tertiary),
        "llm_provider_configured": "yes" if cfg.any_configured() else "no",
        "last_provider_used": _STATUS.last_provider_used,
        "last_failover_reason": _STATUS.last_failover_reason,
        "last_success": _STATUS.last_success,
        "gemini_model": cfg.gemini_model,
        "openrouter_model": cfg.openrouter_model,
        "openai_model": cfg.openai_model,
    }


def _extract_gemini_text(payload: dict) -> str:
    cands = (payload or {}).get("candidates") or []
    if not cands:
        raise FailoverError(PROVIDER_GEMINI, "empty", retryable=True)
    parts = ((cands[0] or {}).get("content") or {}).get("parts") or []
    text = "".join((p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
    if not text:
        raise FailoverError(PROVIDER_GEMINI, "empty", retryable=True)
    return text


def _extract_openai_compat_text(payload: dict, provider: str) -> str:
    choices = (payload or {}).get("choices") or []
    if not choices:
        raise FailoverError(provider, "empty", retryable=True)
    message = (choices[0] or {}).get("message") or {}
    text = (message.get("content") or "").strip()
    if not text:
        raise FailoverError(provider, "empty", retryable=True)
    return text


def _call_gemini(system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
    key = config.key_for(PROVIDER_GEMINI)
    if not key:
        raise ConfigError(PROVIDER_GEMINI)
    model = quote(config.model_for(PROVIDER_GEMINI), safe=".-")
    url = GEMINI_URL_TMPL.format(model=model)
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_output_tokens,
        },
    }
    try:
        resp = _HTTP.post(url, params={"key": key}, json=body, timeout=config.timeout_seconds)
    except Exception as exc:
        raise classify_exception(PROVIDER_GEMINI, exc) from exc
    if not resp.ok:
        raise classify_http_error(PROVIDER_GEMINI, resp.status_code, resp.text[:400])
    try:
        return _extract_gemini_text(resp.json() or {})
    except ProviderError:
        raise
    except Exception as exc:
        raise classify_exception(PROVIDER_GEMINI, exc) from exc


def _call_openrouter(system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
    key = config.key_for(PROVIDER_OPENROUTER)
    if not key:
        raise ConfigError(PROVIDER_OPENROUTER)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.model_for(PROVIDER_OPENROUTER),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
    }
    try:
        resp = _HTTP.post(OPENROUTER_URL, headers=headers, json=body, timeout=config.timeout_seconds)
    except Exception as exc:
        raise classify_exception(PROVIDER_OPENROUTER, exc) from exc
    if not resp.ok:
        raise classify_http_error(PROVIDER_OPENROUTER, resp.status_code, resp.text[:400])
    try:
        return _extract_openai_compat_text(resp.json() or {}, PROVIDER_OPENROUTER)
    except ProviderError:
        raise
    except Exception as exc:
        raise classify_exception(PROVIDER_OPENROUTER, exc) from exc


def _call_openai_chat(system_prompt: str, user_prompt: str, config: LLMConfig) -> str:
    key = config.key_for(PROVIDER_OPENAI)
    if not key:
        raise ConfigError(PROVIDER_OPENAI)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.model_for(PROVIDER_OPENAI),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
    }
    try:
        resp = _HTTP.post(OPENAI_URL, headers=headers, json=body, timeout=config.timeout_seconds)
    except Exception as exc:
        raise classify_exception(PROVIDER_OPENAI, exc) from exc
    if not resp.ok:
        raise classify_http_error(PROVIDER_OPENAI, resp.status_code, resp.text[:400])
    try:
        return _extract_openai_compat_text(resp.json() or {}, PROVIDER_OPENAI)
    except ProviderError:
        raise
    except Exception as exc:
        raise classify_exception(PROVIDER_OPENAI, exc) from exc


_DEFAULT_IMPLS: dict[str, Callable[[str, str, LLMConfig], str]] = {
    PROVIDER_GEMINI: _call_gemini,
    PROVIDER_OPENROUTER: _call_openrouter,
    PROVIDER_OPENAI: _call_openai_chat,
}


def generate_with_provider(
    provider: str,
    system_prompt: str,
    user_prompt: str,
    *,
    config: Optional[LLMConfig] = None,
    impls: Optional[dict[str, Callable[[str, str, LLMConfig], str]]] = None,
) -> str:
    cfg = config or load_config()
    name = (provider or "").strip().lower()
    if name not in KNOWN_PROVIDERS:
        raise UserRequestError(name or "unknown", "unknown_provider")
    if not cfg.is_configured(name):
        raise ConfigError(name)
    callers = impls or _DEFAULT_IMPLS
    caller = callers.get(name)
    if caller is None:
        raise UserRequestError(name, "unknown_provider")
    return caller(system_prompt, user_prompt, cfg)


def generate_with_failover(
    system_prompt: str,
    user_prompt: str,
    *,
    config: Optional[LLMConfig] = None,
    impls: Optional[dict[str, Callable[[str, str, LLMConfig], str]]] = None,
) -> ProviderCallResult:
    cfg = config or load_config()
    callers = impls or _DEFAULT_IMPLS
    system_prompt = clip_input(system_prompt, cfg.max_input_chars)
    user_prompt = clip_input(user_prompt, cfg.max_input_chars)
    providers = cfg.ordered_providers()
    last_reason = "none_configured" if not providers else "all_failed"
    reset_runtime_status()

    for index, provider in enumerate(providers):
        if not cfg.is_configured(provider):
            last_reason = "missing_key"
            _safe_log("LLM_PROVIDER_FAIL", provider, "missing_key")
            _STATUS.attempts.append(provider)
            nxt = providers[index + 1] if index + 1 < len(providers) else None
            if nxt:
                _safe_log("LLM_PROVIDER_FAILOVER", nxt)
            continue

        retries = max(0, int(cfg.max_provider_retries))
        for attempt in range(retries + 1):
            _safe_log("LLM_PROVIDER_ATTEMPT", provider)
            _STATUS.attempts.append(provider)
            try:
                text = generate_with_provider(
                    provider,
                    system_prompt,
                    user_prompt,
                    config=cfg,
                    impls=callers,
                )
                _safe_log("LLM_PROVIDER_SUCCESS", provider)
                _STATUS.last_provider_used = provider
                _STATUS.last_success = True
                _STATUS.last_failover_reason = last_reason if index > 0 or attempt > 0 else None
                return ProviderCallResult(
                    text=text,
                    provider=provider,
                    failover_reason=_STATUS.last_failover_reason,
                )
            except UserRequestError as exc:
                _safe_log("LLM_PROVIDER_FAIL", provider, exc.reason)
                _STATUS.last_provider_used = provider
                _STATUS.last_failover_reason = exc.reason
                _STATUS.last_success = False
                raise
            except ProviderError as exc:
                last_reason = exc.reason
                _safe_log("LLM_PROVIDER_FAIL", provider, exc.reason)
                if exc.category == CATEGORY_FAILOVER and exc.retryable and attempt < retries:
                    time.sleep(0.05)
                    continue
                nxt = providers[index + 1] if index + 1 < len(providers) else None
                if nxt:
                    _safe_log("LLM_PROVIDER_FAILOVER", nxt)
                break

    _STATUS.last_provider_used = None
    _STATUS.last_failover_reason = last_reason
    _STATUS.last_success = False
    raise FailoverError("all", last_reason, retryable=False)
