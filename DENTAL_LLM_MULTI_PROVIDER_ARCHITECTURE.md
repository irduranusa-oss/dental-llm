# Dental-LLM multi-provider architecture

Repo: `irduranusa-oss/dental-llm`  
Hosts: Railway `https://dental-llm-production.up.railway.app` (test) and Render `https://dental-llm.onrender.com` (rollback).  
Webhooks stay on Render until a later cutover.

## Why

Live chat failed when OpenAI returned 429 / no credits. The assistant now has three HTTP providers and automatic failover. Profile routing is unchanged and does not depend on which model answered.

## Order

Default:

1. PRIMARY = Gemini
2. SECONDARY = OpenRouter
3. TERTIARY = OpenAI

Env overrides (names only):

- `LLM_PROVIDER_PRIMARY` (default `gemini`)
- `LLM_PROVIDER_SECONDARY` (default `openrouter`)
- `LLM_PROVIDER_TERTIARY` (default `openai`)

## Request path

```
question
  → detect language
  → detect_relevant_profiles / build_system_context
  → generate_with_failover()
      Gemini → OpenRouter → OpenAI
  → optional Wikipedia enrichment (still on Render)
  → answer
```

`generate_answer()` remains the only path used by `/chat`, WhatsApp text, and WhatsApp audio.

## Defaults (documented 2026-09-10)

| Provider | Env key | Default model | Notes |
| --- | --- | --- | --- |
| Gemini | `GEMINI_API_KEY` (fallback `GOOGLE_API_KEY`) | `gemini-2.5-flash` | Official Flash still listed. `gemini-2.0-flash` is shut down. Override with `GEMINI_MODEL` (`gemini-2.5-flash-lite`, `gemini-3.5-flash-lite`, or `gemini-flash-latest`). |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/free` | Set `OPENROUTER_MODEL` to a currently available free slug if that alias is gone. |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` | Optional tertiary. Quota / billing errors failover away from it. |

No secrets are hardcoded.

## Failover

Failover on: 429, quota, billing, timeout, 5xx, connection errors, missing key, auth/not-found on that provider.

No failover on: invalid request, content validation, programming errors in our wrapper.

`MAX_PROVIDER_RETRIES` default `1` (one extra try on the same provider for temporary errors, then next provider).

`LLM_REQUEST_TIMEOUT_SECONDS` default `75`.

Cost caps: `LLM_MAX_OUTPUT_TOKENS` default `1024`, `LLM_MAX_INPUT_CHARS` default `16000` (does not clip normal questions).

## Health

`GET /health` returns `ok`, `app=ok`, `llm_provider_configured=yes|no`. No keys.

`GET /health/llm` returns provider order, configured flags, last provider used, last failover reason. No keys.

## Logging

Safe lines only:

```
LLM_PROVIDER_ATTEMPT=gemini
LLM_PROVIDER_SUCCESS=gemini
LLM_PROVIDER_FAIL=gemini reason=quota
LLM_PROVIDER_FAILOVER=openrouter
```

## What was not changed

- Wikipedia stays on Render (`WIKIPEDIA_TOOL_URL`).
- No nach-prod-db.
- No webhook cutover.
- NACHGPT laboratory production is out of scope.
