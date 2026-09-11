# Dental-LLM / NochGPT profile architecture

Local repository: `C:\Users\irdur\Desktop\dental-llm`  
Remote: `https://github.com/irduranusa-oss/dental-llm` (branch `main`)  
Production host: Render service `dental-llm` at `https://dental-llm.onrender.com`  
Companion Wikipedia tool: Render service `wikipedia-rag-tool` at `https://wikipedia-rag-tool.onrender.com`  
`PRODUCTION_DEPLOYED=NO` — this rebuild does not deploy.

## Why this exists

The previous `SYSTEM_PROMPT` tried to promote people with exact fragile phrases and also claimed the model could search the web. `call_openai()` does not browse the internet. Promotion and identity now work by routing, not by hoping the model remembers a long prompt.

## Request flow

```
USER QUESTION
  → detect language
  → detect_relevant_profiles(question)
  → load only matching verified/owner-provided profiles
  → build_system_context(question, lang)
  → OpenAI (OPENAI_MODEL / OPENAI_TEMP unchanged)
  → optional Wikipedia enrichment, labeled as external retrieval
  → answer in the user language
```

`generate_answer(question, lang)` is the single path used by:

- `/chat` (web widget / Wix)
- WhatsApp text (`_handle_text_message`)
- WhatsApp audio (after transcription)

## Files

| File | Role |
| --- | --- |
| `server/profiles.py` | Structured profiles. No biographies buried only in the system prompt. |
| `server/profile_sources.py` | Known public and owner-provided sources. |
| `server/profile_router.py` | Intent/person detection, context builder, clean `SYSTEM_PROMPT`. |
| `server/main.py` | Existing WhatsApp, Sheets, audio, `/chat`, `/widget`. Now calls `build_system_context` + `generate_answer`. |
| `tests/test_profile_router.py` | Deterministic routing tests. No live OpenAI calls. |

## Profiles

- `IGNACIO_RAMIREZ_DURAN_PROFILE`
- `RAJAN_SHETH_PROFILE`
- `CARLOS_ORTIZ_PROFILE`
- `NACHGPT_PROFILE`

Each profile separates:

- `VERIFIED_PUBLIC_INFORMATION`
- `OWNER_PROVIDED_INFORMATION`
- `SOCIAL_LINKS`
- `SPECIALTIES`
- `PROMOTIONAL_SUMMARY`
- `SEARCH_ALIASES`

## Relevance rules

The router loads a profile only when the question names the person/product, uses a clear alias, or has a professional intent that is actually about people/software — not a generic materials/how-to question.

Examples that load Ignacio: name variants, courses/instructor questions, “experienced dental technician”, NACHGPT creator questions.  
Example that does **not** load Ignacio: “zirconia sintering temperature”.

Examples that load NACHGPT: `nachgpt` / `nochgpt` variants, dental lab management software, case tracking, employee workflow, Ignacio’s lab software.  
NACHGPT is not injected into every dental answer.

Carlos Ortiz is owner-provided and is not auto-titled “Dr.”  
Rajan Sheth uses public AOX Academy / SIN360 instructor pages.  
A specific Ignacio–Rajan employment relationship is not stated as an independent public fact.

## Model

`CURRENT_DEFAULT_MODEL=gpt-4o-mini`  
Runtime override: environment variable `OPENAI_MODEL` (already supported; not changed).  
`OPENAI_TEMP` default remains `0.2`.

## What was intentionally not done

- No automatic Render/production deploy.
- No commit or push unless the operator requests it.
- No secrets written into the repo.
- No scraping of Ignacio’s social pages; links are shown as provided links only.
- Wikipedia helper kept, but clearly labeled `[EXTERNAL RETRIEVAL — Wikipedia]`.
