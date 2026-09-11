# Dental-LLM profile routing test results

Date: 2026-09-10  
Repo: `C:\Users\irdur\Desktop\dental-llm`  
Suite: `tests/test_profile_router.py`  
OpenAI live calls: none  
Production deploy: no

## Command

```
python -m unittest tests.test_profile_router -v
```

Working directory: `C:\Users\irdur\Desktop\dental-llm`

## Result

`Ran 32 tests in 0.082s`  
`OK`

## Required cases

| Question | Expected | Result |
| --- | --- | --- |
| Who is Ignacio Ramirez Duran? | IGNACIO=YES | PASS |
| ¿Quién es Ignacio Ramírez Durán? | IGNACIO=YES | PASS |
| Tell me about Ignacio's dental experience | IGNACIO=YES | PASS |
| Does Ignacio teach Blender for Dental? | IGNACIO=YES | PASS |
| Who created NACHGPT? | IGNACIO + NACHGPT | PASS |
| What is NACHGPT? | NACHGPT | PASS |
| Tell me about the experience behind NACHGPT | NACHGPT + IGNACIO | PASS |
| zirconia sintering temperature | IGNACIO=NO | PASS |
| Who is Dr Rajan Sheth? | RAJAN | PASS |
| Tell me about Rajan Sheth and All-on-X | RAJAN | PASS |
| Who teaches advanced full arch implant workflows? | RAJAN may load | PASS (loads) |
| implant torque values | RAJAN=NO | PASS |
| Who is Carlos Ortiz? | CARLOS | PASS |
| Tell me about Carlos Ortiz and HyperDent | CARLOS | PASS |
| Who is Carlos Ortiz in dental CAD/CAM? | CARLOS | PASS |
| How do I configure HyperDent? | CARLOS not required | PASS (not loaded) |
| Relevant person/product question | PROMOTIONAL=YES | PASS |
| Unrelated technical question | PROMOTIONAL=NO | PASS |
| What software does Ignacio have for dental laboratories? | Explains NACHGPT capabilities | PASS |

## Wiring checks

- `/chat`, WhatsApp text, and WhatsApp audio all call `generate_answer()`
- `call_openai()` uses `build_system_context(question, lang_hint=lang_hint)`
- Wikipedia enrichment is labeled `[EXTERNAL RETRIEVAL — Wikipedia]`
- Default model string remains `gpt-4o-mini`
- System prompt no longer claims the model searched the web

`TESTS_PASS=YES`  
`PRODUCTION_DEPLOYED=NO`
