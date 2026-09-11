# Dental-LLM multi-provider test results

Date: 2026-09-10  
Repo: `C:\Users\irdur\Desktop\dental-llm`  
Command: `python -m unittest tests.test_profile_router tests.test_llm_providers`  
Live paid APIs: none in unit tests

## Result

`Ran 53 tests`  
`OK`

- Profile routing: 32/32
- Provider failover + classification + profile independence: 21/21

## Required mock cases

| Case | Result |
| --- | --- |
| PRIMARY_SUCCESS | PASS |
| PRIMARY_429_SECONDARY_SUCCESS | PASS |
| PRIMARY_TIMEOUT_SECONDARY_SUCCESS | PASS |
| PRIMARY_AND_SECONDARY_FAIL_TERTIARY_SUCCESS | PASS |
| ALL_PROVIDERS_FAIL | PASS |
| MISSING_PRIMARY_KEY | PASS |
| OPENAI_429_FAILOVER | PASS |
| User invalid_request does not failover | PASS |

## Profile routing (provider-independent)

| Question | Result |
| --- | --- |
| Who is Ignacio Ramirez Duran? | IGNACIO=YES |
| Who created NACHGPT? | IGNACIO + NACHGPT |
| Who is Dr Rajan Sheth? | RAJAN=YES |
| Who is Carlos Ortiz? | CARLOS=YES |
| zirconia sintering temperature | no promotional people |

`TESTS_PASS=YES`  
`WEBHOOKS_CUTOVER=NO`
