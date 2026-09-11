"""Mocked multi-provider failover tests. No live paid APIs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.llm_providers import (  # noqa: E402
    CATEGORY_FAILOVER,
    CATEGORY_USER,
    DEFAULT_GEMINI_MODEL,
    FailoverError,
    LLMConfig,
    UserRequestError,
    classify_http_error,
    generate_with_failover,
    generate_with_provider,
    public_llm_status,
    reset_runtime_status,
)
from server.profile_router import (  # noqa: E402
    PROFILE_CARLOS,
    PROFILE_IGNACIO,
    PROFILE_NACHGPT,
    PROFILE_RAJAN,
    detect_relevant_profiles,
    route_question,
)


def _cfg(**kwargs) -> LLMConfig:
    data = dict(
        primary="gemini",
        secondary="openrouter",
        tertiary="openai",
        gemini_api_key="test-gemini",
        openrouter_api_key="test-openrouter",
        openai_api_key="test-openai",
        max_provider_retries=1,
        timeout_seconds=5,
    )
    data.update(kwargs)
    return LLMConfig(**data)


def _ok(text: str):
    def _fn(system_prompt, user_prompt, config):
        return text

    return _fn


def _fail_named(provider: str, reason: str, retryable: bool = True):
    def _fn(system_prompt, user_prompt, config):
        raise FailoverError(provider, reason, retryable=retryable)

    return _fn


class ClassifyTests(unittest.TestCase):
    def test_429_is_failover_quota(self):
        err = classify_http_error("openai", 429, '{"error":{"code":"insufficient_quota"}}')
        self.assertEqual(err.category, CATEGORY_FAILOVER)
        self.assertEqual(err.reason, "quota")

    def test_invalid_api_key_is_failover_not_user(self):
        err = classify_http_error(
            "gemini",
            400,
            '{"error":{"status":"INVALID_ARGUMENT","message":"API key not valid. Please pass a valid API key."}}',
        )
        self.assertEqual(err.category, CATEGORY_FAILOVER)
        self.assertEqual(err.reason, "auth")

    def test_5xx_is_failover(self):
        err = classify_http_error("gemini", 503, "unavailable")
        self.assertEqual(err.category, CATEGORY_FAILOVER)

    def test_400_invalid_request_is_user(self):
        err = classify_http_error("gemini", 400, "invalid_request")
        self.assertEqual(err.category, CATEGORY_USER)
        self.assertIsInstance(err, UserRequestError)

    def test_content_validation_is_user(self):
        err = classify_http_error("openrouter", 400, "content blocked by safety")
        self.assertEqual(err.reason, "content_validation")


class FailoverTests(unittest.TestCase):
    def setUp(self):
        reset_runtime_status()

    def test_primary_success(self):
        impls = {
            "gemini": _ok("gemini-answer"),
            "openrouter": _ok("should-not-run"),
            "openai": _ok("should-not-run"),
        }
        result = generate_with_failover("sys", "user", config=_cfg(), impls=impls)
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.text, "gemini-answer")

    def test_primary_429_secondary_success(self):
        impls = {
            "gemini": _fail_named("gemini", "quota"),
            "openrouter": _ok("openrouter-answer"),
            "openai": _ok("should-not-run"),
        }
        result = generate_with_failover("sys", "user", config=_cfg(), impls=impls)
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.text, "openrouter-answer")

    def test_primary_timeout_secondary_success(self):
        impls = {
            "gemini": _fail_named("gemini", "timeout"),
            "openrouter": _ok("openrouter-after-timeout"),
            "openai": _ok("should-not-run"),
        }
        result = generate_with_failover("sys", "user", config=_cfg(), impls=impls)
        self.assertEqual(result.provider, "openrouter")
        self.assertIn("timeout", result.failover_reason or "")

    def test_primary_and_secondary_fail_tertiary_success(self):
        impls = {
            "gemini": _fail_named("gemini", "quota"),
            "openrouter": _fail_named("openrouter", "timeout"),
            "openai": _ok("openai-answer"),
        }
        result = generate_with_failover("sys", "user", config=_cfg(), impls=impls)
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.text, "openai-answer")

    def test_all_providers_fail(self):
        impls = {
            "gemini": _fail_named("gemini", "quota"),
            "openrouter": _fail_named("openrouter", "timeout"),
            "openai": _fail_named("openai", "http_503"),
        }
        with self.assertRaises(FailoverError) as ctx:
            generate_with_failover("sys", "user", config=_cfg(), impls=impls)
        self.assertEqual(ctx.exception.provider, "all")

    def test_missing_primary_key(self):
        cfg = _cfg(gemini_api_key="")
        impls = {
            "gemini": _ok("should-not-run"),
            "openrouter": _ok("secondary-ok"),
            "openai": _ok("should-not-run"),
        }
        result = generate_with_failover("sys", "user", config=cfg, impls=impls)
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual(result.text, "secondary-ok")

    def test_openai_429_failover(self):
        cfg = _cfg(primary="openai", secondary="gemini", tertiary="openrouter")
        impls = {
            "openai": _fail_named("openai", "quota"),
            "gemini": _ok("gemini-after-openai-quota"),
            "openrouter": _ok("should-not-run"),
        }
        result = generate_with_failover("sys", "user", config=cfg, impls=impls)
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(result.text, "gemini-after-openai-quota")

    def test_user_error_does_not_failover(self):
        def bad_user(system_prompt, user_prompt, config):
            raise UserRequestError("gemini", "invalid_request")

        impls = {
            "gemini": bad_user,
            "openrouter": _ok("should-not-run"),
            "openai": _ok("should-not-run"),
        }
        with self.assertRaises(UserRequestError):
            generate_with_failover("sys", "user", config=_cfg(), impls=impls)

    def test_generate_with_provider_missing_key(self):
        with self.assertRaises(Exception):
            generate_with_provider("gemini", "s", "u", config=_cfg(gemini_api_key=""))

    def test_default_gemini_model_documented(self):
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-2.5-flash")

    def test_status_has_no_secret_fields(self):
        payload = public_llm_status(_cfg())
        blob = str(payload).lower()
        self.assertNotIn("api_key", blob)
        self.assertNotIn("bearer", blob)
        self.assertIn("primary", payload)
        self.assertIn("last_provider_used", payload)


class ProfileIndependenceTests(unittest.TestCase):
    def test_ignacio_profile(self):
        self.assertIn(PROFILE_IGNACIO, detect_relevant_profiles("Who is Ignacio Ramirez Duran?"))

    def test_nachgpt_and_ignacio(self):
        flags = route_question("Who created NACHGPT?").flags()
        self.assertEqual(flags["IGNACIO_PROFILE_LOADED"], "YES")
        self.assertEqual(flags["NACHGPT_PROFILE_LOADED"], "YES")

    def test_rajan_profile(self):
        self.assertIn(PROFILE_RAJAN, detect_relevant_profiles("Who is Dr Rajan Sheth?"))

    def test_carlos_profile(self):
        self.assertIn(PROFILE_CARLOS, detect_relevant_profiles("Who is Carlos Ortiz?"))

    def test_irrelevant_promotion(self):
        flags = route_question("zirconia sintering temperature").flags()
        self.assertEqual(flags["IGNACIO_PROFILE_LOADED"], "NO")
        self.assertEqual(flags["RAJAN_PROFILE_LOADED"], "NO")
        self.assertEqual(flags["CARLOS_PROFILE_LOADED"], "NO")
        self.assertEqual(flags["PROMOTIONAL_PROFILE_INCLUDED"], "NO")


if __name__ == "__main__":
    unittest.main()
