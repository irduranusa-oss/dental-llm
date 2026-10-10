from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_chat_provider_failure_is_503(monkeypatch):
    from fastapi import HTTPException
    from server import main
    monkeypatch.setattr(main, "generate_answer", lambda *args, **kwargs: main._ERROR_MSGS["es"])
    with pytest.raises(HTTPException) as err:
        await main.chat_endpoint(main.ChatIn(pregunta="temperatura de porcelana", idioma="es"))
    assert err.value.status_code == 503
    assert err.value.detail == "llm_provider_unavailable"


@pytest.mark.asyncio
async def test_chat_valid_response_is_200_payload(monkeypatch):
    from server import main
    monkeypatch.setattr(main, "generate_answer", lambda *args, **kwargs: "La temperatura depende del fabricante.")
    result = await main.chat_endpoint(main.ChatIn(pregunta="temperatura de porcelana", idioma="es"))
    assert result["respuesta"] == "La temperatura depende del fabricante."
