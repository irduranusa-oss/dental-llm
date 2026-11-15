"""Prevent operational fact leakage through the shared conversation history endpoint."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "server" / "main.py").read_text(encoding="utf-8")


def test_operational_reply_never_enters_global_history():
    start = SOURCE.index('@app.post("/nachgpt/chat")')
    end = SOURCE.index('@app.get("/history")', start)
    assert "_append_history(" not in SOURCE[start:end]


def test_history_endpoint_enforces_gateway_header():
    start = SOURCE.index('@app.get("/history")')
    end = SOURCE.index('@app.get("/webhook")', start)
    body = SOURCE[start:end]
    assert "request: Request" in body
    assert 'NACHGPT_GATEWAY_TOKEN' in body
    assert 'X-NACHGPT-Gateway-Token' in body
    assert "hmac.compare_digest" in body
    assert "status_code=403" in body
