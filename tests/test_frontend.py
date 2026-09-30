"""Runs the Streamlit app headlessly against a fake backend."""

from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

FRONTEND = str(Path(__file__).resolve().parents[1] / "frontend.py")
API_URL = "http://127.0.0.1:8000"

SOURCE = {
    "number": 1,
    "chunk_id": "abc:0",
    "doc_id": "abc",
    "filename": "melodi.pdf",
    "page": 3,
    "text": "The core principle of MELODI is to represent short-term and long-term memory ...",
    "score": 7.5,
}


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeBackend:
    def __init__(self, llm_configured: bool = True, ask_status: int = 200):
        self.llm_configured = llm_configured
        self.ask_status = ask_status
        self.asks: list[dict] = []

    def __call__(self, method: str, url: str, timeout: float, **kwargs) -> FakeResponse:
        path = url.removeprefix(API_URL)
        if (method, path) == ("GET", "/documents"):
            return FakeResponse(
                200, [{"doc_id": "abc", "filename": "melodi.pdf", "pages": 23, "chunks": 140}]
            )
        if (method, path) == ("GET", "/health"):
            return FakeResponse(
                200, {"status": "ok", "documents": 1, "chunks": 140, "llm_configured": self.llm_configured}
            )
        if (method, path) == ("POST", "/ask"):
            self.asks.append(kwargs["json"])
            if self.ask_status != 200:
                return FakeResponse(self.ask_status, {"detail": "The LLM provider returned an error."})
            question = kwargs["json"]["question"]
            return FakeResponse(
                200,
                {
                    "question": question,
                    "standalone_question": question,
                    "answer": f"Answer to: {question} [1]",
                    "sources": [SOURCE],
                    "timings_ms": {"total": 1.0},
                },
            )
        raise AssertionError(f"unexpected request {method} {path}")


def start(monkeypatch, backend: FakeBackend) -> AppTest:
    monkeypatch.setattr(requests, "request", backend)
    return AppTest.from_file(FRONTEND, default_timeout=30).run()


def test_sidebar_lists_indexed_documents(monkeypatch):
    app = start(monkeypatch, FakeBackend())
    assert not app.exception
    assert any("melodi.pdf" in item.value and "23 pages" in item.value for item in app.sidebar.markdown)
    assert not app.sidebar.warning


def test_warns_when_the_server_has_no_llm_key(monkeypatch):
    app = start(monkeypatch, FakeBackend(llm_configured=False))
    assert any("GROQ_API_KEY" in warning.value for warning in app.sidebar.warning)


def test_answers_show_sources_and_follow_ups_send_history(monkeypatch):
    backend = FakeBackend()
    app = start(monkeypatch, backend)

    app.chat_input[0].set_value("What is MELODI?").run()
    assert app.chat_message[1].markdown[0].value == "Answer to: What is MELODI? [1]"
    assert [expander.label for expander in app.chat_message[1].expander] == ["Sources (1)"]

    app.chat_input[0].set_value("What are its limitations?").run()

    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant", "user", "assistant"]
    assert backend.asks[0] == {"question": "What is MELODI?", "history": []}
    assert backend.asks[1]["history"] == [
        {"role": "user", "content": "What is MELODI?"},
        {"role": "assistant", "content": "Answer to: What is MELODI? [1]"},
    ]


@pytest.mark.parametrize("status", [502, 503])
def test_failed_turns_are_shown_but_not_sent_as_history(monkeypatch, status):
    backend = FakeBackend(ask_status=status)
    app = start(monkeypatch, backend)

    app.chat_input[0].set_value("First question").run()
    assert any(f"Error {status}" in error.value for error in app.error)

    backend.ask_status = 200
    app.chat_input[0].set_value("Second question").run()
    assert backend.asks[1] == {"question": "Second question", "history": []}
