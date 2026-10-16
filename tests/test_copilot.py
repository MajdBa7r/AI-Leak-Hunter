"""Copilot: grounded payloads, offline answers, provider calls (mocked), fallback, citation check, app chat."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from leak_hunter import copilot as cp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ws(real_result, tmp_path_factory):
    from leak_hunter.investigation import Workspace
    from leak_hunter.pipeline import save_outputs
    out = tmp_path_factory.mktemp("cop")
    save_outputs(real_result, out)
    return Workspace.load(out)


def case_of(ws, t):
    return ws.incidents[ws.incidents.incident_type == t].incident_id.iloc[0]


TYPES = ["DUPLICATE_INVOICE", "QUANTITY_DISCREPANCY", "PO_PRICE_MISMATCH", "HISTORICAL_PRICE_SPIKE",
         "UNUSUAL_TRANSACTION"]


@pytest.mark.parametrize("t", TYPES)
def test_offline_answers_are_grounded_for_every_case_type(ws, t):
    c = cp.case_context(ws, case_of(ws, t))
    json.dumps(c, default=str)                                   # serializable payload
    for q in list(cp.QUICK_PROMPTS.values()) + ["next steps?", "supplier profile", "linked cases", "timeline",
                                                 "why this rank?", "is this fraud?", "hello"]:
        r = cp.ask(q, c)
        assert r.mode == "offline" and r.text
        assert not r.citations["unverified_ids"] and not r.citations["unverified_amounts"], (q, r.citations)


def test_summary_cites_exposure_and_case(ws):
    i = case_of(ws, "DUPLICATE_INVOICE")
    c = cp.case_context(ws, i)
    txt = cp.ask(cp.QUICK_PROMPTS["summary"], c).text
    assert i in txt and f"${c['case']['potential_exposure_usd']:,.2f}" in txt


def test_letter_is_neutral_and_cites_invoice_numbers(ws):
    i = case_of(ws, "PO_PRICE_MISMATCH")
    c = cp.case_context(ws, i)
    txt = cp.ask(cp.QUICK_PROMPTS["letter"], c).text
    assert "does not imply any wrongdoing" in txt and "fraud" not in txt.lower()
    assert c["findings"][0]["supplier_invoice_number"] in txt


def test_fraud_question_is_declined(ws):
    c = cp.case_context(ws, case_of(ws, "DUPLICATE_INVOICE"))
    assert "cannot determine fraud" in cp.ask("Is the supplier committing fraud?", c).text


def test_portfolio_answers(ws):
    pc = cp.portfolio_context(ws, ws.incidents)
    for q in cp.OVERVIEW_PROMPTS.values():
        r = cp.ask(q, pc)
        assert r.text and not r.citations["unverified_ids"] and not r.citations["unverified_amounts"]


def test_citation_check_flags_invented_ids_and_amounts(ws):
    c = cp.case_context(ws, case_of(ws, "PO_PRICE_MISMATCH"))
    real_line = c["findings"][0]["invoice_line_id"]
    v = cp.verify_citations(f"Line {real_line} and INV-999999-01 overcharged by $123,456.78.", c)
    assert v["unverified_ids"] == ["INV-999999-01"] and v["unverified_amounts"] == [123456.78]
    assert real_line not in v["unverified_ids"]


# ------------------------------------------------------------------ providers (mocked HTTP)
class FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._p


@pytest.mark.parametrize("provider,env,payload", [
    ("openai", "OPENAI_API_KEY", {"choices": [{"message": {"content": "Grounded answer about {id}."}}]}),
    ("groq", "GROQ_API_KEY", {"choices": [{"message": {"content": "Grounded answer about {id}."}}]}),
    ("gemini", "GEMINI_API_KEY", {"candidates": [{"content": {"parts": [{"text": "Grounded answer about {id}."}]}}]}),
])
def test_provider_call_and_prompt_grounding(ws, monkeypatch, provider, env, payload):
    import requests
    for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LEAK_HUNTER_LLM_PROVIDER"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(env, "test-key")
    i = case_of(ws, "DUPLICATE_INVOICE")
    c = cp.case_context(ws, i)
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers)
        p = {k: v for k, v in payload.items()}
        txt = __import__("json").dumps(p).replace("{id}", i)
        return FakeResp(200, __import__("json").loads(txt))

    monkeypatch.setattr(requests, "post", fake_post)
    cfg = cp.resolve_provider()
    assert cfg.name == provider and cfg.online
    r = cp.ask("Summarize", c, provider=cfg)
    assert r.mode == "llm" and i in r.text and not r.citations["unverified_ids"]
    body = json.dumps(seen["body"])
    assert "<evidence>" in body and i in body and "Never call anything fraud" in body   # grounded system prompt
    assert "test-key" not in seen["url"]                                                # key never in the URL


@pytest.mark.parametrize("failure", ["http500", "timeout", "malformed"])
def test_provider_failure_falls_back_to_offline(ws, monkeypatch, failure):
    import requests
    c = cp.case_context(ws, case_of(ws, "QUANTITY_DISCREPANCY"))

    def fake_post(*a, **k):
        if failure == "timeout":
            raise requests.Timeout("timed out")
        if failure == "http500":
            return FakeResp(500, {"error": "server"})
        return FakeResp(200, {"unexpected": True})

    monkeypatch.setattr(requests, "post", fake_post)
    r = cp.ask(cp.QUICK_PROMPTS["summary"], c, provider=cp.ProviderConfig("openai", "gpt-4o-mini", "k"))
    assert r.mode == "offline-fallback" and r.error and r.text
    assert c["case"]["case_id"] in r.text


def test_offline_switch_and_no_keys(monkeypatch):
    for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    assert not cp.resolve_provider().online
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert cp.resolve_provider().name == "groq"
    assert not cp.resolve_provider(force_offline=True).online
    monkeypatch.setenv("LEAK_HUNTER_COPILOT_OFFLINE", "true")
    assert not cp.resolve_provider().online


def test_provider_preference_and_secrets_mapping(monkeypatch):
    for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LEAK_HUNTER_LLM_PROVIDER"]:
        monkeypatch.delenv(k, raising=False)
    secrets = {"OPENAI_API_KEY": "a", "GEMINI_API_KEY": "b", "LEAK_HUNTER_LLM_PROVIDER": "gemini",
               "GEMINI_MODEL": "gemini-x"}
    cfg = cp.resolve_provider(secrets)
    assert cfg.name == "gemini" and cfg.model == "gemini-x"


# ------------------------------------------------------------------ app
def test_app_copilot_quick_prompts_and_chat(ws, tmp_path, monkeypatch):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    for k in ["OPENAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LEAK_HUNTER_OUT_DIR", str(ws.out_dir))
    monkeypatch.setenv("LEAK_HUNTER_CASE_DB", str(tmp_path / "a.sqlite"))
    at = st_testing.AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
    from leak_hunter.copilot import OVERVIEW_PROMPTS, QUICK_PROMPTS
    [g for g in at.button_group if g.key == "qp_overview"][0].set_value(OVERVIEW_PROMPTS["money"]).run()
    assert not at.exception and len(at.chat_message) == 2
    at.sidebar.radio(key="page").set_value("Case file").run()
    case_id = at.session_state["case_id"]
    for k in ["summary", "evidence", "letter"]:
        [g for g in at.button_group if g.key == f"qp_{case_id}"][0].set_value(QUICK_PROMPTS[k]).run()
        assert not at.exception
    at.chat_input[-1].set_value("What are the next steps?").run()
    assert not at.exception
    assert len(at.chat_message) == 8
    assert any("all found in the evidence" in c.value for c in at.caption)
