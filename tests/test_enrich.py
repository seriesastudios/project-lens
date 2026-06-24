"""Quick-add LLM enrich: best-effort cleanup of a raw entry into
content/deadline/priority, with graceful fallback when the model misbehaves."""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

from app.engine import brain


def fake_client(content):
    """A stand-in AsyncOpenAI whose completion returns `content`. Pass an
    Exception instance to simulate the model being unreachable."""
    async def create(*args, **kwargs):
        if isinstance(content, Exception):
            raise content
        msg = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def enrich(raw, reply, monkeypatch):
    monkeypatch.setattr(brain, "client", fake_client(reply))
    return asyncio.run(brain.enrich_quick_task(raw))


def test_splits_title_and_deadline(monkeypatch):
    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
    out = enrich("Send booking proposal tomorrow",
                 '{"title": "Send booking proposal", "deadline": "tomorrow", "priority": null}',
                 monkeypatch)
    assert out["content"] == "Send booking proposal"
    assert out["target_date"] == tomorrow
    assert "priority" not in out


def test_extracts_priority(monkeypatch):
    out = enrich("call vendor urgent",
                 '{"title": "Call vendor", "deadline": null, "priority": "high"}',
                 monkeypatch)
    assert out["content"] == "Call vendor"
    assert out["priority"] == "high"
    assert "target_date" not in out


def test_unchanged_title_is_omitted(monkeypatch):
    out = enrich("Buy stamps",
                 '{"title": "Buy stamps", "deadline": null, "priority": null}',
                 monkeypatch)
    assert out == {}  # nothing to change → leave the task as typed


def test_tolerates_prose_around_json(monkeypatch):
    out = enrich("water plants today",
                 'Sure! Here you go: {"title": "Water plants", "deadline": "today"} hope that helps',
                 monkeypatch)
    assert out["content"] == "Water plants"
    assert out["target_date"] == datetime.now().date().isoformat()


def test_unparseable_deadline_is_dropped_others_kept(monkeypatch):
    out = enrich("ship it whenevs",
                 '{"title": "Ship it", "deadline": "whenevs", "priority": "low"}',
                 monkeypatch)
    assert out["content"] == "Ship it"
    assert out["priority"] == "low"
    assert "target_date" not in out  # normalize_deadline rejected the phrase


def test_garbage_output_falls_back_to_empty(monkeypatch):
    assert enrich("anything", "I cannot help with that.", monkeypatch) == {}


def test_model_exception_falls_back_to_empty(monkeypatch):
    assert enrich("anything", RuntimeError("LM Studio offline"), monkeypatch) == {}


def test_blank_input_short_circuits():
    assert asyncio.run(brain.enrich_quick_task("   ")) == {}
