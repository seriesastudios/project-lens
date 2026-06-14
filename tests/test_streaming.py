"""Tests for the streaming conversation loop: protocol events arrive in the
right order, optimistic tokens get retracted on leaks, and the non-streaming
wrapper still returns the final reply. The LLM is mocked at brain.client."""
import asyncio
import json
from types import SimpleNamespace

from app.database import models
from app.engine import brain


def chunk(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def tool_delta(index, name=None, arguments=None, call_id=None):
    return SimpleNamespace(
        index=index, id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments))


def fake_client(chunk_lists):
    """A stand-in for brain.client whose create() returns each chunk list in
    turn as an async stream (one list per expected LLM call)."""
    calls = iter(chunk_lists)

    async def create(**kwargs):
        assert kwargs.get("stream") is True
        chunks = next(calls)

        async def stream():
            for c in chunks:
                yield c
        return stream()

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def run_events(monkeypatch, chunk_lists, user_text):
    monkeypatch.setattr(brain, "client", fake_client(chunk_lists))
    brain.reset_session()

    async def collect():
        return [e async for e in brain.process_user_input_events(user_text)]
    events = asyncio.run(collect())
    brain.reset_session()
    return events


def test_action_turn_emits_status_action_lens_done(monkeypatch):
    args = json.dumps({"tasks": [{"content": "Order business cards"}]})
    stream = [
        chunk(tool_calls=[tool_delta(0, name="capture_tasks", call_id="c1")]),
        chunk(tool_calls=[tool_delta(0, arguments=args)]),
    ]
    events = run_events(monkeypatch, [stream], "need to order business cards")

    types = [e["type"] for e in events]
    assert types == ["status", "action", "lens", "done"]
    assert events[1]["text"] == "Captured **Order business cards**."
    assert events[-1]["reply"] == "Captured **Order business cards**."
    assert models.find_node_by_content("Order business cards") is not None


def test_plain_chat_turn_streams_tokens(monkeypatch):
    stream = [chunk(content="Rough "), chunk(content="day — "), chunk(content="hang in there.")]
    events = run_events(monkeypatch, [stream], "ugh, today was exhausting")

    types = [e["type"] for e in events]
    assert types == ["status", "token", "token", "token", "done"]
    assert events[-1]["reply"] == "Rough day — hang in there."


def test_leaked_call_retracts_streamed_tokens_and_executes(monkeypatch):
    # The model writes the tool call as prose; tokens stream optimistically,
    # then a replace event retracts them and the salvaged call runs. Navigation
    # then speaks via a stage-2 call (no templated 'action' event).
    leak = 'open_view {"view": "today"}'
    stream = [chunk(content=leak[:10]), chunk(content=leak[10:])]
    speak = [chunk(content="Here's "), chunk(content="your day.")]
    events = run_events(monkeypatch, [stream, speak], "show me today")

    types = [e["type"] for e in events]
    assert types == ["status", "token", "token", "replace", "lens", "token", "token", "done"]
    assert events[3] == {"type": "replace", "text": ""}
    assert events[-1]["reply"] == "Here's your day."


def test_tokens_before_real_tool_call_are_retracted(monkeypatch):
    # Content spill before a genuine tool call is reasoning, not a reply.
    # The navigation then gets a real spoken reply from the stage-2 call.
    args = json.dumps({"view": "projects"})
    stream = [
        chunk(content="Let me open that. "),
        chunk(tool_calls=[tool_delta(0, name="open_view", call_id="c1", arguments=args)]),
    ]
    speak = [chunk(content="You've got "), chunk(content="9 projects going.")]
    events = run_events(monkeypatch, [stream, speak], "show my projects")
    types = [e["type"] for e in events]
    assert types == ["status", "token", "replace", "lens", "token", "token", "done"]
    assert events[-1]["reply"] == "You've got 9 projects going."


def test_engine_down_yields_done_with_warning(monkeypatch):
    async def create(**kwargs):
        raise ConnectionError("refused")
    monkeypatch.setattr(brain, "client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    brain.reset_session()

    async def collect():
        return [e async for e in brain.process_user_input_events("hello")]
    events = asyncio.run(collect())
    brain.reset_session()
    assert events[-1]["type"] == "done"
    assert "inference engine unreachable" in events[-1]["reply"]


def test_non_streaming_wrapper_returns_reply(monkeypatch):
    stream = [chunk(content="Just a chat reply.")]
    monkeypatch.setattr(brain, "client", fake_client([stream]))
    brain.reset_session()
    reply = asyncio.run(brain.process_user_input("hi"))
    brain.reset_session()
    assert reply == "Just a chat reply."
