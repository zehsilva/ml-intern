import asyncio
from types import SimpleNamespace

import httpx
import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.core.agent_loop import (
    LLMResult,
    _assistant_message_from_result,
    _call_llm_non_streaming,
    _call_llm_streaming,
    _strip_thinking_state_from_messages,
)


def test_assistant_message_from_result_keeps_content_and_tool_calls():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "date"}'},
    )
    result = LLMResult(
        content="working",
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
    )

    message = _assistant_message_from_result(result, tool_calls=[tool_call])

    assert message.content == "working"
    assert message.tool_calls == [tool_call]
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_strip_thinking_state_from_saved_messages():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "stale"},
                {"type": "text", "text": "done"},
            ],
            "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
            "reasoning_content": "stale",
            "provider_specific_fields": {
                "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
                "reasoning_content": "stale",
                "other": "kept",
            },
        }
    ]

    stripped = _strip_thinking_state_from_messages(messages)

    assert stripped == 5
    assert messages == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "provider_specific_fields": {"other": "kept"},
        }
    ]


@pytest.mark.asyncio
async def test_streaming_call_returns_wire_safe_result(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4.8:fal-ai"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/anthropic/claude-opus-4.8:fal-ai"},
    )

    assert result.content == "done"
    assert result.token_count == 3


@pytest.mark.asyncio
async def test_streaming_call_retries_read_timeout_before_output(monkeypatch):
    async def timeout_stream():
        raise httpx.ReadTimeout("Timeout on reading data from socket")
        yield  # pragma: no cover

    async def success_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=2))

    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return timeout_stream()
        return success_stream()

    events = []

    async def send_event(event):
        events.append(event)

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="MiniMaxAI/MiniMax-M3:novita"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)
    monkeypatch.setattr("agent.core.agent_loop.asyncio.sleep", fake_sleep)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/MiniMaxAI/MiniMax-M3:novita"},
    )

    assert attempts == 2
    assert sleeps == [5]
    assert result.content == "ok"
    assert [event.event_type for event in events] == [
        "tool_log",
        "assistant_chunk",
        "llm_call",
    ]


@pytest.mark.asyncio
async def test_streaming_call_does_not_retry_after_partial_output(monkeypatch):
    async def partial_timeout_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="partial", tool_calls=None),
                    finish_reason=None,
                )
            ],
        )
        raise httpx.ReadTimeout("Timeout on reading data from socket")

    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        return partial_timeout_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="MiniMaxAI/MiniMax-M3:novita"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    with pytest.raises(httpx.ReadTimeout):
        await _call_llm_streaming(
            session,
            messages=[Message(role="user", content="hi")],
            tools=[],
            llm_params={"model": "openai/MiniMaxAI/MiniMax-M3:novita"},
        )

    assert attempts == 1
    assert [event.event_type for event in events] == [
        "assistant_chunk",
        "assistant_stream_end",
        "llm_call",
    ]


@pytest.mark.asyncio
async def test_streaming_call_stops_retry_when_cancelled_during_delay(monkeypatch):
    async def timeout_stream():
        raise httpx.ReadTimeout("Timeout on reading data from socket")
        yield  # pragma: no cover

    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        return timeout_stream()

    events = []

    class CancellableSession:
        def __init__(self):
            self.config = SimpleNamespace(model_name="MiniMaxAI/MiniMax-M3:novita")
            self._cancelled = asyncio.Event()

        @property
        def is_cancelled(self):
            return self._cancelled.is_set()

        def cancel(self):
            self._cancelled.set()

        async def send_event(self, event):
            events.append(event)
            if event.event_type == "tool_log":
                self.cancel()

    session = CancellableSession()
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/MiniMaxAI/MiniMax-M3:novita"},
    )

    assert attempts == 1
    assert result.content is None
    assert session.is_cancelled is True
    assert [event.event_type for event in events] == ["tool_log"]


@pytest.mark.asyncio
async def test_non_streaming_call_flattens_provider_content_blocks(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[
                        {"type": "text", "text": "hello"},
                        {"type": "output_text", "text": "world"},
                    ],
                    tool_calls=None,
                    reasoning_content=None,
                    provider_specific_fields=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(total_tokens=4),
    )

    async def fake_acompletion(**_kwargs):
        return response

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/responses/gpt-5.6"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_non_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/responses/gpt-5.6"},
    )

    assert result.content == "hello\nworld"
    assistant_events = [e for e in events if e.event_type == "assistant_message"]
    assert assistant_events[-1].data == {"content": "hello\nworld"}


@pytest.mark.asyncio
async def test_streaming_call_flattens_provider_content_blocks(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=[{"type": "output_text", "text": "chunk"}],
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/responses/gpt-5.6"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/responses/gpt-5.6"},
    )

    assert result.content == "chunk"
    assert events[0].data == {"content": "chunk"}


@pytest.mark.asyncio
async def test_non_streaming_call_reads_responses_api_output_text(monkeypatch):
    response = SimpleNamespace(
        output_text="visible from response",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=None,
                    reasoning_content=None,
                    provider_specific_fields=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(total_tokens=4),
    )

    async def fake_acompletion(**_kwargs):
        return response

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/responses/gpt-5.5"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_non_streaming(
        session,
        messages=[Message(role="user", content="test")],
        tools=[],
        llm_params={"model": "openai/responses/gpt-5.5"},
    )

    assert result.content == "visible from response"
    assert [e for e in events if e.event_type == "assistant_message"][-1].data == {
        "content": "visible from response"
    }


@pytest.mark.asyncio
async def test_streaming_call_reads_responses_api_final_message(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    message=SimpleNamespace(content="final visible text"),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/responses/gpt-5.5"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="test")],
        tools=[],
        llm_params={"model": "openai/responses/gpt-5.5"},
    )

    assert result.content == "final visible text"
    assert events[0].data == {"content": "final visible text"}


@pytest.mark.asyncio
async def test_streaming_call_reads_raw_responses_output_text_delta(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(type="response.output_text.delta", delta="raw ")
        yield SimpleNamespace(type="response.output_text.delta", delta="delta")
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/responses/gpt-5.5"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="test")],
        tools=[],
        llm_params={"model": "openai/responses/gpt-5.5"},
    )

    assert result.content == "raw delta"
    assert [
        event.data for event in events if event.event_type == "assistant_chunk"
    ] == [
        {"content": "raw "},
        {"content": "delta"},
    ]


@pytest.mark.asyncio
async def test_streaming_call_reads_raw_responses_completed_when_no_delta(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output_text="completed text"),
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="openai/responses/gpt-5.5"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="test")],
        tools=[],
        llm_params={"model": "openai/responses/gpt-5.5"},
    )

    assert result.content == "completed text"
    assert events[0].data == {"content": "completed text"}
