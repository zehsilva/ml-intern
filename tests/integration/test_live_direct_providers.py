"""Opt-in live checks for direct LiteLLM providers.

Set ``ML_INTERN_LIVE_LLM_TESTS=1`` and the provider API key to run these tests.
They intentionally make paid network calls and are skipped by default.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from litellm import Message

from agent.core.agent_loop import _call_llm_non_streaming
from agent.core.llm_params import _resolve_llm_params


LIVE_TESTS_ENABLED = os.environ.get("ML_INTERN_LIVE_LLM_TESTS") == "1"


def _skip_unless_live_provider(env_var: str) -> None:
    if not LIVE_TESTS_ENABLED:
        pytest.skip("set ML_INTERN_LIVE_LLM_TESTS=1 to run paid live LLM tests")
    if not os.environ.get(env_var):
        pytest.skip(f"set {env_var} to run this live provider test")


def _session(model_name: str):
    events = []

    async def send_event(event):
        events.append(event)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        is_cancelled=False,
        send_event=send_event,
        events=events,
    )


@pytest.mark.asyncio
async def test_live_openai_responses_direct_route_returns_visible_text():
    _skip_unless_live_provider("OPENAI_API_KEY")
    model = (
        os.environ.get("ML_INTERN_LIVE_OPENAI_RESPONSES_MODEL")
        or "openai/responses/gpt-5.6"
    )
    session = _session(model)
    result = await _call_llm_non_streaming(
        session,
        messages=[Message(role="user", content="Reply with exactly: DIRECT_OPENAI_OK")],
        tools=[],
        llm_params=_resolve_llm_params(model, session_hf_token="must-not-be-used"),
    )

    assert result.content
    assert "DIRECT_OPENAI_OK" in result.content
    assert any(
        event.event_type == "assistant_message" and event.data.get("content")
        for event in session.events
    )
