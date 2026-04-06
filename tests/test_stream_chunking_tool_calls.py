"""Tests for stream_with_chunking tool_calls support."""

from __future__ import annotations

import asyncio

import pytest

from mellea.core.backend import Backend
from mellea.core.base import GenerateType, ModelOutputThunk
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.components.instruction import Instruction

from mellea_partial.chunking import stream_with_chunking, StreamChunkingResult


# ─── Mock backend ───────────────────────────────────────────────────────────


class ToolCallStreamingMockBackend(Backend):
    """Mock backend that optionally populates tool_calls on the thunk."""

    def __init__(self, responses: list[str], tool_calls_data: dict | None = None):
        self.responses = list(responses)
        self.tool_calls_data = tool_calls_data

    async def generate_from_raw(self, actions, ctx, **kwargs):
        raise NotImplementedError

    async def _generate_from_context(
        self,
        action,
        ctx,
        *,
        format=None,
        model_options=None,
        tool_calls=False,
    ):
        response = self.responses.pop(0)

        mot = ModelOutputThunk(None)
        mot._generate_type = GenerateType.ASYNC
        mot._action = ModelOutputThunk(response)

        async def _process(m: ModelOutputThunk, chunk_str: str) -> None:
            if m._underlying_value is None:
                m._underlying_value = ""
            m._underlying_value += chunk_str

        async def _post_process(m: ModelOutputThunk) -> None:
            if tool_calls and self.tool_calls_data:
                m.tool_calls = self.tool_calls_data

        mot._process = _process
        mot._post_process = _post_process

        async def _generate_task() -> None:
            await mot._async_queue.put(response)
            await mot._async_queue.put(None)

        mot._generate = asyncio.create_task(_generate_task())

        new_ctx = ctx.add(action).add(mot)
        return mot, new_ctx


# ─── Tests ──────────────────────────────────────────────────────────────────

SAMPLE_TOOL_CALLS = {
    "get_weather": {
        "name": "get_weather",
        "arguments": '{"location": "NYC"}',
    },
}


@pytest.mark.asyncio
async def test_stream_with_chunking_tool_calls():
    """tool_calls=True populates result.tool_calls from the thunk."""
    backend = ToolCallStreamingMockBackend(["Hello world."], SAMPLE_TOOL_CALLS)
    ctx = SimpleContext()
    instruction = Instruction(description="Say something.")

    result = await stream_with_chunking(
        instruction, backend, ctx, tool_calls=True,
    )
    await result.acomplete()

    assert result.completed is True
    assert result.tool_calls == SAMPLE_TOOL_CALLS


@pytest.mark.asyncio
async def test_stream_with_chunking_no_tool_calls():
    """Default tool_calls=False leaves result.tool_calls as None."""
    backend = ToolCallStreamingMockBackend(["Hello world."], SAMPLE_TOOL_CALLS)
    ctx = SimpleContext()
    instruction = Instruction(description="Say something.")

    result = await stream_with_chunking(instruction, backend, ctx)
    await result.acomplete()

    assert result.completed is True
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_stream_with_chunking_tool_calls_none_from_backend():
    """tool_calls=True but backend returns no tool calls: result.tool_calls stays None."""
    backend = ToolCallStreamingMockBackend(["Hello world."], tool_calls_data=None)
    ctx = SimpleContext()
    instruction = Instruction(description="Say something.")

    result = await stream_with_chunking(
        instruction, backend, ctx, tool_calls=True,
    )
    await result.acomplete()

    assert result.completed is True
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_stream_with_chunking_text_and_tool_calls():
    """Both text content and tool_calls are captured."""
    backend = ToolCallStreamingMockBackend(
        ["The weather is sunny."], SAMPLE_TOOL_CALLS,
    )
    ctx = SimpleContext()
    instruction = Instruction(description="What's the weather?")

    result = await stream_with_chunking(
        instruction, backend, ctx, tool_calls=True,
    )

    # Consume the stream
    chunks = []
    async for chunk in result.astream():
        chunks.append(chunk)

    await result.acomplete()

    assert result.completed is True
    assert len(chunks) > 0
    assert "sunny" in result.full_text
    assert result.tool_calls == SAMPLE_TOOL_CALLS
