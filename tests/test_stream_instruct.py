"""Tests for stream_instruct().

Uses a custom StreamingMockBackend to produce proper streaming ModelOutputThunks
without requiring a real LLM.
"""

from __future__ import annotations

import asyncio
import re

import pytest

from mellea.core.backend import Backend
from mellea.core.base import GenerateType, ModelOutputThunk
from mellea.core.requirement import Requirement
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements.requirement import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.sampling.base import RepairTemplateStrategy
from mellea.stdlib.session import MelleaSession

from mellea_partial import (  # importing activates the powerup  # noqa: F401
    ChunkEvent,
    ChunkRepairedEvent,
    ChunkingMode,
    ChunkingStrategy,
    CompletedEvent,
    FullValidationEvent,
    QuickCheckEvent,
    RetryEvent,
    StreamingDoneEvent,
    StreamInstructResult,
    ToolCallEvent,
)


# ─── StreamingMockBackend ────────────────────────────────────────────────────


class StreamingMockBackend(Backend):
    """Mock backend that creates real streaming ModelOutputThunks for testing.

    Each call to _generate_from_context pops the next response from the list
    and streams it as a single chunk followed by the sentinel.
    """

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    async def generate_from_raw(self, actions, ctx, **kwargs):
        raise NotImplementedError("StreamingMockBackend does not support generate_from_raw")

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
        # Use a computed MOT as the action — hits the CBlock branch in astream()
        # and sets parsed_repr = mot.value without calling _parse.
        mot._action = ModelOutputThunk(response)

        async def _process(m: ModelOutputThunk, chunk_str: str) -> None:
            if m._underlying_value is None:
                m._underlying_value = ""
            m._underlying_value += chunk_str

        async def _post_process(m: ModelOutputThunk) -> None:
            pass

        mot._process = _process
        mot._post_process = _post_process

        # Stream the entire response as one item, then the sentinel.
        async def _generate_task() -> None:
            await mot._async_queue.put(response)
            await mot._async_queue.put(None)

        mot._generate = asyncio.create_task(_generate_task())

        new_ctx = ctx.add(action).add(mot)
        return mot, new_ctx


# ─── Helpers ─────────────────────────────────────────────────────────────────


def session_with(responses: list[str]) -> MelleaSession:
    return MelleaSession(StreamingMockBackend(responses), SimpleContext())


def contains_word_req(word: str) -> Requirement:
    return Requirement(
        f"Response must contain '{word}'.",
        simple_validate(lambda x, w=word: w.lower() in x.lower()),
        check_only=True,
    )


def no_digit_req() -> Requirement:
    return Requirement(
        "Chunk must not contain a digit.",
        simple_validate(lambda x: not any(c.isdigit() for c in x)),
        check_only=True,
    )


def short_chunk_req() -> Requirement:
    return Requirement(
        "Chunk must be under 100 characters.",
        simple_validate(lambda x: len(x.strip()) < 100),
        check_only=True,
    )


async def collect_events(result: StreamInstructResult) -> list:
    events = []
    async for event in result.astream_events():
        events.append(event)
    return events


# ─── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_basic_streaming_no_strategy():
    """No strategy, no requirements: single attempt succeeds."""
    m = session_with(["Hello world. This is nice."])
    result = await m.stream_instruct("Say something.")
    await result.acomplete()

    assert result.success is True
    assert result.final_text != ""
    assert len(result.attempts) == 1
    assert result.attempts[0].full_validation_passed is True
    assert len(result.attempts[0].validated_chunks) >= 1


@pytest.mark.asyncio
async def test_astream_events_order():
    """Events arrive in the expected order: ChunkEvents → StreamingDoneEvent → CompletedEvent."""
    m = session_with(["Hello world. Foo bar."])
    result = await m.stream_instruct("Say something.")

    events = await collect_events(result)

    assert len(events) > 0
    assert isinstance(events[-1], CompletedEvent)

    done_idx = next(i for i, e in enumerate(events) if isinstance(e, StreamingDoneEvent))
    completed_idx = next(i for i, e in enumerate(events) if isinstance(e, CompletedEvent))
    assert done_idx < completed_idx

    chunk_indices = [i for i, e in enumerate(events) if isinstance(e, ChunkEvent)]
    assert len(chunk_indices) >= 1
    assert all(ci < done_idx for ci in chunk_indices)


@pytest.mark.asyncio
async def test_quick_check_pass():
    """Quick-check requirement that always passes: QuickCheckEvent(passed=True) per chunk."""
    m = session_with(["Hello world. Foo bar."])
    result = await m.stream_instruct(
        "Say something.",
        quick_check_requirements=[short_chunk_req()],
    )

    events = await collect_events(result)

    qc_events = [e for e in events if isinstance(e, QuickCheckEvent)]
    assert len(qc_events) > 0
    assert all(e.passed for e in qc_events)

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is True


@pytest.mark.asyncio
async def test_quick_check_fail_stops_streaming():
    """A failing quick-check stops streaming and emits CompletedEvent(success=False)."""
    m = session_with(["Hello 1 world. Foo bar."])
    result = await m.stream_instruct(
        "Say something.",
        quick_check_requirements=[no_digit_req()],
    )

    events = await collect_events(result)

    qc_events = [e for e in events if isinstance(e, QuickCheckEvent)]
    assert any(not e.passed for e in qc_events), "Expected at least one failed QuickCheckEvent"

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is False

    # Streaming stopped early — no StreamingDoneEvent
    assert not any(isinstance(e, StreamingDoneEvent) for e in events)


@pytest.mark.asyncio
async def test_chunk_repair_callback():
    """repair repairs a failing chunk; ChunkRepairedEvent and fixed ChunkEvent are emitted."""
    async def strip_digits(chunk, ctx, qc_reqs, results):
        return (True, re.sub(r"\d+", "", chunk))

    m = session_with(["Hello 1 world. Foo bar."])
    result = await m.stream_instruct(
        "Say something.",
        quick_check_requirements=[no_digit_req()],
        quick_repair=strip_digits,
    )

    events = await collect_events(result)

    repaired_events = [e for e in events if isinstance(e, ChunkRepairedEvent)]
    assert len(repaired_events) > 0

    rep = repaired_events[0]
    assert "1" in rep.original
    assert "1" not in rep.repaired

    chunk_events = [e for e in events if isinstance(e, ChunkEvent)]
    assert any(e.text == rep.repaired for e in chunk_events)

    # Repair allowed streaming to continue — CompletedEvent should be success
    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is True


@pytest.mark.asyncio
async def test_full_validation_retry_rejection():
    """RejectionSamplingStrategy: first attempt fails full validation, second succeeds."""
    m = session_with([
        "Hello world. Foo bar.",       # attempt 0: no "three" → fails
        "Three blind mice. And more.", # attempt 1: has "three" → passes
    ])
    strategy = RejectionSamplingStrategy(loop_budget=2)

    result = await m.stream_instruct(
        "Say something.",
        requirements=[contains_word_req("three")],
        strategy=strategy,
    )

    events = await collect_events(result)

    retry_events = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].reason == "full validation failed"

    fv_events = [e for e in events if isinstance(e, FullValidationEvent)]
    assert fv_events[0].passed is False
    assert fv_events[1].passed is True

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is True
    assert completed.attempts_used == 2


@pytest.mark.asyncio
async def test_full_validation_retry_repair_template():
    """RepairTemplateStrategy: first attempt fails, second (with repair hint) succeeds."""
    m = session_with([
        "Hello world. Foo bar.",
        "Three blind mice. And more.",
    ])
    strategy = RepairTemplateStrategy(loop_budget=2)

    result = await m.stream_instruct(
        "Say something.",
        requirements=[contains_word_req("three")],
        strategy=strategy,
    )

    events = await collect_events(result)

    retry_events = [e for e in events if isinstance(e, RetryEvent)]
    assert len(retry_events) == 1
    assert retry_events[0].reason == "full validation failed"

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is True


@pytest.mark.asyncio
async def test_budget_exhausted():
    """All attempts fail full validation: CompletedEvent(success=False)."""
    m = session_with(["Hello world.", "Hello world."])
    strategy = RejectionSamplingStrategy(loop_budget=2)

    result = await m.stream_instruct(
        "Say something.",
        requirements=[contains_word_req("three")],
        strategy=strategy,
    )

    events = await collect_events(result)

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is False
    assert completed.attempts_used == 2


@pytest.mark.asyncio
async def test_as_thunk_interop():
    """result.as_thunk returns a ModelOutputThunk wrapping final_text."""
    m = session_with(["Hello world."])
    result = await m.stream_instruct("Say something.")
    await result.acomplete()

    thunk = result.as_thunk
    assert isinstance(thunk, ModelOutputThunk)
    assert thunk.value == result.final_text
    assert result.final_text != ""


@pytest.mark.asyncio
async def test_session_ctx_updated():
    """After successful completion, session.ctx is updated to include the new output."""
    m = session_with(["Hello world."])
    initial_ctx = m.ctx

    result = await m.stream_instruct("Say something.")
    await result.acomplete()

    assert result.success is True
    assert m.ctx is not initial_ctx


@pytest.mark.asyncio
async def test_astream_text_tracks_latest_attempt():
    """astream_text yields ChunkEvent text; RetryEvent correctly advances current_attempt."""
    m = session_with([
        "First attempt text. More here.",
        "Three success text. Done here.",
    ])
    strategy = RejectionSamplingStrategy(loop_budget=2)

    result = await m.stream_instruct(
        "Say something.",
        requirements=[contains_word_req("three")],
        strategy=strategy,
    )

    collected = []
    retry_count = 0
    async for event in result.astream_events():
        if isinstance(event, RetryEvent):
            retry_count += 1
        elif isinstance(event, ChunkEvent):
            collected.append(event.text)

    assert retry_count == 1

    # Chunks from attempt 0 (before RetryEvent) and attempt 1 are both visible
    combined = "".join(collected).lower()
    assert "first attempt" in combined
    assert "three success" in combined

    assert result.success is True


@pytest.mark.asyncio
async def test_chunking_modes():
    """WORD and PARAGRAPH chunking modes produce different split patterns."""
    # --- WORD mode ---
    m_word = session_with(["Hello world foo bar"])
    result_word = await m_word.stream_instruct(
        "Say words.",
        chunking=ChunkingMode.WORD,
    )
    await result_word.acomplete()

    word_chunks = [c for c in result_word.attempts[0].validated_chunks if c.strip()]
    assert len(word_chunks) >= 2
    # Each chunk is a single word — no internal whitespace
    assert all(" " not in c.strip() for c in word_chunks)

    # --- PARAGRAPH mode ---
    m_para = session_with(["First paragraph.\n\nSecond paragraph."])
    result_para = await m_para.stream_instruct(
        "Say paragraphs.",
        chunking=ChunkingMode.PARAGRAPH,
    )
    await result_para.acomplete()

    para_chunks = [c for c in result_para.attempts[0].validated_chunks if c.strip()]
    assert len(para_chunks) >= 2
    assert any("First paragraph" in c for c in para_chunks)
    assert any("Second paragraph" in c for c in para_chunks)


@pytest.mark.asyncio
async def test_chunk_repair_returns_false_stops_streaming():
    """repair returning (False, text) stops streaming; CompletedEvent(success=False)."""
    async def reject_chunk(chunk, ctx, qc_reqs, results):
        return (False, chunk)

    m = session_with(["Hello 1 world. Foo bar."])
    result = await m.stream_instruct(
        "Say something.",
        quick_check_requirements=[no_digit_req()],
        quick_repair=reject_chunk,
    )

    events = await collect_events(result)

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is False

    # No ChunkRepairedEvent since repair returned False
    assert not any(isinstance(e, ChunkRepairedEvent) for e in events)
    # No StreamingDoneEvent since streaming was stopped
    assert not any(isinstance(e, StreamingDoneEvent) for e in events)


@pytest.mark.asyncio
async def test_custom_chunking_strategy():
    """A custom ChunkingStrategy subclass is accepted and used for splitting."""
    class CommaChunking(ChunkingStrategy):
        def split(self, text: str) -> list[str]:
            return text.split(",")

    m = session_with(["first,second,third"])
    result = await m.stream_instruct(
        "Say something.",
        chunking=CommaChunking(),
    )
    await result.acomplete()

    assert result.success is True
    chunks = [c for c in result.attempts[0].validated_chunks if c.strip()]
    assert len(chunks) >= 2
    assert any("first" in c for c in chunks)
    assert any("second" in c for c in chunks)


# ─── Tool call mock backend ───────────────────────────────────────────────────


class ToolCallMockBackend(Backend):
    """Mock backend that returns a response with tool_calls set on the thunk."""

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
            # Simulate tool calls being populated during post-processing
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


def session_with_tools(
    responses: list[str], tool_calls_data: dict | None = None,
) -> MelleaSession:
    return MelleaSession(
        ToolCallMockBackend(responses, tool_calls_data), SimpleContext(),
    )


# ─── Tool call tests ──────────────────────────────────────────────────────────


SAMPLE_TOOL_CALLS = {
    "get_weather": {
        "name": "get_weather",
        "arguments": '{"location": "San Francisco"}',
    },
}


@pytest.mark.asyncio
async def test_tool_calls_event_emitted():
    """When tool_calls=True and backend returns tool calls, a ToolCallEvent is emitted."""
    m = session_with_tools(["Checking the weather."], SAMPLE_TOOL_CALLS)
    result = await m.stream_instruct(
        "What's the weather?",
        tool_calls=True,
    )

    events = await collect_events(result)

    tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tc_events) == 1
    assert tc_events[0].tool_calls == SAMPLE_TOOL_CALLS

    completed = next(e for e in events if isinstance(e, CompletedEvent))
    assert completed.success is True


@pytest.mark.asyncio
async def test_tool_calls_on_result():
    """result.tool_calls is populated after completion when tools are called."""
    m = session_with_tools(["Checking the weather."], SAMPLE_TOOL_CALLS)
    result = await m.stream_instruct(
        "What's the weather?",
        tool_calls=True,
    )
    await result.acomplete()

    assert result.tool_calls == SAMPLE_TOOL_CALLS
    assert result.attempts[0].tool_calls == SAMPLE_TOOL_CALLS


@pytest.mark.asyncio
async def test_tool_calls_false_no_event():
    """When tool_calls=False (default), no ToolCallEvent is emitted even with tool data."""
    m = session_with_tools(["Hello world."], SAMPLE_TOOL_CALLS)
    result = await m.stream_instruct("Say something.")

    events = await collect_events(result)

    tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tc_events) == 0
    assert result.tool_calls is None


@pytest.mark.asyncio
async def test_tool_calls_no_tools_returned():
    """When tool_calls=True but backend returns no tool calls, no ToolCallEvent."""
    m = session_with_tools(["Just text response."], tool_calls_data=None)
    result = await m.stream_instruct(
        "Say something.",
        tool_calls=True,
    )

    events = await collect_events(result)

    tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tc_events) == 0
    assert result.tool_calls is None
