"""Stream LLM output via MelleaSession with chunking, quick checks, and sampling strategies.

Provides ``stream_instruct()``, a streaming counterpart to ``MelleaSession.instruct()``
that emits a rich event stream covering chunks, quick-check results, retries, and
completion.  The function is registered as a powerup on ``MelleaSession``; import this
module to activate it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mellea.core.backend import Backend
from mellea.core.base import ModelOutputThunk
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import avalidate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.sampling.base import BaseSamplingStrategy

from mellea_partial.chunking import (
    ChunkingMode,
    ChunkingStrategy,
    ChunkRepair,
    RegexChunking,
)

if TYPE_CHECKING:
    from mellea.stdlib.session import MelleaSession


# ─── Event types ────────────────────────────────────────────────────────────────


@dataclass
class StreamEvent:
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChunkEvent(StreamEvent):
    """Emitted when a validated chunk is ready for consumption."""

    text: str = ""
    chunk_index: int = 0
    attempt: int = 0


@dataclass
class QuickCheckEvent(StreamEvent):
    """Emitted after a quick check runs on a single chunk."""

    chunk_index: int = 0
    attempt: int = 0
    passed: bool = True
    results: list[ValidationResult] = field(default_factory=list)


@dataclass
class ChunkRepairedEvent(StreamEvent):
    """Emitted when ``on_chunk_failure`` successfully repairs a chunk."""

    chunk_index: int = 0
    attempt: int = 0
    original: str = ""
    repaired: str = ""


@dataclass
class StreamingDoneEvent(StreamEvent):
    """Emitted once all chunks for an attempt have been received."""

    attempt: int = 0
    full_text: str = ""


@dataclass
class FullValidationEvent(StreamEvent):
    """Emitted after full-output validation completes for an attempt."""

    attempt: int = 0
    passed: bool = True
    results: list[tuple[Requirement, ValidationResult]] = field(default_factory=list)


@dataclass
class RetryEvent(StreamEvent):
    """Emitted just before a new attempt begins."""

    attempt: int = 0
    reason: str = ""


@dataclass
class ToolCallEvent(StreamEvent):
    """Emitted when the model returns tool calls instead of (or alongside) text."""

    attempt: int = 0
    tool_calls: dict | None = None


@dataclass
class CompletedEvent(StreamEvent):
    """Emitted once processing is fully done (success or budget exhausted)."""

    success: bool = True
    full_text: str = ""
    attempts_used: int = 0


# ─── AttemptRecord ──────────────────────────────────────────────────────────────


@dataclass
class AttemptRecord:
    """Per-attempt bookkeeping collected in ``StreamInstructResult.attempts``."""

    attempt_index: int
    validated_chunks: list[str] = field(default_factory=list)
    quick_check_results: list[list[ValidationResult]] = field(default_factory=list)
    full_validation_results: list[tuple[Requirement, ValidationResult]] | None = None
    full_text: str = ""
    quick_checks_passed: bool = True
    full_validation_passed: bool = False
    context: object = None
    thunk: ModelOutputThunk | None = None
    tool_calls: dict | None = None


# ─── StreamInstructResult ───────────────────────────────────────────────────────

_SENTINEL: StreamEvent | None = None


@dataclass
class StreamInstructResult:
    """Result object returned immediately by ``stream_instruct()``.

    Consumers pull events via ``astream_events()`` or convenience text via
    ``astream_text()``.  Call ``acomplete()`` to block until the background
    task finishes, after which all fields are fully populated.
    """

    attempts: list[AttemptRecord] = field(default_factory=list)
    success: bool = False
    final_text: str = ""
    tool_calls: dict | None = None
    _event_queue: asyncio.Queue[StreamEvent | None] = field(
        default_factory=asyncio.Queue, repr=False
    )
    _task: asyncio.Task | None = field(default=None, repr=False)

    async def astream_events(self) -> AsyncIterator[StreamEvent]:
        """Yield every ``StreamEvent`` in arrival order until completion."""
        while True:
            event = await self._event_queue.get()
            if event is _SENTINEL:
                return
            yield event

    async def astream_text(self) -> AsyncIterator[str]:
        """Yield text from ``ChunkEvent``s belonging to the latest attempt only.

        ``RetryEvent``s update the tracked attempt so that chunks from
        superseded attempts are silently discarded.
        """
        current_attempt = 0
        async for event in self.astream_events():
            if isinstance(event, RetryEvent):
                current_attempt = event.attempt
            elif isinstance(event, ChunkEvent) and event.attempt == current_attempt:
                yield event.text

    async def acomplete(self) -> "StreamInstructResult":
        """Drain the event queue and return ``self`` with all fields populated."""
        if self._task is not None:
            await self._task
        return self

    @property
    def as_thunk(self) -> ModelOutputThunk:
        """Return a ``ModelOutputThunk`` wrapping ``final_text`` for mellea interop."""
        return ModelOutputThunk(self.final_text)


# ─── Internal helpers ────────────────────────────────────────────────────────────


def _cancel_thunk(thunk: ModelOutputThunk) -> None:
    if thunk._generate is not None:
        thunk._generate.cancel()


async def _validate_and_emit_chunk(
    chunk: str,
    chunk_index: int,
    attempt_idx: int,
    qc_reqs: list[Requirement | str],
    backend: Backend,
    attempt_rec: AttemptRecord,
    result: StreamInstructResult,
    quick_repair: ChunkRepair | None,
    ctx: object,
) -> tuple[bool, str]:
    """Validate *chunk* against quick-check requirements and emit the appropriate events.

    Returns ``(passed, chunk_text)`` where ``chunk_text`` may have been repaired.
    """
    if not chunk.strip():
        attempt_rec.quick_check_results.append([])
        return True, chunk

    val_ctx = SimpleContext()
    chunk_results = await avalidate(
        qc_reqs, val_ctx, backend, output=ModelOutputThunk(chunk)
    )
    attempt_rec.quick_check_results.append(chunk_results)
    passed = all(bool(r) for r in chunk_results)

    await result._event_queue.put(
        QuickCheckEvent(
            chunk_index=chunk_index,
            attempt=attempt_idx,
            passed=passed,
            results=chunk_results,
        )
    )

    if not passed:
        if quick_repair is not None:
            should_continue, repaired = await quick_repair(chunk, ctx, qc_reqs, chunk_results)
            if should_continue:
                await result._event_queue.put(
                    ChunkRepairedEvent(
                        chunk_index=chunk_index,
                        attempt=attempt_idx,
                        original=chunk,
                        repaired=repaired,
                    )
                )
                return True, repaired
        return False, chunk

    return True, chunk


# ─── Core _run loop ──────────────────────────────────────────────────────────────


async def _run(
    result: StreamInstructResult,
    instruction: Instruction,
    backend: Backend,
    strategy: BaseSamplingStrategy | None,
    qc_reqs: list[Requirement | str],
    chunking: ChunkingMode | ChunkingStrategy,
    quick_repair: ChunkRepair | None,
    model_options: dict | None,
    initial_context: object,
    session: "MelleaSession",
    tool_calls: bool = False,
) -> None:
    """Background coroutine that drives streaming, chunking, quick checks, and retries."""
    loop_budget = strategy.loop_budget if strategy is not None else 1
    chunker = RegexChunking(chunking) if isinstance(chunking, ChunkingMode) else chunking

    # Determine requirements for full-output validation.
    if strategy is not None:
        if strategy.requirements is not None:
            reqs: list[Requirement] = list(strategy.requirements)
        else:
            reqs = [
                Requirement(r) if isinstance(r, str) else r
                for r in (instruction.requirements or [])
            ]
    else:
        reqs = []

    # Bookkeeping for strategy.repair() / select_from_failure().
    sampled_results: list[ModelOutputThunk] = []
    sampled_scores: list[list[tuple[Requirement, ValidationResult]]] = []
    sampled_actions: list[Instruction] = []

    next_action: Instruction = deepcopy(instruction)
    next_context = initial_context

    try:
        for attempt_idx in range(loop_budget):
            attempt_rec = AttemptRecord(attempt_index=attempt_idx)
            result.attempts.append(attempt_rec)

            # ── 1. Generate ──────────────────────────────────────────────────
            thunk, result_ctx = await backend.generate_from_context(
                next_action, ctx=next_context, model_options=model_options,
                tool_calls=tool_calls,
            )
            attempt_rec.thunk = thunk
            attempt_rec.context = result_ctx

            # ── 2. Stream chunks with quick checks ──────────────────────────
            buffer = ""
            chunk_index = 0
            quick_check_failed = False

            while not thunk.is_computed():
                delta = await thunk.astream()
                attempt_rec.full_text += delta
                buffer += delta
                parts = chunker.split(buffer)
                buffer = parts[-1]

                for chunk in parts[:-1]:
                    if qc_reqs:
                        passed, chunk = await _validate_and_emit_chunk(
                            chunk, chunk_index, attempt_idx, qc_reqs,
                            backend, attempt_rec, result, quick_repair, next_context,
                        )
                        if not passed:
                            quick_check_failed = True
                            _cancel_thunk(thunk)
                            break
                    attempt_rec.validated_chunks.append(chunk)
                    await result._event_queue.put(
                        ChunkEvent(text=chunk, chunk_index=chunk_index, attempt=attempt_idx)
                    )
                    chunk_index += 1

                if quick_check_failed:
                    break

            if not quick_check_failed:
                # Process any remaining buffer using the final full text.
                final_parts = chunker.split(attempt_rec.full_text)
                already_done = len(attempt_rec.validated_chunks)
                for i in range(already_done, len(final_parts)):
                    chunk = final_parts[i]
                    if qc_reqs:
                        passed, chunk = await _validate_and_emit_chunk(
                            chunk, chunk_index, attempt_idx, qc_reqs,
                            backend, attempt_rec, result, quick_repair, next_context,
                        )
                        if not passed:
                            quick_check_failed = True
                            break
                    attempt_rec.validated_chunks.append(chunk)
                    await result._event_queue.put(
                        ChunkEvent(text=chunk, chunk_index=chunk_index, attempt=attempt_idx)
                    )
                    chunk_index += 1

            # ── 3. Quick check failed → retry without repair ─────────────────
            if quick_check_failed:
                attempt_rec.quick_checks_passed = False
                if attempt_idx < loop_budget - 1:
                    await result._event_queue.put(
                        RetryEvent(attempt=attempt_idx + 1, reason="quick check failed")
                    )
                continue

            # ── 4. All chunks passed — emit StreamingDoneEvent ───────────────
            attempt_rec.quick_checks_passed = True
            attempt_rec.full_text = "".join(attempt_rec.validated_chunks)
            await result._event_queue.put(
                StreamingDoneEvent(attempt=attempt_idx, full_text=attempt_rec.full_text)
            )

            # ── 4b. Capture tool calls from the thunk if present ───────────
            if thunk.tool_calls:
                attempt_rec.tool_calls = thunk.tool_calls
                result.tool_calls = thunk.tool_calls
                await result._event_queue.put(
                    ToolCallEvent(attempt=attempt_idx, tool_calls=thunk.tool_calls)
                )

            # ── 5. Full-output validation ────────────────────────────────────
            if reqs:
                output_thunk = ModelOutputThunk(attempt_rec.full_text)
                val_ctx = SimpleContext()
                val_results = await avalidate(
                    reqs, val_ctx, backend,
                    output=output_thunk, model_options=model_options,
                )
                scores: list[tuple[Requirement, ValidationResult]] = list(zip(reqs, val_results))
                attempt_rec.full_validation_results = scores
                all_passed = all(bool(s[1]) for s in scores)
                attempt_rec.full_validation_passed = all_passed

                await result._event_queue.put(
                    FullValidationEvent(attempt=attempt_idx, passed=all_passed, results=scores)
                )

                sampled_results.append(output_thunk)
                sampled_scores.append(scores)
                sampled_actions.append(next_action)

                # ── 6. Passed → done ─────────────────────────────────────────
                if all_passed:
                    result.final_text = attempt_rec.full_text
                    result.success = True
                    session.ctx = result_ctx
                    await result._event_queue.put(
                        CompletedEvent(
                            success=True,
                            full_text=result.final_text,
                            attempts_used=attempt_idx + 1,
                        )
                    )
                    return

                # ── 7. Failed → repair and retry ─────────────────────────────
                if attempt_idx < loop_budget - 1:
                    next_action, next_context = strategy.repair(  # type: ignore[union-attr]
                        next_context,
                        result_ctx,
                        sampled_actions,
                        sampled_results,
                        sampled_scores,
                    )
                    await result._event_queue.put(
                        RetryEvent(
                            attempt=attempt_idx + 1,
                            reason="full validation failed",
                        )
                    )

            else:
                # No requirements — single-attempt success.
                attempt_rec.full_validation_passed = True
                result.final_text = attempt_rec.full_text
                result.success = True
                session.ctx = result_ctx
                await result._event_queue.put(
                    CompletedEvent(
                        success=True,
                        full_text=result.final_text,
                        attempts_used=attempt_idx + 1,
                    )
                )
                return

        # ── Budget exhausted without success ─────────────────────────────────
        if sampled_results:
            best_idx = strategy.select_from_failure(  # type: ignore[union-attr]
                sampled_actions, sampled_results, sampled_scores
            )
            result.final_text = await sampled_results[best_idx].avalue()
        elif result.attempts:
            # All failures were quick-check failures — pick the last partial text.
            result.final_text = result.attempts[-1].full_text

        result.success = False
        # Update session context to the last result context if available.
        if result.attempts and result.attempts[-1].context is not None:
            session.ctx = result.attempts[-1].context  # type: ignore[assignment]

        await result._event_queue.put(
            CompletedEvent(
                success=False,
                full_text=result.final_text,
                attempts_used=loop_budget,
            )
        )

    finally:
        await result._event_queue.put(_SENTINEL)


# ─── Public API ──────────────────────────────────────────────────────────────────


async def stream_instruct(
    session: "MelleaSession",
    description: str,
    *,
    requirements: list[Requirement | str] | None = None,
    quick_check_requirements: list[Requirement | str] | None = None,
    icl_examples=None,
    grounding_context=None,
    user_variables=None,
    prefix=None,
    output_prefix=None,
    images=None,
    strategy: BaseSamplingStrategy | None = None,
    chunking: ChunkingMode | ChunkingStrategy = ChunkingMode.SENTENCE,
    quick_repair: ChunkRepair | None = None,
    model_options: dict | None = None,
    tool_calls: bool = False,
) -> StreamInstructResult:
    """Stream LLM output with per-chunk quick checks and optional sampling strategies.

    Mirrors ``MelleaSession.instruct()``'s parameter surface while returning a
    ``StreamInstructResult`` that exposes a rich event stream.  The background
    generation task starts immediately; the caller can consume events via
    ``async for event in result.astream_events()`` or convenience text via
    ``async for text in result.astream_text()``.

    Args:
        session: The active ``MelleaSession`` (``session.ctx`` is updated on completion).
        description: Instruction description forwarded to ``Instruction``.
        requirements: Full-output requirements validated after streaming finishes.
        quick_check_requirements: Per-chunk requirements validated during streaming.
        icl_examples: In-context learning examples.
        grounding_context: Grounding context dict.
        user_variables: Jinja placeholder values.
        prefix: Instruction prefix.
        output_prefix: Output generation prefix.
        images: Images passed to the instruction.
        strategy: Sampling strategy for retry/repair loops.  ``None`` means a single
            attempt with no full-output validation.
        chunking: How to split the stream into chunks (ChunkingMode enum or ChunkingStrategy).
        quick_repair: Optional callback invoked when a chunk fails quick checks.
        model_options: Extra model options forwarded to the backend.
        tool_calls: If ``True``, enable tool calling on the backend so the model
            can request tool invocations.  Tool call results are exposed via
            ``ToolCallEvent`` in the event stream and ``result.tool_calls``.

    Returns:
        A ``StreamInstructResult`` with a running background task.
    """
    instruction = Instruction(
        description=description,
        requirements=requirements or [],
        icl_examples=icl_examples or [],
        grounding_context=grounding_context or {},
        user_variables=user_variables,
        prefix=prefix,
        output_prefix=output_prefix,
        images=images,
    )

    qc_reqs: list[Requirement | str] = quick_check_requirements or []
    sr = StreamInstructResult()
    sr._task = asyncio.create_task(
        _run(
            result=sr,
            instruction=instruction,
            backend=session.backend,
            strategy=strategy,
            qc_reqs=qc_reqs,
            chunking=chunking,
            quick_repair=quick_repair,
            model_options=model_options,
            initial_context=session.ctx,
            session=session,
            tool_calls=tool_calls,
        )
    )
    return sr


# ─── MelleaSession powerup ───────────────────────────────────────────────────────


class _StreamInstructPowerup:
    stream_instruct = stream_instruct


from mellea.stdlib.session import MelleaSession  # noqa: E402

MelleaSession.powerup(_StreamInstructPowerup)
