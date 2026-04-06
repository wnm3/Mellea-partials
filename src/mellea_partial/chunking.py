"""Stream LLM output with chunk-by-chunk validation.

Two-phase validation:
1. Quick checks: validated per-chunk during streaming (stops on failure)
2. Full output checks: validated once after complete output (instruction requirements)
"""

from __future__ import annotations

import abc
import asyncio
import enum
import re
from collections.abc import AsyncIterator, Callable, Awaitable
from dataclasses import dataclass, field

from mellea.core.backend import Backend
from mellea.core.base import ModelOutputThunk, Context
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import avalidate


@dataclass
class StreamChunkingResult:
    validated_chunks: list[str] = field(default_factory=list)
    failed_chunk: str | None = None
    quick_check_results: list[list[ValidationResult]] = field(default_factory=list)
    full_validation_results: list[ValidationResult] | None = None
    completed: bool = False
    full_text: str = ""
    tool_calls: dict | None = None
    _chunk_queue: asyncio.Queue[str | None] = field(
        default_factory=asyncio.Queue, repr=False
    )
    _task: asyncio.Task | None = field(default=None, repr=False)

    async def astream(self) -> AsyncIterator[str]:
        """Yield validated chunk strings as they become available.

        After iteration completes, all result fields (validated_chunks,
        failed_chunk, full_text, completed, etc.) are fully populated.
        """
        while True:
            item = await self._chunk_queue.get()
            if item is _SENTINEL:
                return
            yield item

    async def acomplete(self) -> StreamChunkingResult:
        """Wait for the background task to finish and return self."""
        if self._task is not None:
            await self._task
        return self

    @property
    def as_thunk(self) -> ModelOutputThunk:
        """Return a computed ModelOutputThunk wrapping the full validated text."""
        return ModelOutputThunk(self.full_text)


ChunkRepair = Callable[
    [str, Context, list[Requirement | str], list[ValidationResult]],
    Awaitable[tuple[bool, str]],
]


class ChunkingMode(enum.Enum):
    SENTENCE = "sentence"
    WORD = "word"
    PARAGRAPH = "paragraph"


_SPLIT_PATTERNS: dict[ChunkingMode, re.Pattern[str]] = {
    ChunkingMode.SENTENCE: re.compile(r"(?<=[.!?])(?=\s+)"),
    ChunkingMode.WORD: re.compile(r"\s+"),
    ChunkingMode.PARAGRAPH: re.compile(r"\n\n+"),
}


class ChunkingStrategy(abc.ABC):
    @abc.abstractmethod
    def split(self, text: str) -> list[str]: ...


class RegexChunking(ChunkingStrategy):
    def __init__(self, mode: ChunkingMode):
        self._pattern = _SPLIT_PATTERNS[mode]

    def split(self, text: str) -> list[str]:
        return self._pattern.split(text)


_SENTINEL = None


def _cancel_thunk(thunk: ModelOutputThunk) -> None:
    """Cancel a thunk's underlying generate task to avoid async generator warnings."""
    if thunk._generate is not None:
        thunk._generate.cancel()


async def _validate_chunk(
        chunk: str,
        qc_reqs: list[Requirement | str],
        backend: Backend,
        result: StreamChunkingResult,
        ctx: Context,
        quick_repair: ChunkRepair | None = None,
) -> tuple[bool, str]:
    """Validate a single chunk. Returns (passed, chunk_text)."""
    if not chunk.strip():
        result.quick_check_results.append([])
        return True, chunk

    validation_ctx = SimpleContext()
    chunk_results = await avalidate(
        qc_reqs, validation_ctx, backend, output=ModelOutputThunk(chunk)
    )
    result.quick_check_results.append(chunk_results)

    if not all(chunk_results):
        if quick_repair is not None:
            should_continue, repaired = await quick_repair(chunk, ctx, qc_reqs, chunk_results)
            if should_continue:
                return True, repaired
        result.failed_chunk = chunk
        return False, chunk

    return True, chunk


async def stream_with_chunking(
        instruction: Instruction,
        backend: Backend,
        ctx: Context,
        *,
        quick_check_requirements: list[Requirement | str] | None = None,
        chunking: ChunkingMode | ChunkingStrategy = ChunkingMode.SENTENCE,
        model_options: dict | None = None,
        quick_repair: ChunkRepair | None = None,
        quick_check_backend: Backend | None = None,
        tool_calls: bool = False,
) -> StreamChunkingResult:
    """Stream LLM output, validating chunks against quick-check requirements.

    Returns a StreamChunkingResult immediately. Use ``async for chunk in
    result.astream()`` to consume validated chunks as they arrive, or
    ``await result.acomplete()`` to wait for all processing to finish.
    """
    result = StreamChunkingResult()
    if quick_check_backend is None:
        quick_check_backend = backend

    async def _run() -> None:
        try:
            thunk, new_ctx = await backend.generate_from_context(
                instruction, ctx, model_options=model_options, tool_calls=tool_calls,
            )

            qc_reqs: list[Requirement | str] = quick_check_requirements or []
            chunker = RegexChunking(chunking) if isinstance(chunking, ChunkingMode) else chunking
            buffer = ""

            while not thunk.is_computed():
                delta = await thunk.astream()
                result.full_text += delta

                buffer += delta
                parts = chunker.split(buffer)
                buffer = parts[-1]

                for chunk in parts[:-1]:
                    if qc_reqs:
                        passed, chunk = await _validate_chunk(chunk,
                                                              qc_reqs,
                                                              quick_check_backend,
                                                              result,
                                                              ctx,
                                                              quick_repair)
                        if not passed:
                            result.completed = False
                            _cancel_thunk(thunk)
                            return
                    result.validated_chunks.append(chunk)
                    await result._chunk_queue.put(chunk)

            # Validate remaining buffer from the final text
            final_parts = chunker.split(result.full_text)
            already_done = len(result.validated_chunks)
            for i in range(already_done, len(final_parts)):
                chunk = final_parts[i]
                if qc_reqs:
                    passed, chunk = await _validate_chunk(chunk,
                                                          qc_reqs,
                                                          quick_check_backend,
                                                          result,
                                                          ctx,
                                                          quick_repair)
                    if not passed:
                        result.completed = False
                        _cancel_thunk(thunk)
                        return
                result.validated_chunks.append(chunk)
                await result._chunk_queue.put(chunk)

            result.full_text = "".join(result.validated_chunks)
            result.completed = True

            # Capture tool calls from the thunk if present
            if thunk.tool_calls:
                result.tool_calls = thunk.tool_calls

        except BaseException:
            result.completed = False
            raise
        finally:
            await result._chunk_queue.put(_SENTINEL)

    result._task = asyncio.create_task(_run())
    return result
