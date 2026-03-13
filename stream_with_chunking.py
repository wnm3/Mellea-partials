"""Stream LLM output with chunk-by-chunk validation.

Two-phase validation:
1. Quick checks: validated per-chunk during streaming (stops on failure)
2. Full output checks: validated once after complete output (instruction requirements)
"""

import asyncio
import enum
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from mellea.core.backend import Backend
from mellea.core.base import ModelOutputThunk
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.functional import avalidate


class ChunkingMode(enum.Enum):
    SENTENCE = "sentence"
    WORD = "word"
    PARAGRAPH = "paragraph"


_SPLIT_PATTERNS: dict[ChunkingMode, re.Pattern[str]] = {
    ChunkingMode.SENTENCE: re.compile(r"(?<=[.!?])\s+"),
    ChunkingMode.WORD: re.compile(r"\s+"),
    ChunkingMode.PARAGRAPH: re.compile(r"\n\n+"),
}

_SENTINEL = None


@dataclass
class StreamChunkingResult:
    validated_chunks: list[str] = field(default_factory=list)
    failed_chunk: str | None = None
    quick_check_results: list[list[ValidationResult]] = field(default_factory=list)
    full_validation_results: list[ValidationResult] | None = None
    completed: bool = False
    full_text: str = ""
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

    async def acomplete(self) -> "StreamChunkingResult":
        """Wait for the background task to finish and return self."""
        if self._task is not None:
            await self._task
        return self


def _cancel_thunk(thunk: ModelOutputThunk) -> None:
    """Cancel a thunk's underlying generate task to avoid async generator warnings."""
    if thunk._generate is not None:
        thunk._generate.cancel()


async def _validate_chunk(
    chunk: str,
    qc_reqs: list[Requirement | str],
    backend: Backend,
    result: StreamChunkingResult,
) -> bool:
    """Validate a single chunk. Returns True if passed, False if failed."""
    if not chunk.strip():
        result.quick_check_results.append([])
        return True

    validation_ctx = SimpleContext()
    chunk_results = await avalidate(
        qc_reqs, validation_ctx, backend, output=ModelOutputThunk(chunk)
    )
    result.quick_check_results.append(chunk_results)

    if not all(chunk_results):
        result.failed_chunk = chunk
        return False

    return True


async def stream_with_chunking(
    instruction: Instruction,
    backend: Backend,
    *,
    quick_check_requirements: list[Requirement | str] | None = None,
    chunking_mode: ChunkingMode = ChunkingMode.SENTENCE,
    model_options: dict | None = None,
) -> StreamChunkingResult:
    """Stream LLM output, validating chunks against quick-check requirements.

    Returns a StreamChunkingResult immediately. Use ``async for chunk in
    result.astream()`` to consume validated chunks as they arrive, or
    ``await result.acomplete()`` to wait for all processing to finish.
    """
    result = StreamChunkingResult()

    async def _run() -> None:
        try:
            ctx = SimpleContext()
            thunk, ctx = await backend.generate_from_context(
                instruction, ctx, model_options=model_options
            )

            qc_reqs: list[Requirement | str] = quick_check_requirements or []
            pattern = _SPLIT_PATTERNS[chunking_mode]
            buffer = ""

            while not thunk.is_computed():
                delta = await thunk.astream()
                result.full_text += delta

                buffer += delta
                parts = pattern.split(buffer)
                buffer = parts[-1]

                for chunk in parts[:-1]:
                    if qc_reqs and not await _validate_chunk(chunk, qc_reqs, backend, result):
                        result.completed = False
                        _cancel_thunk(thunk)
                        return
                    result.validated_chunks.append(chunk)
                    await result._chunk_queue.put(chunk)

            # Stream finished — get final text
            result.full_text = await thunk.avalue()

            # Validate remaining buffer from final text
            final_parts = pattern.split(result.full_text)
            already_done = len(result.validated_chunks)
            for i in range(already_done, len(final_parts)):
                chunk = final_parts[i]
                if qc_reqs and not await _validate_chunk(chunk, qc_reqs, backend, result):
                    result.completed = False
                    _cancel_thunk(thunk)
                    return
                result.validated_chunks.append(chunk)
                await result._chunk_queue.put(chunk)

            result.completed = True

            # Phase 2: full-output validation against instruction requirements
            if instruction.requirements:
                full_ctx = SimpleContext()
                result.full_validation_results = await avalidate(
                    instruction.requirements,
                    full_ctx,
                    backend,
                    output=ModelOutputThunk(result.full_text),
                )
        except BaseException:
            result.completed = False
            raise
        finally:
            await result._chunk_queue.put(_SENTINEL)

    result._task = asyncio.create_task(_run())
    return result
