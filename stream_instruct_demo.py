"""Demo for stream_instruct: two scenarios showcasing sampling strategies.

Requires Ollama running locally with a model available (e.g. llama3.2).

Usage:
    python stream_instruct_demo.py
"""

from __future__ import annotations

import asyncio

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.core import Backend
from mellea.core.requirement import Requirement
from mellea.stdlib.requirements.requirement import simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.sampling.base import RepairTemplateStrategy
from mellea.stdlib.session import MelleaSession

import stream_instruct  # activates stream_instruct powerup on MelleaSession  # noqa: F401
from mellea_extra import LMStudioBackend
from stream_instruct import (
    ChunkEvent,
    ChunkRepairedEvent,
    CompletedEvent,
    FullValidationEvent,
    QuickCheckEvent,
    RetryEvent,
    StreamingDoneEvent,
)


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ─── Demo 1: Rejection Sampling ──────────────────────────────────────────────


async def demo_rejection_sampling(backend: Backend) -> None:
    print_header("Demo 1: Rejection Sampling — haiku with line-count validation")

    m = MelleaSession(backend)

    result = await m.stream_instruct(
        "Write a haiku about the ocean.",
        quick_check_requirements=[
            Requirement(
                "Each chunk must be under 100 characters.",
                simple_validate(lambda x: len(x.strip()) < 100),
                check_only=True,
            ),
        ],
        requirements=[
            Requirement(
                "The response must be exactly 3 lines.",
                simple_validate(lambda x: len(x.strip().splitlines()) == 3),
                check_only=True,
            ),
        ],
        strategy=RejectionSamplingStrategy(loop_budget=3),
    )

    async for event in result.astream_events():
        if isinstance(event, ChunkEvent):
            print(f"  [chunk  {event.attempt}:{event.chunk_index}] {event.text.strip()!r}")
        elif isinstance(event, QuickCheckEvent):
            status = "PASS" if event.passed else "FAIL"
            print(f"  [qcheck {event.attempt}:{event.chunk_index}] {status}")
        elif isinstance(event, ChunkRepairedEvent):
            print(
                f"  [repair {event.attempt}:{event.chunk_index}] {event.original.strip()!r} → {event.repaired.strip()!r}")
        elif isinstance(event, StreamingDoneEvent):
            print(f"  [done   {event.attempt}] full_text={event.full_text.strip()!r}")
        elif isinstance(event, FullValidationEvent):
            status = "PASS" if event.passed else "FAIL"
            print(f"  [fullval {event.attempt}] {status}")
        elif isinstance(event, RetryEvent):
            print(f"  [retry  → attempt {event.attempt}] reason={event.reason!r}")
        elif isinstance(event, CompletedEvent):
            status = "SUCCESS" if event.success else "FAILED"
            print(f"\n  [{status}] attempts_used={event.attempts_used}")
            print(f"  Final text:\n    {event.full_text.strip()}")


# ─── Demo 2: Repair Template ─────────────────────────────────────────────────


async def demo_repair_template(backend: Backend) -> None:
    print_header("Demo 2: Repair Template — exercise benefits with sentence count check")

    m = MelleaSession(backend)

    result = await m.stream_instruct(
        "List 3 benefits of exercise. Keep each benefit to one sentence.",
        quick_check_requirements=[
            Requirement(
                "No chunk should exceed 150 characters.",
                simple_validate(lambda x: len(x.strip()) <= 150),
                check_only=True,
            ),
        ],
        requirements=[
            Requirement(
                "The response must contain exactly 3 sentences.",
                simple_validate(
                    lambda x: len([s for s in x.replace("!", ".").replace("?", ".").split(".") if s.strip()]) == 3
                ),
                check_only=True,
            ),
        ],
        strategy=RepairTemplateStrategy(loop_budget=3),
    )

    print("  Streaming text (latest attempt only):")
    print("  ", end="", flush=True)

    async for text in result.astream_text():
        print(text, end="", flush=True)

    await result.acomplete()

    print(f"\n\n  Attempts summary:")
    for rec in result.attempts:
        qc_status = "QC passed" if rec.quick_checks_passed else "QC failed"
        fv_status = "FV passed" if rec.full_validation_passed else "FV failed"
        n_chunks = len(rec.validated_chunks)
        print(f"    Attempt {rec.attempt_index}: {qc_status}, {fv_status}, {n_chunks} chunks")

    status = "SUCCESS" if result.success else "FAILED (budget exhausted)"
    print(f"\n  Final result: {status}")


# ─── Entry point ─────────────────────────────────────────────────────────────


async def main() -> None:
    backend = LMStudioBackend("granite-4.0-micro", model_options={ModelOption.STREAM: True})

    await demo_rejection_sampling(backend)
    await demo_repair_template(backend)


if __name__ == "__main__":
    asyncio.run(main())
