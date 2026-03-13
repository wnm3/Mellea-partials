"""Demo for stream_with_chunking: two scenarios showcasing code-based quick checks."""

import asyncio

from mellea.backends import ModelOption
from mellea.backends.ollama import OllamaModelBackend
from mellea.core.requirement import Requirement
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.requirements.requirement import simple_validate

from stream_with_chunking import ChunkingMode, stream_with_chunking


def print_header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_result(result) -> None:
    print(f"\n--- Generated text ---\n{result.full_text}\n")

    for i, chunk in enumerate(result.validated_chunks):
        if not chunk.strip():
            continue
        statuses = result.quick_check_results[i] if i < len(result.quick_check_results) else []
        passed = all(statuses) if statuses else True
        print(f"  Chunk {i + 1}: {'PASS' if passed else 'FAIL'} | {chunk.strip()[:80]}")

    if result.failed_chunk:
        print(f"\n  Failed chunk: {result.failed_chunk.strip()[:120]}")

    print(f"\n  Status: {'Completed' if result.completed else 'Stopped early'}")

    if result.full_validation_results is not None:
        print("\n  Full validation results:")
        for vr in result.full_validation_results:
            print(f"    - {bool(vr)}: {vr.reason or '(no reason)'}")


async def demo_happy_path(backend: OllamaModelBackend) -> None:
    print_header("Demo 1: Happy path — sentence chunking with code-based quick checks")

    instruction = Instruction(
        description="Explain what a neural network is in 5 sentences.",
        requirements=["The response must be exactly 5 sentences."],
    )

    quick_checks = [
        Requirement(
            "Each sentence must be under 200 characters.",
            simple_validate(lambda x: len(x.strip()) < 200),
            check_only=True,
        ),
    ]

    result = await stream_with_chunking(
        instruction,
        backend,
        quick_check_requirements=quick_checks,
        chunking_mode=ChunkingMode.SENTENCE,
    )

    async for chunk in result.astream():
        print(f"  [chunk] {chunk.strip()}", flush=True)

    print_result(result)


async def demo_fail_fast(backend: OllamaModelBackend) -> None:
    print_header("Demo 2: Fail-fast — code-based check catches violation mid-stream")

    instruction = Instruction(
        description="List 5 fun facts about space. Number each fact starting with the digit (e.g. 1. ...)",
    )

    quick_checks = [
        Requirement(
            "Sentence must not contain a digit.",
            simple_validate(lambda x: not any(c.isdigit() for c in x)),
            check_only=True,
        ),
    ]

    result = await stream_with_chunking(
        instruction,
        backend,
        quick_check_requirements=quick_checks,
        chunking_mode=ChunkingMode.SENTENCE,
    )
    await result.acomplete()

    print_result(result)


async def main() -> None:
    backend = OllamaModelBackend(model_options={ModelOption.STREAM: True})

    await demo_happy_path(backend)
    await demo_fail_fast(backend)


if __name__ == "__main__":
    asyncio.run(main())
