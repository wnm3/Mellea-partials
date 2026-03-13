"""Demo: hallucination detection intrinsic as a streaming quick check.

Uses two backends to avoid deadlock: LMStudioBackend for streaming generation
and LocalHFBackend for hallucination validation (needs AdapterMixin for LoRA).
"""

import asyncio

from mellea.backends import ModelOption
from mellea.backends.huggingface import LocalHFBackend
from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B
from mellea.core import Backend
from mellea.core.requirement import Requirement, ValidationResult
from mellea.stdlib.components import Document
from mellea.stdlib.components.instruction import Instruction
from mellea.stdlib.components.intrinsic import rag
from mellea.stdlib.context import ChatContext
from mellea_extra import LMStudioBackend, FixedDocument

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


async def demo_hallucination_check(
    gen_backend: Backend, val_backend: LocalHFBackend,
) -> None:
    print_header("Intrinsic quick check — hallucination detection per chunk")

    source_docs = [
        FixedDocument(
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. "
            "It was constructed from 1887 to 1889 as the centerpiece of the 1889 "
            "World's Fair. The tower is 330 metres tall and was the tallest "
            "man-made structure in the world until 1930.",
            doc_id="doc1",
        ),
    ]

    instruction = Instruction(
        description="Summarize the following document about the Eiffel Tower in 3 sentences.",
        grounding_context={"source": source_docs[0]},
    )

    def check_hallucination(ctx):
        chunk_text = ctx.last_output().value
        if not chunk_text or not chunk_text.strip():
            return ValidationResult(True)
        result_rag_test = rag.flag_hallucinated_content(
            chunk_text, source_docs, ChatContext(), val_backend
        )
        all_faithful = all(
            r.get("faithfulness_likelihood", 0) > 0.5 for r in result_rag_test
        )
        return ValidationResult(
            all_faithful,
            reason="Hallucination detected" if not all_faithful else None,
        )

    quick_checks = [
        Requirement(
            "Each chunk must be faithful to the source documents.",
            check_hallucination,
            check_only=True,
        ),
    ]

    result = await stream_with_chunking(
        instruction,
        gen_backend,
        quick_check_requirements=quick_checks,
        chunking_mode=ChunkingMode.SENTENCE,
    )

    async for chunk in result.astream():
        print(f"  [chunk] {chunk.strip()}", flush=True)

    print_result(result)


async def main() -> None:
    gen_backend = LMStudioBackend(
        "granite-4.0-micro@q8_0",
        model_options={ModelOption.STREAM: True},
    )
    val_backend = LocalHFBackend(model_id=IBM_GRANITE_4_MICRO_3B)
    await demo_hallucination_check(gen_backend, val_backend)


if __name__ == "__main__":
    asyncio.run(main())
