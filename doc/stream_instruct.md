# `stream_instruct`

Streaming counterpart to `MelleaSession.instruct()`. Emits a rich event stream covering chunks, quick-check results, repairs, retries, and completion. Registered as a powerup on `MelleaSession` — import `mellea_partial` to activate it.

## Usage

```python
import mellea_partial  # activates the powerup

result = await session.stream_instruct(
    "Write a short story about a robot.",
    quick_check_requirements=["no violence"],
    requirements=["ends on a hopeful note"],
    strategy=RejectionSamplingStrategy(loop_budget=3),
)

async for text in result.astream_text():
    print(text, end="", flush=True)

await result.acomplete()
print(result.success, result.final_text)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `description` | `str` | — | Instruction description forwarded to `Instruction`. |
| `requirements` | `list[Requirement \| str] \| None` | `None` | Full-output requirements validated after streaming completes. |
| `quick_check_requirements` | `list[Requirement \| str] \| None` | `None` | Per-chunk requirements validated during streaming. |
| `quick_repair` | `ChunkRepair \| None` | `None` | Callback invoked when a chunk fails quick checks. Can fix and continue, or signal abort. See [quick_repair](#quick_repair) below. |
| `strategy` | `BaseSamplingStrategy \| None` | `None` | Controls retry/repair logic. `None` = single attempt, no full-output validation. |
| `chunking` | `ChunkingMode \| ChunkingStrategy` | `SENTENCE` | How to split the stream into chunks. |
| `model_options` | `dict \| None` | `None` | Extra options forwarded to the backend. |
| `icl_examples` | | `None` | In-context learning examples. |
| `grounding_context` | | `None` | Grounding context dict. |
| `user_variables` | | `None` | Jinja placeholder values. |
| `prefix` | | `None` | Instruction prefix. |
| `output_prefix` | | `None` | Output generation prefix. |
| `images` | | `None` | Images passed to the instruction. |

Returns a `StreamInstructResult` with a running background task. The call returns immediately.

## How It Works

```
stream_instruct()
  └─ creates StreamInstructResult
  └─ fires asyncio.Task(_run(...))
  └─ returns result  ← caller consumes events while generation runs
```

`_run` executes a retry loop up to `strategy.loop_budget` attempts:

1. **Generate** — calls `backend.generate_from_context()` to get a streaming `ModelOutputThunk`.
2. **Chunk + quick-check** — reads deltas, splits by the chunking pattern, runs quick-check requirements on each chunk. Emits `ChunkEvent` and `QuickCheckEvent` per chunk.
3. **Quick-check failure** — if `quick_repair` is set, calls it first; see [quick_repair](#quick_repair). If the chunk still fails (or no repair is set), cancels the thunk, emits `RetryEvent(reason="quick check failed")`, and immediately retries **without** calling `strategy.repair()`.
4. **Full-output validation** — once all chunks pass, reassembles the full text and validates against full-output requirements.
5. **Success** — emits `CompletedEvent(success=True)`, updates `session.ctx`, returns.
6. **Full-validation failure** — calls `strategy.repair()` to produce an updated instruction and context for the next attempt, emits `RetryEvent(reason="full validation failed")`.
7. **Budget exhausted** — calls `strategy.select_from_failure()` to pick the best result from all sampled attempts. Emits `CompletedEvent(success=False)`.

### No strategy / no requirements

If `strategy=None` and no `requirements` are provided, the loop runs exactly once and succeeds as long as all quick checks pass.

## quick_repair

`quick_repair` is an async callable with this signature:

```python
async def quick_repair(
    chunk: str,
    ctx: Context,
    qc_reqs: list[Requirement | str],
    results: list[ValidationResult],
) -> tuple[bool, str]:
    ...
```

| Return value | Meaning |
|---|---|
| `(True, repaired_text)` | Use `repaired_text` in place of the original chunk and continue streaming. Emits `ChunkRepairedEvent`. |
| `(False, any_text)` | Abort streaming for this attempt (same as having no repair). |

`quick_repair` is called only when at least one quick-check requirement fails. The `results` list corresponds 1-to-1 with `qc_reqs` and carries the `ValidationResult` for each requirement.

### Example — strip leading list numbers

```python
import re

async def strip_numbering(chunk, ctx, qc_reqs, results):
    return (True, re.sub(r"^\s*\d+[\.\)]\s*", "", chunk))

result = await session.stream_instruct(
    "Write a haiku about the ocean.",
    quick_check_requirements=[
        Requirement(
            "Chunk must not start with a number.",
            simple_validate(lambda x: not re.match(r"\s*\d", x)),
            check_only=True,
        ),
    ],
    quick_repair=strip_numbering,
)
```

When a chunk fails, `strip_numbering` removes the leading number and returns `(True, fixed)`. `ChunkRepairedEvent` is emitted and streaming continues with the fixed text.

To abort instead (e.g. if the repair is not possible):

```python
async def abort_on_failure(chunk, ctx, qc_reqs, results):
    return (False, chunk)
```

## Sampling Strategy

The strategy is consulted in three places:

| Property / Method | When | Purpose |
|---|---|---|
| `strategy.loop_budget` | Once at startup | Maximum number of attempts |
| `strategy.repair(ctx, result_ctx, actions, results, scores)` | After each full-validation failure | Produce an updated instruction/context for the next attempt |
| `strategy.select_from_failure(actions, results, scores)` | When budget is exhausted | Pick the best attempt index from all sampled results |

**Note:** `strategy.requirements` overrides `instruction.requirements` for full-output validation. If `strategy.requirements is None`, the instruction's own requirements are used.

**Quick-check failures bypass `strategy.repair()`** — they retry the same instruction immediately.

## Event Types

All events inherit from `StreamEvent` (carries a `timestamp`).

| Event | Fields | Emitted when |
|---|---|---|
| `ChunkEvent` | `text`, `chunk_index`, `attempt` | A validated (or repaired) chunk is ready |
| `QuickCheckEvent` | `chunk_index`, `attempt`, `passed`, `results` | Quick checks run on a chunk |
| `ChunkRepairedEvent` | `chunk_index`, `attempt`, `original`, `repaired` | `quick_repair` returns `(True, repaired_text)` |
| `StreamingDoneEvent` | `attempt`, `full_text` | All chunks for an attempt received and passed quick checks |
| `FullValidationEvent` | `attempt`, `passed`, `results` | Full-output validation completes |
| `RetryEvent` | `attempt`, `reason` | A new attempt is about to begin |
| `CompletedEvent` | `success`, `full_text`, `attempts_used` | Processing is fully done |

## `StreamInstructResult`

| Member | Description |
|---|---|
| `attempts` | List of `AttemptRecord`, one per attempt |
| `success` | `True` if any attempt fully passed |
| `final_text` | Best output text (winning attempt, or best-from-failure) |
| `astream_events()` | `AsyncIterator[StreamEvent]` — all events in arrival order |
| `astream_text()` | `AsyncIterator[str]` — text from `ChunkEvent`s of the latest attempt only; skips superseded attempts on retry |
| `acomplete()` | Awaitable — drains the queue and returns `self` with all fields populated |
| `as_thunk` | `ModelOutputThunk` wrapping `final_text` for mellea interop |

### `AttemptRecord`

Per-attempt bookkeeping stored in `StreamInstructResult.attempts`.

| Field | Description |
|---|---|
| `attempt_index` | Zero-based attempt number |
| `validated_chunks` | Chunks that passed all quick checks (after any repair) |
| `quick_check_results` | Per-chunk list of `ValidationResult` |
| `full_validation_results` | `list[tuple[Requirement, ValidationResult]]` or `None` |
| `full_text` | Accumulated raw text for this attempt |
| `quick_checks_passed` | Whether all chunks passed quick checks |
| `full_validation_passed` | Whether full-output validation passed |
| `context` | Result context from the backend |
| `thunk` | The `ModelOutputThunk` used for generation |
