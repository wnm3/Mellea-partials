# `stream_instruct`

Streaming counterpart to `MelleaSession.instruct()`. Emits a rich event stream covering chunks, quick-check results, repairs, retries, and completion. Registered as a powerup on `MelleaSession` — import `stream_instruct` to activate it.

## Usage

```python
import stream_instruct  # activates the powerup

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
| `strategy` | `BaseSamplingStrategy \| None` | `None` | Controls retry/repair logic. `None` = single attempt, no full-output validation. |
| `chunking_mode` | `ChunkingMode` | `SENTENCE` | How to split the stream into chunks (`SENTENCE`, `PARAGRAPH`, etc.). |
| `on_chunk_failure` | `OnChunkFailure \| None` | `None` | Callback invoked when a chunk fails quick checks. Return a repaired string or `None`. |
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
3. **Quick-check failure** — cancels the thunk, emits `RetryEvent(reason="quick check failed")`, and immediately retries **without** calling `strategy.repair()`.
4. **Full-output validation** — once all chunks pass, reassembles the full text and validates against full-output requirements.
5. **Success** — emits `CompletedEvent(success=True)`, updates `session.ctx`, returns.
6. **Full-validation failure** — calls `strategy.repair()` to produce an updated instruction and context for the next attempt, emits `RetryEvent(reason="full validation failed")`.
7. **Budget exhausted** — calls `strategy.select_from_failure()` to pick the best result from all sampled attempts. Emits `CompletedEvent(success=False)`.

### No strategy / no requirements

If `strategy=None` and no `requirements` are provided, the loop runs exactly once and succeeds as long as all quick checks pass.

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
| `ChunkEvent` | `text`, `chunk_index`, `attempt` | A validated chunk is ready |
| `QuickCheckEvent` | `chunk_index`, `attempt`, `passed`, `results` | Quick checks run on a chunk |
| `ChunkRepairedEvent` | `chunk_index`, `attempt`, `original`, `repaired` | `on_chunk_failure` successfully repairs a chunk |
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
| `validated_chunks` | Chunks that passed all quick checks |
| `quick_check_results` | Per-chunk list of `ValidationResult` |
| `full_validation_results` | `list[tuple[Requirement, ValidationResult]]` or `None` |
| `full_text` | Accumulated raw text for this attempt |
| `quick_checks_passed` | Whether all chunks passed quick checks |
| `full_validation_passed` | Whether full-output validation passed |
| `context` | Result context from the backend |
| `thunk` | The `ModelOutputThunk` used for generation |