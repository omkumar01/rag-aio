# rag-generation
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Provider-agnostic LLM generation for rag-aio: normal and streaming generation, structured
output, tool calling where supported, model routing, retries, timeout policies, fallback
models, usage/cost tracking, and capability discovery. HTTP-native adapters (`httpx`) for
OpenAI, Anthropic, Google/Gemini, and OpenAI-compatible endpoints (LM Studio, vLLM,
Ollama, llama.cpp servers). Provider-specific request/response models never leak into
rag-core.

## Installation

```bash
uv pip install rag-generation
```

The package depends only on `rag-core` and `httpx>=0.27` — no vendor SDKs are pulled in.

## Architecture / Design Principles

- **Raw HTTP, no vendor SDKs.** Every adapter is built on
  `ProviderHttpClient`, a thin `httpx`-backed client that owns auth resolution and
  HTTP-status-to-error-taxonomy mapping. Wire formats (OpenAI, Anthropic, Gemini,
  Ollama) live behind that client and never leak into `rag-core` models.
- **Secrets by reference.** Authentication is resolved *per request* through an
  `api_key_provider: Callable[[], str | None]` callable (env var, file, keyring). A secret
  rotated between requests is picked up without rebuilding the client, and credentials are
  never stored on long-lived models. See ADR-0004 / ADR-0006.
- **Single contract, many implementations.** All adapters implement the
  `rag_core.protocols.LLMProvider` contract (`async complete(request) -> GenerationResult`)
  and additionally expose `stream(request)` and `list_models()`. Providers are addressed by
  the `ProviderKind` literal defined in `rag-core`, so the platform stays
  vendor-agnostic at every layer above.
- **Operational policy is centralized, not per-adapter.** Routing, retries, fallback
  chains, and usage hooks live on `GenerationService` so every backend gets the same
  resilience behavior for free.
- **Offline-first by default.** Per ADR-0004, the local default backend is LM Studio
  (`http://localhost:1234/v1`). Adapters are unit-tested fully offline with
  `httpx.MockTransport`; no network or live endpoint is required.

## Adapters

All adapters implement the `LLMProvider` protocol (`async complete`) and share the HTTP
plumbing in `http.py` (`ProviderHttpClient`): auth is resolved per request via an
`api_key_provider` callable (secrets-by-reference), and HTTP status codes map to the
rag-core error taxonomy.

| Class                         | Kind / wire format                                       | Auth header                                | Streaming        |
| ----------------------------- | -------------------------------------------------------- | ------------------------------------------ | ---------------- |
| `OpenAICompatibleProvider`    | OpenAI `/chat/completions` (LM Studio, vLLM, llama.cpp, Ollama OpenAI layer) | `Authorization: Bearer`                    | SSE (`data:`)    |
| `OpenAIProvider`              | `https://api.openai.com/v1`                              | `Authorization: Bearer` (+ Org/Project)    | SSE (`data:`)    |
| `AnthropicProvider`           | `/v1/messages`                                           | `x-api-key` + `anthropic-version`          | SSE (`data:`)    |
| `GeminiProvider`              | `generateContent`                                      | `x-goog-api-key`                           | SSE (`data:`, `?alt=sse`) |
| `OllamaProvider`              | native `/api/chat`                                       | none                                       | NDJSON (line-delimited) |

`ProviderKind` aliases `vllm` and `llamacpp` both resolve to `OpenAICompatibleProvider`
(see the registry below).

## Capability notes

- **Structured output**: when a request carries `json_schema`, `OpenAICompatibleProvider`
  sends `response_format`. The `supports_json_schema` flag selects the structured
  `json_schema` form (OpenAI o-models) versus a `json_object` fallback (LM Studio / local
  servers) — best-effort and never raises on unsupported servers.
- **Tool calling**: openai-compatible providers pass `tools` through verbatim and map
  `tool_calls` back onto `GenerationResult`; Anthropic maps `tool_use` blocks similarly.
- **Streaming**: `OpenAICompatibleProvider`, `AnthropicProvider`, and `GeminiProvider`
  consume `text/event-stream` (the `[DONE]` sentinel terminates OpenAI/Gemini; Anthropic
  ends on `message_stop`). `OllamaProvider` consumes raw NDJSON. Malformed lines are
  skipped gracefully. `collect_stream()` concatenates deltas into a single `str`.
- **Finish reasons**: mapped onto rag-core `FinishReason` (`stop`, `length`,
  `tool_calls`, `content_filter`). Unknown reasons degrade to `stop`.

## Routing

`GenerationService` holds a `providers` dict and routes via `generate(request,
provider_name=None, model=None)`:

- Default provider = first entry; override with `provider_name` or a `router` callable.
- `model` overrides `request.model` per call.
- Retry with exponential backoff (`backoff_base * 2**n`, default 0.5s, capped at
  `max_retries`) only on `RateLimitError` / `ProviderUnavailableError`; auth (`code="auth"`)
  errors propagate immediately without retry or fallback.
- `fallback_names` names a provider chain tried in order once the primary is exhausted.
- `on_usage(result)` is invoked with every result carrying a `Usage`.

## Public API

### Factory & registry

```python
from rag_generation import (
    GeneratorRegistry,
    create_provider,
    collect_stream,
    GenerationService,
    ProviderHttpClient,
    OpenAICompatibleProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
)
from rag_core.models_info import ProviderKind

# Build a single provider. base_url may be None for providers with a known default.
provider = create_provider(
    kind="openai",  # "openai" | "anthropic" | "gemini" |
    #  "openai_compatible" | "ollama" | "vllm" | "llamacpp"
    base_url=None,  # None -> provider default (e.g. api.openai.com)
    api_key_provider=lambda: __import__("os").environ.get("OPENAI_API_KEY"),
    supports_json_schema=True,  # OpenAI-compatible only; gates json_schema vs json_object
)

# Resolve a kind to its adapter class (vllm/llamacpp -> OpenAICompatibleProvider).
GeneratorRegistry.provider_class("vllm")  # -> OpenAICompatibleProvider
```

### HTTP client

`ProviderHttpClient` is the shared base for every adapter. It is also usable directly for
custom providers:

```python
import httpx
from rag_generation import ProviderHttpClient


async def handler(req: httpx.Request) -> httpx.Response: ...  # return an httpx.Response


client = ProviderHttpClient(
    base_url="https://api.my-provider.com/v1",
    api_key_provider=lambda: "env-token",  # resolved per request
    timeout_s=60.0,
    transport=httpx.MockTransport(handler),  # injectable for tests
)

data = await client.post_json("/chat/completions", {"model": "x", "messages": []})
async for event in client.stream_sse("/chat/completions", {"stream": True}):
    ...
```

Status codes map onto the rag-core error taxonomy:

| HTTP status      | Raised exception                          | Code            | Retries? |
| ---------------- | ----------------------------------------- | --------------- | -------- |
| `401` / `403`    | `ProviderError`                           | `auth`          | no       |
| `404`            | `ProviderUnavailableError`                | `provider_unavailable` | yes |
| `429`            | `RateLimitError`                          | `rate_limited`  | yes      |
| `>= 500`         | `ProviderUnavailableError`                | `provider_unavailable` | yes |
| other `>= 400`   | `ProviderError`                           | `provider_error` | no      |
| transport/timeout | `ProviderUnavailableError`               | `provider_unavailable` | yes |

### GenerationService (routing + resilience)

```python
import os
from rag_generation import GenerationService, OpenAICompatibleProvider, OpenAIProvider
from rag_core.generation import GenerationRequest, Message

lm_studio = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
openai = OpenAIProvider(api_key_provider=lambda: os.environ["OPENAI_API_KEY"])


def router(request: GenerationRequest) -> str | None:
    # Route by model name, cost, or capability — returning a registered provider name.
    return "openai" if request.model and "gpt" in request.model else None


async def on_usage(result) -> None:
    print(result.usage and result.usage.total_tokens)


service = GenerationService(
    providers={"lm_studio": lm_studio, "openai": openai},
    router=router,
    max_retries=3,
    backoff_base=0.5,
    fallback_names=["openai"],  # tried when the primary chain is exhausted
    on_usage=on_usage,
)

request = GenerationRequest(
    messages=[Message(role="user", content="Explain JWTs in one sentence.")],
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=128,
)
result = await service.generate(request)  # -> GenerationResult
print(result.text, result.finish_reason, result.usage)
```

### Streaming

```python
# Non-blocking: stream deltas directly from the resolved provider.
stream = await service.generate(
    GenerationRequest(messages=[Message(role="user", content="Write a haiku.")]),
    provider_name="lm_studio",
    model="mistralai/ministral-3-3b",
)
# GenerationService exposes the provider's native stream through the orchestrator's
# Generator shim; for direct adapter streaming see the usage guide below.
```

### Collecting a stream

```python
from rag_generation import OpenAICompatibleProvider, collect_stream
from rag_core.generation import GenerationRequest, Message

provider = OpenAICompatibleProvider(base_url="http://localhost:1234/v1")
request = GenerationRequest(messages=[Message(role="user", content="Hello")])
delta_stream = provider.stream(request)  # AsyncIterator[str]
text = await collect_stream(delta_stream)  # -> str
```

## Usage Guides

### Beginner: run against LM Studio locally

LM Studio exposes an OpenAI-compatible endpoint on `http://localhost:1234/v1` with no
API key required — the zero-config default for local development:

```python
import asyncio
from rag_generation import GenerationService, OpenAICompatibleProvider
from rag_core.generation import GenerationRequest, Message

service = GenerationService({"lm_studio": OpenAICompatibleProvider()})


async def main():
    req = GenerationRequest(
        messages=[Message(role="user", content="What is the capital of France?")],
        model="mistralai/ministral-3-3b",
    )
    result = await service.generate(req)
    print(result.text)  # the answer
    print(result.finish_reason)  # "stop" | "length" | ...


asyncio.run(main())
```

### Intermediate: structured output + tool calling

```python
import json
from pydantic import BaseModel, Field as PydanticField
from rag_core.generation import GenerationRequest, Message
from rag_generation import GenerationService, OpenAICompatibleProvider


class Summary(BaseModel):
    title: str
    bullets: list[str] = PydanticField(default_factory=list)


provider = OpenAICompatibleProvider(
    base_url="http://localhost:1234/v1",
    supports_json_schema=True,  # sends response_format=json_schema; else json_object fallback
)
service = GenerationService({"lm_studio": provider})

req = GenerationRequest(
    messages=[Message(role="user", content="Summarize quantum computing.")],
    model="mistralai/ministral-3-3b",
    json_schema=Summary.model_json_schema(),
)
result = await service.generate(req)
parsed = Summary.model_validate_json(result.text)
print(parsed.title, parsed.bullets)
```

Tools are forwarded verbatim on openai-compatible providers and mapped back onto
`GenerationResult.tool_calls`; Anthropic providers map `tool_use` blocks the same way.

### Intermediate: streaming a response

```python
async for delta in provider.stream(request):
    print(delta, end="", flush=True)
```

`stream()` yields text deltas only; `[DONE` / `message_stop` / end-of-NDJSON terminates
iteration. Malformed lines are skipped, so a single bad SSE frame never aborts a stream.

### Advanced: multi-provider fallback chain

```python
import os
from rag_generation import (
    GenerationService,
    OpenAICompatibleProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
)

providers = {
    "lm_studio": OpenAICompatibleProvider(base_url="http://localhost:1234/v1"),
    "openai": OpenAIProvider(api_key_provider=lambda: os.environ["OPENAI_API_KEY"]),
    "anthropic": AnthropicProvider(api_key_provider=lambda: os.environ["ANTHROPIC_API_KEY"]),
    "gemini": GeminiProvider(api_key_provider=lambda: os.environ["GEMINI_API_KEY"]),
}

service = GenerationService(
    providers,
    max_retries=4,
    backoff_base=0.3,
    fallback_names=["lm_studio", "openai"],
)

# Primary = "openai"; if it rate-limits or is unavailable, the chain rolls to
# "lm_studio", then "openai" again via the fallback list. Auth errors raise
# immediately (no fallback).
result = await service.generate(request, provider_name="openai")
```

### Advanced: provider discovery via `list_models`

```python
models = await openai.list_models()  # OpenAI / vLLM / LM Studio
# await ollama.list_models()  # Ollama native; Anthropic/Gemini do not expose this.
```

## Configuration

Providers are constructed in code (see the Public API above); `rag-generation` itself
holds no config file. Operational knobs on `GenerationService`:

| Field            | Type                       | Default | Meaning                                                       |
| ---------------- | -------------------------- | ------- | ------------------------------------------------------------ |
| `providers`      | `dict[str, LLMProvider]`   | *req*   | Named providers; the first is the default when `router` returns `None`. |
| `router`         | `Callable[[GenerationRequest], str \| None] \| None` | `None` | Chooses a provider name per request; `None` falls back to the default. |
| `max_retries`    | `int`                      | `3`     | Max retry attempts on `RateLimitError` / `ProviderUnavailableError`. |
| `backoff_base`   | `float` (seconds)          | `0.5`   | Exponential backoff base: `backoff_base * 2**attempt`.       |
| `fallback_names` | `list[str]`                | `[]`    | Provider chain tried in order once the primary's retries are exhausted. |
| `on_usage`       | `Callable[[GenerationResult], None] \| None` | `None` | Invoked with every result that carries a `Usage`.            |

Adapter-level knobs (`timeout_s`, `transport`, `supports_json_schema`,
`extra_headers`, `organization`) are constructor arguments documented on each class.

## Testing

All unit tests run fully offline via `httpx.MockTransport`; no network or live
endpoint is needed.

```bash
uv run ruff format packages/rag-generation && uv run ruff check packages/rag-generation
uv run mypy packages/rag-generation/src
uv run pytest packages/rag-generation -q
```

Test layout:

```
tests/
  conftest.py              # make_request fixture + wire() -> MockTransport factory
  test_smoke.py            # import / version guard
  test_generation_service.py  # routing, retry, backoff, fallback, on_usage
  test_openai_compatible.py    # payload build, SSE parsing, finish reasons, error taxonomy
  test_openai.py               # org/project header injection
  test_anthropic.py            # system hoisting, tool_use mapping, stop_reason
  test_gemini.py               # role map, finishReason, usageMetadata
  test_ollama.py               # NDJSON parsing, done_reason, options mapping
  test_secrets.py             # api_key_provider resolved per request
```

## Dependencies

- `rag-core` — `LLMProvider`, `GenerationRequest`, `GenerationResult`, `Usage`,
  `FinishReason`, `Message`, and the error taxonomy (`ProviderError`,
  `ProviderUnavailableError`, `RateLimitError`).
- `httpx>=0.27` — all network I/O; `MockTransport` is also the unit-test seam.

## Cross-Package Relationships

- **Depends on `rag-core`** exclusively for domain models and the `LLMProvider`
  protocol. Provider wire formats never cross back into `rag-core`.
- **Consumed by `rag-orchestrator`** via the `Generator` protocol:
  `OrchestratorServices.generator` is produced by wrapping a `GenerationService` in an
  `_ServiceGenerator` adapter (see `services.py`), and `load_local_services()` constructs
  the default LM Studio `OpenAICompatibleProvider` end-to-end.
- **Used alongside `rag-llm-provider`**: `rag-llm-provider` owns *control-plane* provider
  config (names, base URLs, `SecretRef`s, model registry, routing), while `rag-generation`
  owns the *data-plane* HTTP clients. The orchestrator bridges the two:
  `ModelRouter`/`ProviderRegistry` decide *which* provider/model, and `GenerationService`
  *calls* it. `create_provider` mirrors `ProviderKind` exactly.
- The top-level `rag-aio` facade wires `GenerationService` (live or stub) into the
  orchestrator's generator slot; in the offline profile it substitutes
  `_StubGenerator` (no HTTP).
- See ADR-0002 (protocol-based contracts) and ADR-0004 (LM Studio as the default local
  provider).

## License

MIT
