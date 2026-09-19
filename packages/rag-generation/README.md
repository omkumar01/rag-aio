# rag-generation

Provider-agnostic LLM generation for rag-aio: normal and streaming generation, structured
output, tool calling where supported, model routing, retries, timeout policies, fallback
models, usage/cost tracking, and capability discovery. HTTP-native adapters (`httpx`) for
OpenAI, Anthropic, Google/Gemini, and OpenAI-compatible endpoints (LM Studio, vLLM,
Ollama, llama.cpp servers). Provider-specific request/response models never leak into
rag-core.

## Adapters

All adapters implement the `LLMProvider` protocol (`async complete`) and share the http
plumbing in `http.py` (`ProviderHttpClient`): auth is resolved per request via a
`Callable[[], str | None]` (secrets-by-reference), and HTTP status codes map to the
rag-core error taxonomy.

| Class                | Kind / wire format                     | Auth header            | Streaming           |
| -------------------- | -------------------------------------- | ---------------------- | ------------------- |
| `OpenAICompatibleProvider` | OpenAI `/chat/completions` (LM Studio, vLLM, llama.cpp, Ollama OpenAI layer) | `Authorization: Bearer` | SSE (`data:`)       |
| `OpenAIProvider`     | `https://api.openai.com/v1`            | `Authorization: Bearer` (+ Org headers) | SSE (`data:`)       |
| `AnthropicProvider`  | `/v1/messages`                         | `x-api-key` + `anthropic-version` | SSE (`data:`)       |
| `GeminiProvider`     | `generateContent`                      | `x-goog-api-key`       | SSE (`data:`, `?alt=sse`) |
| `OllamaProvider`     | native `/api/chat`                     | none                   | NDJSON (line-delimited) |

`ProviderKind` aliases `vllm` and `llamacpp` both resolve to `OpenAICompatibleProvider`.

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

## Local development

Per ADR-0004, the default local backend is LM Studio (`http://localhost:1234/v1`) — no
cloud API keys are required for development or CI. All unit tests run fully offline via
`httpx.MockTransport`; no network or live endpoint is needed.

```bash
uv run ruff format packages/rag-generation && uv run ruff check packages/rag-generation
uv run mypy packages/rag-generation/src
uv run pytest packages/rag-generation -q
```
