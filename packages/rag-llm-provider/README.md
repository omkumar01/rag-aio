# rag-llm-provider

Provider and model management control layer for the rag-aio platform.

`rag-llm-provider` owns the **control-plane** description of LLM providers and
models: endpoint config, auth *references*, declared model metadata (roles,
context window, embedding dim, capabilities, aliases), default-model routing,
and health state. It is **async-first**, fully typed, and depends only on
`rag-core` — **no `httpx`** is pulled in here. Concrete API clients and health
probes are injected by consumers (generators, embedders, the orchestrator,
etc.).

Secrets are stored **by reference only** (ADRs 0004 / 0006): a `SecretRef`
holds an env-var name or a file path, never a credential value. Resolved
values are never written back onto models, emitted by `repr`/`str`/
`model_dump`, or surfaced in logs.

## Install

```bash
uv pip install rag-llm-provider
```

## Public API

### Secrets (`secrets.py`)
- `SecretRef` — a `RagBaseModel` with `kind: Literal["env", "file", "none"]`
  and `ref: str | None`. Forbids extra fields and is repr-safe by design
  (it only ever holds the reference, never the value).
- `resolve_secret(ref) -> str | None` — resolves env / file / none. Missing
  env vars and missing files raise `ConfigError` with a clear message.
- `mask_secret(value, keep=4) -> str` — renders `sk-...abcd`-style masks for
  logs; returns `"<unset>"` for `None`/empty.

### Configuration (`config.py`)
- `ModelConfig` — `model_id`, `roles`, `context_window`, `max_output_tokens`,
  `embedding_dim`, `capabilities`, `temperature`, `aliases`.
- `ProviderConfig` — `name`, `kind: ProviderKind`, `base_url` (defaults per
  kind), `secret: SecretRef`, `models`, `default_models`, `enabled`,
  `timeout_s`, `max_retries`. Validates unique model ids and that
  `default_models` only references declared model ids.
- `ProvidersConfig` — `providers: list[ProviderConfig]` plus a
  `ProvidersConfig.local_default()` classmethod.
- `DEFAULT_BASE_URLS` — per-`ProviderKind` default endpoints.

### Registry (`registry.py`)
- `ProviderRegistry(config)` — indexes providers by name, caches per-provider
  health.
  - `get(name)` → `ProviderConfig` (raises `ConfigError` if unknown).
  - `find_model(name_or_alias)` — searches enabled providers in config order,
    model id then alias.
  - `resolve_role(role)` — first model (config order) whose `roles` include it.
  - `to_provider_info(provider) -> ProviderInfo` — projects to the public
    `rag_core` schema; exposes only `auth_ref` (the reference name), never a
    secret value, plus the cached `health` state.
  - `names()`, `enabled()`, `set_health(name, state)`.
- `HealthProbe` — a `Protocol` with `async def probe(provider) -> HealthState`.
  Consumers inject a concrete probe (e.g. an `httpx`-based one); this package
  declares only the contract.
- `RegistryHealth(registry)` — `async def check_all(probe) -> dict[str, HealthState]`,
  updating the registry's cached health for each provider.

### Routing (`routing.py`, `aliases.py`)
- `expand_alias(registry, name) -> str | None` — canonicalizes a model id or
  alias to its `model_id` (or `None`).
- `ModelRouter(registry, fallbacks=None)` — `async def route(role, prefer=None,
  required_capabilities=None) -> RouteDecision`. Resolution order:
  1. `prefer` (explicit id/alias) satisfying capabilities,
  2. role resolution over declared `roles`,
  3. `fallbacks[role]` chain in order,
  4. else `ProviderError("no model for role ...")`.
  - `RouteDecision(provider, model_id, fallback_used, attempted)`.
  - `route` is async despite currently synchronous resolution: the contract is
    async so future dynamic capability/health checks can be added without
    breaking callers.

## Default profile (ADR-0004)

The platform ships a local-first default via `ProvidersConfig.local_default()`:
a single **OpenAI-compatible** provider targeting LM Studio at
`http://localhost:1234/v1` with no configured secret. This keeps local
development and CI credential-free; cloud providers (OpenAI, Anthropic,
Gemini) are opt-in via an `env`/`file` `SecretRef`.

Default endpoints per `ProviderKind`:

| kind              | default `base_url`                              |
| ----------------- | ----------------------------------------------- |
| `openai`          | `https://api.openai.com/v1`                     |
| `anthropic`       | `https://api.anthropic.com`                     |
| `gemini`          | `https://generativelanguage.googleapis.com`     |
| `openai_compatible` | `http://localhost:1234/v1` (LM Studio)        |
| `ollama`          | `http://localhost:11434`                        |
| `vllm`            | `http://localhost:8000/v1`                      |
| `llamacpp`        | `http://localhost:8080`                         |

## Secret handling

```python
from rag_llm_provider import SecretRef, resolve_secret, mask_secret

ref = SecretRef(kind="env", ref="OPENAI_API_KEY")  # holds only the name
value = resolve_secret(ref)  # reads os.environ[...]
print(mask_secret(value))  # -> "sk-...wxyz"
```

A `SecretRef` is structurally incapable of holding a credential, so its
`repr`/`str`/`model_dump` are always log- and wire-safe.

## Health checks (abstract)

Health probing is intentionally abstract:

```python
class MyProbe(HealthProbe):
    async def probe(self, provider) -> HealthState: ...


states = await RegistryHealth(registry).check_all(MyProbe())
```

This keeps the package `httpx`-free; transport concerns live with the injected
probe.

## License

MIT
