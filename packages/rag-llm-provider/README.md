# rag-llm-provider
> Part of the [rag-aio](https://github.com/omkumar01/rag-aio/blob/main/README.md) monorepo — see the root README for the platform overview, quickstart, and full documentation index.

Provider and model management control layer for the
[rag-aio](https://github.com/omkumar01/rag-aio) platform.

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

## Overview

This package answers two questions the data plane asks: *"which provider/model
should I use for this role?"* and *"what are the endpoints and auth refs for
that provider?"* It is deliberately thin — it describes, routes, and resolves,
but never performs HTTP itself. Transport adapters live in `rag-generation`
(which uses `httpx`); this package only declares the `HealthProbe` contract
that such adapters implement.

Configuration is a typed Pydantic hierarchy loaded from YAML/JSON: a
`ProvidersConfig` holds a list of `ProviderConfig`, each with `ModelConfig`
entries, `default_models` (role → model mapping), and a `SecretRef`.
`ProviderRegistry` indexes that config and exposes lookup + caching; `ModelRouter`
turns a logical role into a concrete `(provider, model_id)` decision with an
audit trail of every candidate considered.

## Architecture / Design Principles

- **Control plane only.** Endpoint URLs, model metadata, capabilities, aliases,
  default-model routing, and health *state* live here. Actual API calls are the
  consumer's job (`rag-generation` provides the `httpx` adapters). This keeps
  the dependency footprint to `rag-core` alone — **zero HTTP transit**.
- **Secrets by reference (ADR-0004/0006).** A `SecretRef` is structurally
  incapable of holding a credential: `kind` is `env`/`file`/`none` and `ref` is a
  variable name / file path. `resolve_secret` reads the value on demand and
  never writes it back onto the model. `SecretRef.__repr__` emits only the
  reference, so logs and API responses are always safe.
- **Strict, validated configuration.** `ProviderConfig` enforces unique model
  ids within a provider and validates that every `default_models` entry points
  at a declared `model_id`. `ProvidersConfig` rejects duplicate provider names.
  Models are `RagBaseModel` (`extra="forbid"`) so drift fails loudly.
- **Capability-gated routing.** `ModelRouter.route` checks
  `required_capabilities` against `ModelConfig.capabilities` before selecting a
  model, so a role can require (for example) vision or tool-calling support.
- **Deterministic resolution with fallback.** Routing order is: explicit
  `prefer` → role-based default → named fallback chain → `ProviderError`. Every
  candidate examined is recorded in `RouteDecision.attempted` for auditing.
- **Health as a cached side-channel.** Health state is cached *inside the
  registry* (never on the immutable config) and refreshed by an injected
  `HealthProbe`. This decouples "does the config say this provider exists?" from
  "is it reachable right now?" without coupling the package to a transport.
- **Local-first default profile.** `ProvidersConfig.local_default()` ships an
  LM-Studio (OpenAI-compatible) provider on `localhost:1234/v1` with no secret,
  so local development and CI are credential-free (ADR-0004).
- **Async routing contract.** `ModelRouter.route` is `async` even though
  resolution is currently in-memory-synchronous. The contract is async so
  future dynamic capability/health checks can be added without breaking callers.

## Public API

### Secrets (`secrets.py`)

- `SecretRef` — a `RagBaseModel` with `kind: Literal["env", "file", "none"]`
  and `ref: str | None`. Forbids extra fields (`ConfigDict(extra="forbid")`) and
  is repr-safe by design (it only ever holds the reference, never the value).
  `SecretRef.none()` builds the no-op reference used by local providers.
- `resolve_secret(ref) -> str | None` — resolves env / file / none. Missing env
  vars and missing files raise :class:`~rag_core.ConfigError` with a clear
  message. The returned string is never stored back onto the `SecretRef`.
- `mask_secret(value, keep=4) -> str` — renders `sk-...abcd`-style masks for
  logs; returns `"<unset>"` for `None`/empty, the raw value when shorter than
  `keep`.

```python
from rag_llm_provider import SecretRef, resolve_secret, mask_secret

ref = SecretRef(kind="env", ref="OPENAI_API_KEY")  # holds only the name
value = resolve_secret(ref)  # reads os.environ[...]
print(mask_secret(value))  # -> "sk-...wxyz"
print(repr(ref))  # -> SecretRef(kind='env', ref='OPENAI_API_KEY')
```

A `SecretRef` is structurally incapable of holding a credential, so its
`repr`/`str`/`model_dump` are always log- and wire-safe.

### Configuration (`config.py`)

- `ModelConfig` — `model_id`, `roles`, `context_window`, `max_output_tokens`,
  `embedding_dim`, `capabilities`, `temperature`, `aliases`.
- `ProviderConfig` — `name`, `kind: ProviderKind`, `base_url` (defaults per
  kind when omitted), `secret: SecretRef`, `models`, `default_models`,
  `enabled`, `timeout_s` (default `120.0`), `max_retries` (default `2`).
  Validates unique model ids and that `default_models` only references declared
  model ids.
- `ProvidersConfig` — `providers: list[ProviderConfig]` plus a
  `ProvidersConfig.local_default()` classmethod.
- `DEFAULT_BASE_URLS` — per-`ProviderKind` default endpoints.

```python
from rag_llm_provider import ProvidersConfig

config = ProvidersConfig(
    providers=[
        ProviderConfig(
            name="openai",
            kind="openai",
            secret=SecretRef(kind="env", ref="OPENAI_API_KEY"),
            models=[
                ModelConfig(
                    model_id="gpt-4o",
                    roles=["generate"],
                    context_window=128000,
                    capabilities=["vision"],
                ),
            ],
            default_models={"generate": "gpt-4o"},
        )
    ]
)
```

### Registry (`registry.py`)

- `ProviderRegistry(config)` — indexes providers by name, caches per-provider
  health.
  - `get(name)` → `ProviderConfig` (raises `ConfigError` if unknown).
  - `find_model(name_or_alias)` — searches enabled providers in config order,
    model id first then alias.
  - `resolve_role(role)` — first model (config order) whose `roles` include it.
  - `to_provider_info(provider) -> ProviderInfo` — projects to the public
    `rag_core` schema; exposes only `auth_ref` (the reference name), never a
    secret value, plus the cached `health` state.
  - `names()`, `enabled()`, `set_health(name, state)`.
- `HealthProbe` — a `Protocol` with `async def probe(provider) -> HealthState`.
  Consumers inject a concrete probe (e.g. an `httpx`-based one); this package
  declares only the contract.
- `RegistryHealth(registry)` — `async def check_all(probe) -> dict[str,
  HealthState]`, updating the registry's cached health for each provider.

```python
from rag_llm_provider import ProviderRegistry, RegistryHealth, HealthProbe

registry = ProviderRegistry(config)
provider = registry.get("openai")  # raises ConfigError if unknown


class MyProbe(HealthProbe):
    async def probe(self, provider) -> HealthState: ...


states = await RegistryHealth(registry).check_all(MyProbe())
```

### Routing (`routing.py`, `aliases.py`)

- `expand_alias(registry, name) -> str | None` — canonicalizes a model id or
  alias to its `model_id` (or `None`).
- `ModelRouter(registry, fallbacks=None)` —
  `async def route(role, prefer=None, required_capabilities=None) -> RouteDecision`.
  Resolution order:
  1. `prefer` (explicit id/alias) satisfying capabilities,
  2. role resolution over declared `roles`,
  3. `fallbacks[role]` chain in order,
  4. else `ProviderError("no model for role ...")`.
  - `RouteDecision(provider, model_id, fallback_used, attempted)`.
  - `route` is async despite currently synchronous resolution: the contract is
    async so future dynamic capability/health checks can be added without
    breaking callers.

```python
from rag_llm_provider import ModelRouter

router = ModelRouter(registry, fallbacks={"generate": ["gpt-4o-mini", "claude-3-haiku"]})
decision = await router.route("generate", prefer="gpt-4o", required_capabilities=["vision"])
print(decision.provider, decision.model_id, decision.fallback_used)
print(decision.attempted)  # audit trail of every candidate examined
```

## Configuration

All configuration is a `rag_core.base.RagBaseModel` (strict, `extra="forbid"`),
so it round-trips through YAML/JSON:

```python
import yaml
from rag_llm_provider import ProvidersConfig

config = ProvidersConfig.model_validate(yaml.safe_load(open("providers.yaml")))
```

`default_models` maps a *role* (e.g. `"generate"`, `"embed"`) to a `model_id`,
giving role-based defaults. `aliases` on `ModelConfig` let callers refer to a
model by a friendly name that `expand_alias` / `find_model` canonicalize.

Providers with `enabled=False` are skipped by `find_model`, `resolve_role`,
and `enabled()`.

### Default profile (ADR-0004)

The platform ships a local-first default via `ProvidersConfig.local_default()`:
a single **OpenAI-compatible** provider targeting LM Studio at
`http://localhost:1234/v1` with no configured secret. This keeps local
development and CI credential-free; cloud providers (OpenAI, Anthropic, Gemini)
are opt-in via an `env`/`file` `SecretRef`.

Default endpoints per `ProviderKind`:

| kind                | default `base_url`                              |
| ------------------- | ----------------------------------------------- |
| `openai`            | `https://api.openai.com/v1`                     |
| `anthropic`         | `https://api.anthropic.com`                     |
| `gemini`            | `https://generativelanguage.googleapis.com`     |
| `openai_compatible` | `http://localhost:1234/v1` (LM Studio)          |
| `ollama`            | `http://localhost:11434`                        |
| `vllm`              | `http://localhost:8000/v1`                      |
| `llamacpp`          | `http://localhost:8080`                         |

## Usage Guides

### Beginner — local-first default

`ProvidersConfig.local_default()` returns a credential-free LM-Studio profile,
so you can get a registry and router running immediately:

```python
from rag_llm_provider import ProvidersConfig, ProviderRegistry, ModelRouter

config = ProvidersConfig.local_default()
registry = ProviderRegistry(config)
router = ModelRouter(registry)
decision = await router.route("generate")
print(decision.provider, decision.model_id)  # "lm-studio" "qwen3-8b"
```

### Intermediate — cloud provider with env secret + routing

```python
import asyncio
from rag_llm_provider import (
    ProvidersConfig,
    ProviderConfig,
    ModelConfig,
    SecretRef,
    ProviderRegistry,
    ModelRouter,
)

config = ProvidersConfig(
    providers=[
        ProviderConfig(
            name="openai",
            kind="openai",
            secret=SecretRef(kind="env", ref="OPENAI_API_KEY"),
            models=[
                ModelConfig(model_id="gpt-4o", roles=["generate"], capabilities=["vision"]),
                ModelConfig(model_id="gpt-4o-mini", roles=["generate"], aliases=["mini"]),
            ],
            default_models={"generate": "gpt-4o-mini"},
        )
    ]
)

registry = ProviderRegistry(config)
router = ModelRouter(registry, fallbacks={"generate": ["gpt-4o-mini", "gpt-4o"]})


async def main():
    # explicit preference, capability-gated
    d = await router.route("generate", prefer="gpt-4o", required_capabilities=["vision"])
    print(d.provider, d.model_id, d.fallback_used)
    print("attempted:", d.attempted)

    # role-based default fallback
    d = await router.route("generate")
    print(d.provider, d.model_id, d.fallback_used)


asyncio.run(main())
```

### Advanced — multiple providers, disabled entries, alias resolution

```python
from rag_llm_provider import (
    ProvidersConfig,
    ProviderConfig,
    ModelConfig,
    ProviderRegistry,
    ModelRouter,
    expand_alias,
)

config = ProvidersConfig.local_default()
config.providers.append(
    ProviderConfig(
        name="anthropic",
        kind="anthropic",
        secret=SecretRef(kind="env", ref="ANTHROPIC_API_KEY"),
        models=[ModelConfig(model_id="claude-3-5-sonnet", roles=["generate"])],
        default_models={"generate": "claude-3-5-sonnet"},
    )
)
registry = ProviderRegistry(config)  # re-index after mutation

# alias resolution canonicalizes friendly names to model ids
canonical = expand_alias(registry, "mini")  # -> a model_id or None

# disabled providers are invisible to lookups:
#   registry.enabled()               # excludes disabled entries
#   registry.find_model("gpt-4o")    # skips disabled providers

info = registry.to_provider_info(registry.get("anthropic"))
print(info.auth_ref)  # "ANTHROPIC_API_KEY" — never the resolved value
```

## Installation

```bash
uv pip install rag-llm-provider
```

## Testing

Tests are offline and never contact a real provider. Secrets are exercised via
`monkeypatch` (env) and `tmp_path` (files); provider config validation uses
`model_validate` + `pytest.raises(ConfigError)`; routing uses a hand-built
in-memory registry.

```bash
uv run pytest packages/rag-llm-provider -q
```

## Dependencies

- `rag-core` **only** — `RagBaseModel`, the `ProviderKind`/`HealthState`/`ModelInfo`/
  `ProviderInfo` info schemas, and the `ConfigError` / `ProviderError` taxonomy.
  No `httpx`, no vendor SDK. Concrete HTTP adapters and probes are injected by
  consumers (`rag-generation`, `rag-embedder`, etc.).

## Cross-Package Relationships

- **rag-core** — defines the info schemas (`ProviderKind`, `ModelInfo`,
  `ProviderInfo`, `HealthState`) that `Config`/`ModelConfig`/`ProviderConfig` and
  `to_provider_info` project onto, plus the error taxonomy. This is a pure
  control-plane layer over those contracts (ADR-0002/0006).
- **rag-generation** — consumes `ProviderInfo` (via `to_provider_info`) to build
  its `httpx`-backed `OpenAIProvider` / `AnthropicProvider` / `GeminiProvider`
  adapters, and routes generation through `ModelRouter` / `default_models`.
- **rag-embedder** — uses `ProviderRegistry` + `ModelRouter` to pick an embedding
  model (the `"embed"` role) and reads its `embedding_dim` / `base_url` /
  `auth_ref` to construct an embedding client.
- **rag-rerank** — resolves the `"rerank"` role (and optionally the `"ocr"` /
  `"vlm"` roles) through the same routing machinery.
- **rag-query** — its LLM-dependent strategies accept a duck-typed `LLM`
  (`async generate(request) -> GenerationResult`); a consumer typically builds
  that generator from a provider selected via this package's `ModelRouter`.
- **rag-orchestrator** — owns top-level composition: it loads `ProvidersConfig`,
  builds the `ProviderRegistry`, wires `ModelRouter` defaults, and injects both
  into the data-plane modules.

## Verification

```bash
uv run ruff format packages/rag-llm-provider && uv run ruff check packages/rag-llm-provider
uv run mypy packages/rag-llm-provider/src
uv run pytest packages/rag-llm-provider -q
```

## Layout

```
src/rag_llm_provider/
  __init__.py      public API
  py.typed         PEP 561 marker
  secrets.py       SecretRef, resolve_secret, mask_secret, SecretKind
  config.py        ModelConfig, ProviderConfig, ProvidersConfig, DEFAULT_BASE_URLS
  registry.py      ProviderRegistry, HealthProbe, RegistryHealth
  routing.py       ModelRouter, RouteDecision
  aliases.py       expand_alias
tests/
  conftest.py       build_config fixture
  test_secrets.py   env/file/none resolution + masking
  test_config.py    validation, defaults, local_default
  test_registry.py  lookup, alias, role, provider_info projection
  test_router.py    prefer / role / fallback / capability gating
  test_health.py    RegistryHealth aggregator
  test_aliases.py   expand_alias
  test_smoke.py     import + version
```
