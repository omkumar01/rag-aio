"""Declarative pipeline definition + registry.

A :class:`Pipeline` is a *declarative spec*: it pairs a :class:`PipelineConfig`
with a lightweight registry of **component identifiers** (names, not live
objects). It is *not* the runner — the runtime lives in
:mod:`rag_orchestrator.orchestrator` and component construction lives in
:mod:`rag_orchestrator.services`. The registry only records *what* is bound so
the spec can be serialized, versioned and audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_core.ids import new_id

from .config import PipelineConfig

__all__ = ["Pipeline", "PipelineRegistry", "default_pipeline", "default_registry"]

DEFAULT_VERSION = "1"


def _prepare_yaml_path(path: str) -> Path:
    """Normalize a caller-supplied YAML target before opening it for writing.

    Resolves symlinks/relative segments so the write target is a concrete
    filesystem location, and rejects directories instead of failing inside
    ``open()``.
    """
    resolved = Path(path).resolve()
    if resolved.is_dir():
        raise ValueError(f"yaml target is a directory: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


@dataclass
class Pipeline:
    """Declarative pipeline: config + a map of stage -> component identifier.

    Attributes:
        config: The versioned :class:`PipelineConfig`.
        components: Mapping of stage name to a component identifier string
            (e.g. ``"HybridRetriever[rrf]"``). These are lazy labels, not live
            objects, so a ``Pipeline`` is cheap to serialize/store.
        version: Optional explicit version tag.
        id: Unique pipeline instance id.
        created_at: When the spec was materialized.
    """

    config: PipelineConfig
    components: dict[str, str] = field(default_factory=dict)
    version: str | None = None
    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_config(cls, config: PipelineConfig, services: object | None = None) -> Pipeline:
        """Materialize a :class:`Pipeline` from a config.

        When ``services`` is supplied its component types are recorded as
        identifiers (best-effort) so the spec is auditable; no live component is
        constructed here.
        """
        components: dict[str, str] = {}
        if services is not None:
            for stage in ("retriever", "reranker", "generator", "embedder", "chunker"):
                obj = getattr(services, stage, None) or getattr(services, f"{stage}s", None)
                if obj is None:
                    continue
                if isinstance(obj, list):
                    names = [type(item).__name__ for item in obj if item is not None]
                    components[stage] = ",".join(names)
                else:
                    components[stage] = type(obj).__name__
        return cls(config=config, components=components, version=DEFAULT_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON/YAML-friendly dict."""
        config_dump = self.config.model_dump(mode="json")
        return {
            "id": self.id,
            "version": self.version or DEFAULT_VERSION,
            "schema_version": self.config.schema_version,
            "created_at": self.created_at.isoformat(),
            "config": config_dump,
            "components": dict(self.components),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Deserialize from a dict (inverse of :meth:`to_dict`)."""
        config = PipelineConfig(**data["config"])
        created_raw = data.get("created_at")
        created = datetime.fromisoformat(created_raw) if created_raw else datetime.now(UTC)
        return cls(
            config=config,
            components=dict(data.get("components", {})),
            version=data.get("version") or DEFAULT_VERSION,
            id=data.get("id", new_id()),
            created_at=created,
        )

    def save_yaml(self, path: str) -> None:
        """Persist this pipeline spec as YAML."""
        import yaml  # lazy: keep rag_orchestrator import cheap.

        target = _prepare_yaml_path(path)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str) -> Pipeline:
        """Load a pipeline spec from YAML."""
        import yaml  # lazy: keep rag_orchestrator import cheap.

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.from_dict(data)


class PipelineRegistry:
    """In-Memory registry of named pipelines + versions.

    Pipelines are keyed by ``"<name>:<version>"``. The registry is purely
    declarative — it stores specs, not live components.
    """

    def __init__(self) -> None:
        self._pipelines: dict[str, Pipeline] = {}

    @staticmethod
    def _key(name: str, version: str | None) -> str:
        return f"{name}:{version or DEFAULT_VERSION}"

    def add(self, pipeline: Pipeline, version: str | None = None) -> None:
        """Register a pipeline (replaces any existing same-name+version)."""
        if version is not None:
            pipeline.version = version
        key = self._key(pipeline.config.name, version or pipeline.version)
        self._pipelines[key] = pipeline

    def get(self, name: str, version: str | None = None) -> Pipeline | None:
        """Return a registered pipeline or ``None``."""
        return self._pipelines.get(self._key(name, version))

    def list(self) -> list[Pipeline]:
        """All registered pipelines."""
        return list(self._pipelines.values())

    def remove(self, name: str, version: str | None = None) -> bool:
        """Remove a pipeline; returns whether something was removed."""
        key = self._key(name, version or DEFAULT_VERSION)
        return self._pipelines.pop(key, None) is not None

    def save(self, path: str) -> None:
        """Persist every registered pipeline as a YAML list."""
        import yaml  # lazy.

        docs = [p.to_dict() for p in self._pipelines.values()]
        target = _prepare_yaml_path(path)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump_all(docs, fh, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> PipelineRegistry:
        """Load a registry previously written by :meth:`save`."""
        import yaml  # lazy.

        with open(path, encoding="utf-8") as fh:
            docs = list(yaml.safe_load_all(fh))
        registry = cls()
        for doc in docs or []:
            if doc:
                registry.add(Pipeline.from_dict(doc))
        return registry


def default_registry() -> PipelineRegistry:
    """A registry pre-populated with the local-default pipeline."""
    registry = PipelineRegistry()
    registry.add(Pipeline.from_config(PipelineConfig.local_default()))
    return registry


def default_pipeline() -> Pipeline:
    """The local-default pipeline spec."""
    return Pipeline.from_config(PipelineConfig.local_default())
