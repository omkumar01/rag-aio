"""Hatch build hook: bundle sibling workspace packages into the rag-aio wheel.

The published ``rag-aio`` wheel contains every ``rag_*`` package's code, so
users install one distribution. The sources live in the sibling workspace
members; this hook copies them into a temporary ``_bundle/`` staging directory
and force-includes it for **standard wheel builds only**.

Editable builds (``uv sync`` during development) intentionally skip the hook:
the workspace members themselves remain the source of truth, and bundling
copies into the dev environment would shadow them.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

SIBLINGS: dict[str, str] = {
    "rag_core": "../rag-core/src/rag_core",
    "rag_observe": "../rag-observe/src/rag_observe",
    "rag_cache": "../rag-cache/src/rag_cache",
    "rag_db_handler": "../rag-db-handler/src/rag_db_handler",
    "rag_doc_handler": "../rag-doc-handler/src/rag_doc_handler",
    "rag_ocr": "../rag-ocr/src/rag_ocr",
    "rag_embedder": "../rag-embedder/src/rag_embedder",
    "rag_retrieval": "../rag-retrieval/src/rag_retrieval",
    "rag_rerank": "../rag-rerank/src/rag_rerank",
    "rag_query": "../rag-query/src/rag_query",
    "rag_context": "../rag-context/src/rag_context",
    "rag_llm_provider": "../rag-llm-provider/src/rag_llm_provider",
    "rag_generation": "../rag-generation/src/rag_generation",
    "rag_orchestrator": "../rag-orchestrator/src/rag_orchestrator",
    "rag_mass_inject": "../rag-mass-inject/src/rag_mass_inject",
    "rag_eval": "../rag-eval/src/rag_eval",
    "rag_ui": "../rag-ui/src/rag_ui",
}

_BUNDLE = "_bundle"


class BundleBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version != "standard":
            return
        staging = Path(self.root) / _BUNDLE
        if staging.exists():
            shutil.rmtree(staging)
        for module, rel in SIBLINGS.items():
            shutil.copytree(Path(self.root) / rel, staging / module)
        for module in SIBLINGS:
            build_data["force_include"][(f"{_BUNDLE}/{module}")] = module

    def finalize(
        self,
        version: str,
        build_data: dict[str, Any],
        artifact: str,
    ) -> None:
        shutil.rmtree(Path(self.root) / _BUNDLE, ignore_errors=True)
