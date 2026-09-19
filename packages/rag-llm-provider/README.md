# rag-llm-provider

Provider and model management control layer: provider/model definitions, capabilities,
context limits, embedding dimensions, pricing metadata, health state, endpoint config,
authentication references, routing preferences, aliases, fallback chains, and per-model
generation settings. Supports OpenAI, Anthropic, Google/Gemini, OpenAI-compatible servers
(LM Studio, vLLM, Ollama, llama.cpp). Secrets are stored by reference and never exposed.
