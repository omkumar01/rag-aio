# rag-generation

Provider-agnostic LLM generation for rag-aio: normal and streaming generation, structured
output, tool calling where supported, model routing, retries, timeout policies, fallback
models, usage/cost tracking, and capability discovery. HTTP-native adapters (httpx) for
OpenAI, Anthropic, Google/Gemini, and OpenAI-compatible endpoints (LM Studio, vLLM,
Ollama, llama.cpp servers). Provider-specific request/response models never leak into
rag-core.
