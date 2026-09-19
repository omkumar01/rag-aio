# rag-context

Context-engineering layer between retrieval and generation: deduplication, overlapping-chunk
merging, parent/neighbor expansion, section reconstruction, ranking, compression, token
budgeting against the target model's tokenizer, source grouping, citation mapping, and
multiple ordering strategies (relevance-first, chronological, section-aware,
document-grouped, diversity-aware). Every context item preserves exact source provenance;
retrieved content is isolated from system instructions.
