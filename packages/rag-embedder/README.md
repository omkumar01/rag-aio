# rag-embedder

End-to-end transformation from parsed documents to indexed representations: chunking
strategies (fixed, sentence, token-aware, recursive, structural, parent/child, semantic,
hybrid), model-native tokenization, embedding strategy selection (FastEmbed default;
sentence-transformers extra), dense + sparse embeddings, batching and backpressure, an
explicit indexer abstraction over `rag-db-handler` stores, and incremental reindexing
driven by content/model/chunker configuration hashes.
