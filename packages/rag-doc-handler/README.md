# rag-doc-handler

Unified ingestion and parsing for rag-aio: MIME/type detection, a plugin parser registry
covering PDF (fast PyMuPDF path; Docling high-fidelity path behind an extra), HTML, Markdown,
DOCX, XLSX, PPTX, RTF (extra), EML (stdlib) / MSG (extra), and images (delegating to
`rag-ocr`). Includes sitemap/website crawling with SSRF guards, rate limiting, and crawl
limits; stable content hashes; duplicate detection; and canonical `Document` output.
