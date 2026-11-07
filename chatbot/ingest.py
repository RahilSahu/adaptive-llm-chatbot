"""Document ingestion: turn owner-supplied files into searchable chunks.

Supported formats: .txt, .md, .pdf, .docx. Anything else is rejected
loudly rather than silently producing an empty knowledge base, which
is the failure mode that makes document chatbots look stupid in demos.

Text is split into overlapping chunks: small enough to retrieve
precisely, overlapped so an answer that straddles a boundary is not
lost. Each chunk remembers its source file for citations.
"""

from __future__ import annotations

import io

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class IngestError(ValueError):
    """Raised for unsupported, oversized or unreadable uploads."""


def _extension(filename: str) -> str:
    name = filename.lower()
    for ext in SUPPORTED_EXTENSIONS:
        if name.endswith(ext):
            return ext
    return ""


def extract_text(filename: str, data: bytes) -> str:
    """Pull plain text out of one uploaded file."""
    if len(data) > MAX_FILE_BYTES:
        raise IngestError(f"{filename}: file is larger than 10 MB")
    ext = _extension(filename)
    if ext in (".txt", ".md"):
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            raise IngestError(f"{filename}: could not read PDF ({exc})") from exc
    if ext == ".docx":
        import docx

        try:
            document = docx.Document(io.BytesIO(data))
            parts = [p.text for p in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            return "\n".join(parts)
        except Exception as exc:
            raise IngestError(f"{filename}: could not read DOCX ({exc})") from exc
    raise IngestError(
        f"{filename}: unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]
    except ImportError:  # pragma: no cover - fallback if the extra is missing
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start : start + CHUNK_SIZE].strip())
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return [c for c in chunks if c]
