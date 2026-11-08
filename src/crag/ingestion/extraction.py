"""Safe extraction for supported document formats."""

from dataclasses import dataclass
from io import BytesIO
from zipfile import BadZipFile, ZipFile


class DocumentExtractionError(ValueError):
    """Raised when an uploaded document cannot be safely extracted."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    file_format: str
    metadata: dict[str, str | int]


def extract_document(filename: str, content: bytes) -> ExtractedDocument:
    """Detect and extract PDF, DOCX, or UTF-8 text content."""
    if not content:
        raise DocumentExtractionError("The uploaded file is empty.")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if content.startswith(b"%PDF-"):
        return _extract_pdf(content)
    if content.startswith(b"PK\x03\x04") and _looks_like_docx(content):
        return _extract_docx(content)
    if suffix == "txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentExtractionError("TXT files must use UTF-8 encoding.") from exc
        return _result(text, "txt", {})
    raise DocumentExtractionError("Unsupported or unrecognized document format.")


def _extract_pdf(content: bytes) -> ExtractedDocument:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise DocumentExtractionError("Encrypted PDFs are not supported.")
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except DocumentExtractionError:
        raise
    except Exception as exc:
        raise DocumentExtractionError("The PDF could not be read.") from exc
    return _result("\n\n".join(pages), "pdf", {"page_count": len(pages)})


def _extract_docx(content: bytes) -> ExtractedDocument:
    try:
        from docx import Document

        document = Document(BytesIO(content))
        parts = [paragraph.text for paragraph in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
    except Exception as exc:
        raise DocumentExtractionError("The DOCX file could not be read.") from exc
    return _result("\n".join(parts), "docx", {"paragraph_count": len(parts)})


def _looks_like_docx(content: bytes) -> bool:
    try:
        with ZipFile(BytesIO(content)) as archive:
            return "[Content_Types].xml" in archive.namelist() and any(
                name.startswith("word/") for name in archive.namelist()
            )
    except BadZipFile:
        return False


def _result(
    text: str, file_format: str, metadata: dict[str, str | int]
) -> ExtractedDocument:
    normalized = "\n".join(" ".join(line.split()) for line in text.splitlines())
    normalized = "\n".join(line for line in normalized.splitlines() if line).strip()
    if not normalized:
        raise DocumentExtractionError(
            "The document contains no extractable text. Scanned PDFs require OCR."
        )
    return ExtractedDocument(normalized, file_format, metadata)
