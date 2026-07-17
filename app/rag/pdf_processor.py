"""
CodeMentor AI - PDF Document Processor
========================================
Extracts text from uploaded PDF files using PyPDF2 and pypdf.

Design Rationale:
- Two-library strategy: pypdf is primary (actively maintained), PyPDF2 as fallback.
- Page-level extraction preserves metadata (page numbers) for citation.
- Graceful handling of encrypted/corrupted PDFs.
- Text content type detection helps the chunker apply the right strategy.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from app.core.exceptions import ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedDocument:
    """
    Result of PDF/text extraction.

    Attributes:
        filename:    Original file name.
        pages:       List of text content per page.
        full_text:   Concatenated text of all pages.
        page_count:  Total number of pages.
        metadata:    File-level metadata (title, author, etc.)
    """

    filename: str
    pages: list[str] = field(default_factory=list)
    page_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Returns all page text joined with double newline."""
        return "\n\n".join(self.pages)

    @property
    def is_empty(self) -> bool:
        return not any(p.strip() for p in self.pages)


class PDFProcessor:
    """
    Extracts text from PDF files for RAG ingestion.

    Supports:
    - Regular text PDFs (books, papers, docs)
    - Multi-page documents with page-level metadata
    - Plain text files (.txt, .md)
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst"}

    def extract_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> ExtractedDocument:
        """
        Extract text from file bytes (from FastAPI UploadFile.read()).

        Args:
            file_bytes: Raw file bytes.
            filename:   Original filename (used to determine extraction strategy).

        Returns:
            ExtractedDocument with pages and metadata.

        Raises:
            ValidationError: If file type is unsupported or extraction fails.
        """
        suffix = Path(filename).suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file type: '{suffix}'. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if suffix == ".pdf":
            return self._extract_pdf(file_bytes, filename)
        else:
            return self._extract_text(file_bytes, filename)

    def _extract_pdf(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """
        Extract text from PDF using pypdf (primary) with PyPDF2 fallback.
        """
        doc = ExtractedDocument(filename=filename)

        # --- Primary: pypdf ---
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))

            if reader.is_encrypted:
                raise ValidationError(
                    f"PDF '{filename}' is encrypted/password-protected. "
                    "Please upload an unlocked version."
                )

            doc.page_count = len(reader.pages)

            # Extract PDF metadata
            pdf_meta = reader.metadata or {}
            doc.metadata = {
                "title": pdf_meta.get("/Title", filename),
                "author": pdf_meta.get("/Author", "Unknown"),
                "subject": pdf_meta.get("/Subject", ""),
                "creator": pdf_meta.get("/Creator", ""),
                "page_count": doc.page_count,
                "filename": filename,
            }

            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    doc.pages.append(text)
                except Exception as e:
                    logger.warning(
                        "Could not extract page %d from '%s': %s",
                        page_num, filename, e
                    )
                    doc.pages.append("")  # Preserve page count alignment

            logger.info(
                "PDF extracted via pypdf: file=%s pages=%d",
                filename, doc.page_count
            )

        except ImportError:
            # --- Fallback: PyPDF2 ---
            logger.warning("pypdf not available, falling back to PyPDF2.")
            try:
                import PyPDF2

                reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                doc.page_count = len(reader.pages)
                doc.metadata = {"filename": filename, "page_count": doc.page_count}

                for page in reader.pages:
                    text = page.extract_text() or ""
                    doc.pages.append(text)

                logger.info(
                    "PDF extracted via PyPDF2: file=%s pages=%d",
                    filename, doc.page_count
                )

            except Exception as exc:
                raise ValidationError(
                    f"Failed to extract PDF '{filename}': {exc}"
                ) from exc

        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"Unexpected error extracting PDF '{filename}': {exc}"
            ) from exc

        if doc.is_empty:
            raise ValidationError(
                f"PDF '{filename}' contains no extractable text. "
                "It may be a scanned image PDF — please use a text-based PDF."
            )

        return doc

    def _extract_text(self, file_bytes: bytes, filename: str) -> ExtractedDocument:
        """
        Extract text from plain text files (.txt, .md, .rst).
        """
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as exc:
                raise ValidationError(
                    f"Could not decode text file '{filename}': {exc}"
                ) from exc

        doc = ExtractedDocument(
            filename=filename,
            pages=[text],
            page_count=1,
            metadata={"filename": filename, "page_count": 1},
        )

        logger.info("Text file extracted: file=%s chars=%d", filename, len(text))
        return doc
