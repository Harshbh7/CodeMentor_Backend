"""
CodeMentor AI - Text Chunking Engine
======================================
Splits raw text and PDF content into optimal chunks for embedding.

Design Rationale:
- Chunk size 1000 chars with 200 overlap: balances context richness vs. noise.
- RecursiveCharacterTextSplitter tries paragraph → sentence → word boundaries,
  producing semantically coherent chunks instead of hard cuts.
- Metadata is preserved per-chunk so we can filter by source, language, topic.
- Different strategies for code vs. prose (code needs larger chunks to preserve logic).
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.text_splitter import (
    Language,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """
    Represents a single chunk of text ready for embedding.

    Attributes:
        content:    The actual text content of the chunk.
        metadata:   Arbitrary key-value metadata (source, page, language, etc.)
        chunk_id:   Stable SHA-256 hash of content — used as ChromaDB document ID.
        chunk_index: Position of this chunk in the original document.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(init=False)
    chunk_index: int = 0

    def __post_init__(self) -> None:
        # Deterministic ID based on content — prevents duplicate embeddings
        self.chunk_id = hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:32]

    def __len__(self) -> int:
        return len(self.content)


class TextChunker:
    """
    Splits documents into optimally-sized chunks for RAG.

    Supports:
    - General prose (documentation, notes, articles)
    - Source code (Python, Java, C++, JavaScript, SQL)
    - PDF-extracted text (handles page artifacts)
    """

    # Optimal chunk sizes by content type
    PROSE_CHUNK_SIZE = 1000
    PROSE_CHUNK_OVERLAP = 200
    CODE_CHUNK_SIZE = 1500       # Code needs larger chunks to preserve function context
    CODE_CHUNK_OVERLAP = 300

    # Language detection patterns
    LANGUAGE_PATTERNS: dict[str, list[str]] = {
        "python": [r"def\s+\w+", r"import\s+\w+", r"class\s+\w+:", r"print\("],
        "java": [r"public\s+class", r"public\s+static\s+void", r"import\s+java\."],
        "cpp": [r"#include\s*<", r"int\s+main\s*\(", r"std::", r"cout\s*<<"],
        "javascript": [r"const\s+\w+\s*=", r"function\s+\w+", r"console\.log", r"=>"],
        "sql": [r"SELECT\s+", r"FROM\s+", r"INSERT\s+INTO", r"CREATE\s+TABLE"],
    }

    def __init__(self) -> None:
        # General prose splitter
        self._prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.PROSE_CHUNK_SIZE,
            chunk_overlap=self.PROSE_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Code-aware splitter (respects function/class boundaries)
        self._code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CODE_CHUNK_SIZE,
            chunk_overlap=self.CODE_CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        logger.info("TextChunker initialized.")

    def detect_language(self, text: str) -> str | None:
        """
        Heuristically detect if text contains source code and which language.

        Args:
            text: Raw text to analyze.

        Returns:
            Language name if detected, None for prose.
        """
        sample = text[:2000]  # Only check first 2000 chars
        scores: dict[str, int] = {}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            matches = sum(
                1 for pattern in patterns if re.search(pattern, sample, re.IGNORECASE)
            )
            if matches > 0:
                scores[lang] = matches

        if not scores:
            return None

        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] >= 2 else None

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing common PDF artifacts and noise.

        Args:
            text: Raw text (often from PDF extraction).

        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace / blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page number artifacts like "Page 1 of 20"
        text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
        # Remove header/footer repetitions (lines shorter than 5 chars on their own line)
        text = re.sub(r"^\s*.{1,4}\s*$", "", text, flags=re.MULTILINE)
        # Normalize Unicode dashes and quotes
        text = text.replace("\u2013", "-").replace("\u2014", "--")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return text.strip()

    def chunk_text(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        force_language: str | None = None,
    ) -> list[TextChunk]:
        """
        Split raw text into chunks with metadata.

        Args:
            text:           Raw document text.
            metadata:       Base metadata to attach to every chunk (e.g., source file, topic).
            force_language: Override language detection (useful for code files).

        Returns:
            List of TextChunk objects ready for embedding.
        """
        if not text or not text.strip():
            logger.warning("chunk_text: Received empty text, returning no chunks.")
            return []

        base_metadata = metadata or {}
        cleaned = self.clean_text(text)

        # Detect if this is code
        detected_lang = force_language or self.detect_language(cleaned)
        is_code = detected_lang is not None

        splitter = self._code_splitter if is_code else self._prose_splitter

        if detected_lang:
            base_metadata["detected_language"] = detected_lang
            logger.debug("Detected language: %s — using code splitter.", detected_lang)

        raw_chunks = splitter.split_text(cleaned)

        chunks: list[TextChunk] = []
        for idx, raw_chunk in enumerate(raw_chunks):
            if not raw_chunk.strip():
                continue

            chunk_meta = {
                **base_metadata,
                "chunk_index": idx,
                "chunk_total": len(raw_chunks),
                "char_count": len(raw_chunk),
                "is_code": is_code,
            }

            chunks.append(
                TextChunk(
                    content=raw_chunk.strip(),
                    metadata=chunk_meta,
                    chunk_index=idx,
                )
            )

        logger.info(
            "Chunked text into %d chunks (language=%s, is_code=%s).",
            len(chunks),
            detected_lang or "prose",
            is_code,
        )
        return chunks

    def chunk_pages(
        self,
        pages: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """
        Chunk a list of page texts (from PDF extraction), preserving page numbers.

        Args:
            pages:    List of strings, one per PDF page.
            metadata: Base metadata to attach to every chunk.

        Returns:
            List of TextChunk objects with page_number in metadata.
        """
        all_chunks: list[TextChunk] = []
        global_index = 0

        for page_num, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue

            page_metadata = {
                **(metadata or {}),
                "page_number": page_num,
                "total_pages": len(pages),
            }

            page_chunks = self.chunk_text(page_text, metadata=page_metadata)

            for chunk in page_chunks:
                chunk.chunk_index = global_index
                global_index += 1
                all_chunks.append(chunk)

        logger.info("Chunked %d pages into %d total chunks.", len(pages), len(all_chunks))
        return all_chunks
