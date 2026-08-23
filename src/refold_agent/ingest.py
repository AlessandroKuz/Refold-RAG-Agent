"""Document ingestion pipeline for Refold markdown documents."""

import logging
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def load_markdown_file(file_path: Path) -> str:
    """Read a markdown file content using UTF-8 encoding."""
    return file_path.read_text(encoding="utf-8")


def process_markdown_file(
    file_path: Path,
    header_splitter: MarkdownHeaderTextSplitter,
    text_splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    """Process a single markdown file through header and character splitting."""
    content = load_markdown_file(file_path)
    filename = file_path.name

    # Step 1: Split on markdown headers to capture semantic structure
    header_docs = header_splitter.split_text(content)

    if not header_docs:
        # Fallback if file has no markdown headers
        header_docs = [Document(page_content=content, metadata={})]

    # Step 2: Split large sections recursively while preserving header metadata
    chunked_docs = text_splitter.split_documents(header_docs)

    # Attach file origin metadata to each chunk
    for doc in chunked_docs:
        doc.metadata["source"] = filename
        doc.metadata["file_path"] = str(file_path)

    return chunked_docs


def ingest_resources(
    resources_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Discover all markdown files in resources_dir and produce split documents."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    all_docs: list[Document] = []
    markdown_files = sorted(resources_dir.glob("*.md"))

    if not markdown_files:
        logger.warning("No markdown files found in %s", resources_dir)
        return all_docs

    logger.info("Found %d markdown files in %s", len(markdown_files), resources_dir)

    for md_file in markdown_files:
        try:
            chunks = process_markdown_file(md_file, header_splitter, text_splitter)
            all_docs.extend(chunks)
        except Exception as e:
            logger.error("Failed to process file %s: %s", md_file, e)

    logger.info("Ingestion complete. Total chunks generated: %d", len(all_docs))
    return all_docs


def build_bm25_retriever(
    docs: list[Document],
    k: int = 10,
) -> BM25Retriever | None:
    """Build a BM25 keyword retriever from split documents."""
    if not docs:
        return None
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever

