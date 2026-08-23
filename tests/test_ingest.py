"""Tests for document ingestion and markdown chunking pipeline."""

from refold_agent.ingest import ingest_resources, process_markdown_file
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from refold_agent.ingest import HEADERS_TO_SPLIT_ON


def test_process_markdown_file(sample_markdown_dir):
    """Test header preservation and source metadata attachment."""
    file_path = sample_markdown_dir / "phase0-test.md"
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = process_markdown_file(file_path, header_splitter, text_splitter)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "phase0-test.md"
        assert "file_path" in chunk.metadata

    # Check header hierarchy captured
    h1_values = [c.metadata.get("Header 1") for c in chunks if "Header 1" in c.metadata]
    assert any("Phase 0 — Immersion" in str(h) for h in h1_values)


def test_ingest_resources(sample_markdown_dir):
    """Test full directory ingestion discovering all markdown files."""
    docs = ingest_resources(sample_markdown_dir, chunk_size=500, chunk_overlap=50)
    assert len(docs) >= 4
    sources = {doc.metadata["source"] for doc in docs}
    assert "phase0-test.md" in sources
    assert "phase1-test.md" in sources


def test_ingest_empty_dir(temp_dir):
    """Test graceful handling of empty resources directory."""
    empty_dir = temp_dir / "empty_resources"
    empty_dir.mkdir()
    docs = ingest_resources(empty_dir)
    assert docs == []
