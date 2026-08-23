"""Tests for RAG Engine retrieval, filtering, citations, and fallback."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from refold_agent.rag import NO_CONTEXT_FALLBACK, RAGEngine


def test_format_sources_section():
    """Test citation formatting and deduplication."""
    doc1 = Document(
        page_content="Phase 1 content",
        metadata={"source": "phase1a.md", "Header 1": "Phase 1", "Header 2": "Basics"},
    )
    doc2 = Document(
        page_content="Phase 2 content",
        metadata={"source": "phase2a.md", "Header 1": "Phase 2"},
    )

    scored_docs = [(doc1, 0.85), (doc2, 0.72)]
    section_md, meta = RAGEngine.format_sources_section(scored_docs)

    assert "### Sources:" in section_md
    assert "- `phase1a.md` (Phase 1 > Basics)" in section_md
    assert "- `phase2a.md` (Phase 2)" in section_md
    assert len(meta) == 2
    assert meta[0]["score"] == 0.85


def test_build_context_string():
    """Test context string structure formatting."""
    doc = Document(
        page_content="Learn core vocabulary words.",
        metadata={"source": "phase2a.md", "Header 1": "Vocabulary"},
    )
    engine = MagicMock(spec=RAGEngine)
    context = RAGEngine.build_context_string(engine, [(doc, 0.9)])

    assert "--- Document 1: phase2a.md (Vocabulary) ---" in context
    assert "Learn core vocabulary words." in context


@pytest.mark.asyncio
async def test_answer_query_no_matching_chunks(test_settings):
    """Test strict out-of-scope fallback when no chunks pass the 0.5 threshold."""
    with patch("refold_agent.rag.Chroma"), patch("refold_agent.rag.ChatOllama"), patch("refold_agent.rag.OllamaEmbeddings"):
        engine = RAGEngine(test_settings)
        engine.retrieve_relevant_chunks = MagicMock(return_value=[])

        result = await engine.answer_query("How do I cook pasta?")

        assert result["has_context"] is False
        assert result["answer"] == NO_CONTEXT_FALLBACK
        assert result["sources"] == []
