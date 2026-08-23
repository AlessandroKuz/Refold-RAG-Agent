"""Tests for RAG Engine retrieval, filtering, citations, intent routing, and fallback."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from langchain_core.documents import Document

from refold_agent.rag import (
    NO_CONTEXT_FALLBACK,
    IntentDecision,
    QueryIntent,
    RAGEngine,
)


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
async def test_classify_intent(test_settings):
    """Test structured intent routing decision."""
    with patch("refold_agent.rag.Chroma"), patch("refold_agent.rag.ChatOllama"), patch("refold_agent.rag.OllamaEmbeddings"), patch("refold_agent.rag.Ranker"):
        engine = RAGEngine(test_settings)
        engine.router_llm = MagicMock()
        engine.router_llm.ainvoke = AsyncMock(
            return_value=IntentDecision(intent=QueryIntent.GREETING)
        )

        decision = await engine.classify_intent("Hello!")
        assert decision == QueryIntent.GREETING


@pytest.mark.asyncio
async def test_answer_query_greeting_routing(test_settings):
    """Test GREETING intent returns persona answer without citations."""
    with patch("refold_agent.rag.Chroma"), patch("refold_agent.rag.ChatOllama"), patch("refold_agent.rag.OllamaEmbeddings"), patch("refold_agent.rag.Ranker"):
        engine = RAGEngine(test_settings)
        engine.classify_intent = AsyncMock(return_value=QueryIntent.GREETING)
        engine.llm = MagicMock()
        engine.llm.ainvoke = AsyncMock(
            return_value=MagicMock(content="Hello! I am the Refold Assistant.")
        )

        result = await engine.answer_query("Hi there!")

        assert result["has_context"] is True
        assert result["answer"] == "Hello! I am the Refold Assistant."
        assert result["sources"] == []


@pytest.mark.asyncio
async def test_answer_query_out_of_scope_routing(test_settings):
    """Test OUT_OF_SCOPE intent returns strict fallback without RAG retrieval."""
    with patch("refold_agent.rag.Chroma"), patch("refold_agent.rag.ChatOllama"), patch("refold_agent.rag.OllamaEmbeddings"), patch("refold_agent.rag.Ranker"):
        engine = RAGEngine(test_settings)
        engine.classify_intent = AsyncMock(return_value=QueryIntent.OUT_OF_SCOPE)

        result = await engine.answer_query("How do I make sourdough bread?")

        assert result["has_context"] is False
        assert result["answer"] == NO_CONTEXT_FALLBACK
        assert result["sources"] == []


def test_hybrid_search_and_rerank_keyword_match(test_settings):
    """Test BM25 + FlashRank surfaces exact keyword acronyms like CARA."""
    doc_cara = Document(
        page_content='The "Comfort first, then Accuracy" sequencing reflects the Refold CARA model.',
        metadata={"source": "phase4-speaking.md", "Header 1": "Speaking"},
    )
    doc_other = Document(
        page_content="Phase 1 Foundations overview.",
        metadata={"source": "phase1-phase-overview.md", "Header 1": "Foundations"},
    )

    with patch("refold_agent.rag.Chroma"), patch("refold_agent.rag.ChatOllama"), patch("refold_agent.rag.OllamaEmbeddings"), patch("refold_agent.rag.Ranker") as MockRanker:
        engine = RAGEngine(test_settings)
        engine.bm25_retriever = MagicMock()
        engine.bm25_retriever.invoke.return_value = [doc_cara]
        engine.vector_store.similarity_search_with_relevance_scores.return_value = [(doc_other, 0.6)]

        mock_ranker_instance = MockRanker.return_value
        mock_ranker_instance.rerank.return_value = [
            {"id": 0, "text": doc_cara.page_content, "meta": doc_cara.metadata, "score": 0.99},
            {"id": 1, "text": doc_other.page_content, "meta": doc_other.metadata, "score": 0.10},
        ]
        engine.ranker = mock_ranker_instance

        results = engine.hybrid_search_and_rerank("What is the CARA model?")

        assert len(results) == 1
        top_doc, score = results[0]
        assert top_doc.metadata["source"] == "phase4-speaking.md"
        assert score == 0.99
