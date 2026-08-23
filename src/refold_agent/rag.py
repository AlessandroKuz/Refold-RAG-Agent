"""RAG core engine: embeddings, vector store manager, retrieval, query contextualization, and generation."""

import logging
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel, Field

from refold_agent.config import Settings
from refold_agent.ingest import ingest_resources

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    """Enumeration of possible user intents."""

    GREETING = "GREETING"
    REFOLD_QUERY = "REFOLD_QUERY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class IntentDecision(BaseModel):
    """Structured decision returned by the intent classifier."""

    intent: QueryIntent = Field(
        description="Classified intent: GREETING for casual pleasantries, REFOLD_QUERY for language acquisition questions, OUT_OF_SCOPE for unrelated topics.",
    )

PERSONA_SYSTEM_PROMPT = """You are the Refold Assistant, an AI coach for the Refold language learning methodology (published at https://refold.la).
Your purpose is to guide learners through the Refold roadmap: Phase 0 (Immersion), Phase 1 (Foundations), Phase 2 (Comprehension), Phase 3 (Listening), Phase 4 (Speaking), Phase 5 (Accuracy), Phase 6 (Fluency), and Phase 7 (Mastery).

When greeting users, introducing yourself, or responding to general pleasantries, introduce yourself warmly as the Refold Assistant, explain that all your knowledge is drawn directly from the official Refold documentation (refold.la), and invite them to ask about immersion habits, sentence mining, SRS flashcards, or specific phases."""

RAG_SYSTEM_PROMPT = """You are the Refold Assistant, an expert AI coach specializing in the Refold language learning methodology (from https://refold.la).
Your knowledge base is drawn directly from the official Refold documentation.

Instructions:
1. Answer the user's question accurately and concisely, strictly grounded in the Refold documentation context below.
2. Use an encouraging, practical tone reflecting Refold philosophy (comprehensible input, low-stress immersion, consistency over intensity).
3. If the context does not contain sufficient details to answer, state clearly that the Refold documentation does not contain information to answer this question.
4. Do not hallucinate or invent rules/methods outside the provided documentation.

Refold Documentation Context:
{context}"""

NO_CONTEXT_FALLBACK = (
    "I'm sorry, but the Refold documentation does not contain information to answer this question. "
    "Please ask a question related to the Refold language learning methodology, phases, immersion techniques, or study habits."
)

REPHRASE_SYSTEM_PROMPT = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

INTENT_ROUTER_PROMPT = """You are an intent classifier for a Refold language learning assistant. Classify the user input into exactly one label:
- GREETING: Greetings, polite pleasantries, introductions, 'who are you', 'what can you do', thanks, goodbyes.
- REFOLD_QUERY: Questions about learning languages, Refold methodology, phases 0-7, immersion, comprehension, listening, speaking, grammar, vocabulary, SRS flashcards, CARA, sentence mining.
- OUT_OF_SCOPE: Completely unrelated topics (cooking, coding, sports, weather, math, politics, general trivia).

Output ONLY the label name: GREETING, REFOLD_QUERY, or OUT_OF_SCOPE."""


class RAGEngine:
    """Core RAG pipeline coordinating ingestion, retrieval, and LLM generation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.llm = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
        self.router_llm = self.llm.with_structured_output(IntentDecision)
        self.vector_store = Chroma(
            collection_name="refold_docs",
            embedding_function=self.embeddings,
            persist_directory=str(settings.chroma_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def initialize_index(self, force_reindex: bool = False) -> int:
        """Initialize or rebuild the Chroma vector index from markdown resources."""
        current_count = 0
        try:
            current_count = self.vector_store._collection.count()
        except Exception as e:
            logger.debug("Error checking collection count: %s", e)

        if current_count > 0 and not force_reindex:
            logger.info(
                "Chroma vector store already contains %d documents. Skipping indexing.",
                current_count,
            )
            return current_count

        logger.info(
            "Indexing documents from %s (force_reindex=%s)...",
            self.settings.resources_dir,
            force_reindex,
        )

        if force_reindex and current_count > 0:
            try:
                self.vector_store.delete_collection()
                # Re-initialize collection reference
                self.vector_store = Chroma(
                    collection_name="refold_docs",
                    embedding_function=self.embeddings,
                    persist_directory=str(self.settings.chroma_dir),
                    collection_metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                logger.warning("Error resetting collection: %s", e)

        docs = ingest_resources(
            self.settings.resources_dir,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        if not docs:
            logger.warning("No documents found to index.")
            return 0

        self.vector_store.add_documents(docs)
        total = self.vector_store._collection.count()
        logger.info("Successfully indexed %d chunks into ChromaDB.", total)
        return total

    async def contextualize_query(
        self,
        query: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        """Rephrase follow-up query using conversation history for standalone retrieval."""
        if not chat_history:
            return query

        history_messages: list[BaseMessage] = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                history_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                history_messages.append(AIMessage(content=content))

        if not history_messages:
            return query

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", REPHRASE_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        chain = prompt_template | self.llm
        try:
            response = await chain.ainvoke(
                {
                    "chat_history": history_messages,
                    "input": query,
                }
            )
            rephrased = (
                response.content.strip() if isinstance(response.content, str) else query
            )
            logger.debug("Rephrased query '%s' -> '%s'", query, rephrased)
            return rephrased
        except Exception as e:
            logger.warning(
                "Failed to contextualize query: %s. Using original query.", e
            )
            return query

    async def classify_intent(self, query: str) -> QueryIntent:
        """Classify user input into GREETING, REFOLD_QUERY, or OUT_OF_SCOPE using structured schema."""
        try:
            messages = [
                SystemMessage(content=INTENT_ROUTER_PROMPT),
                HumanMessage(content=query),
            ]
            decision = await self.router_llm.ainvoke(messages)
            if isinstance(decision, IntentDecision):
                return decision.intent
            if isinstance(decision, dict) and "intent" in decision:
                return QueryIntent(decision["intent"])
            return QueryIntent.REFOLD_QUERY
        except Exception as e:
            logger.warning("Structured intent classification failed: %s. Defaulting to REFOLD_QUERY.", e)
            return QueryIntent.REFOLD_QUERY

    def retrieve_relevant_chunks(
        self,
        query: str,
    ) -> list[tuple[Document, float]]:
        """Retrieve top_k chunks and filter by cosine similarity threshold."""
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query,
            k=self.settings.top_k,
        )

        filtered: list[tuple[Document, float]] = []
        for doc, score in results_with_scores:
            if score >= self.settings.similarity_threshold:
                filtered.append((doc, float(score)))

        logger.debug(
            "Retrieved %d chunks (threshold >= %s, raw results: %d)",
            len(filtered),
            self.settings.similarity_threshold,
            len(results_with_scores),
        )
        return filtered

    @staticmethod
    def format_sources_section(
        scored_docs: list[tuple[Document, float]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Build markdown Sources section and structured metadata list."""
        if not scored_docs:
            return "", []

        sources_metadata: list[dict[str, Any]] = []
        source_labels: list[str] = []
        seen: set[str] = set()

        for doc, score in scored_docs:
            source = doc.metadata.get("source", "unknown")
            h1 = doc.metadata.get("Header 1")
            h2 = doc.metadata.get("Header 2")
            h3 = doc.metadata.get("Header 3")

            headers = [h for h in [h1, h2, h3] if h]
            header_path = " > ".join(headers) if headers else ""
            label_key = f"{source}:{header_path}"

            sources_metadata.append(
                {
                    "source": source,
                    "header_path": header_path,
                    "score": round(score, 4),
                    "content_preview": doc.page_content[:150]
                    + ("..." if len(doc.page_content) > 150 else ""),
                }
            )

            if label_key not in seen:
                seen.add(label_key)
                if header_path:
                    source_labels.append(f"- `{source}` ({header_path})")
                else:
                    source_labels.append(f"- `{source}`")

        section_md = "\n\n### Sources:\n" + "\n".join(source_labels)
        return section_md, sources_metadata

    def build_context_string(self, scored_docs: list[tuple[Document, float]]) -> str:
        """Format retrieved documents into a context block for the prompt."""
        chunks: list[str] = []
        for i, (doc, _) in enumerate(scored_docs, start=1):
            source = doc.metadata.get("source", "unknown")
            h1 = doc.metadata.get("Header 1")
            h2 = doc.metadata.get("Header 2")
            headers = [h for h in [h1, h2] if h]
            header_info = f" ({' > '.join(headers)})" if headers else ""
            chunks.append(
                f"--- Document {i}: {source}{header_info} ---\n{doc.page_content}"
            )
        return "\n\n".join(chunks)

    async def answer_query(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Execute non-streaming query with intent routing and strict fallback."""
        chat_history = chat_history or []
        intent = await self.classify_intent(query)

        # Route 1: Casual Greetings & Persona Chat
        if intent == QueryIntent.GREETING:
            system_message = SystemMessage(content=PERSONA_SYSTEM_PROMPT)
            messages: list[BaseMessage] = [system_message]
            for msg in chat_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            messages.append(HumanMessage(content=query))

            response = await self.llm.ainvoke(messages)
            answer_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return {
                "query": query,
                "rephrased_query": query,
                "answer": answer_text,
                "sources": [],
                "scores": [],
                "has_context": True,
            }

        # Route 2: Explicitly Out of Scope
        if intent == QueryIntent.OUT_OF_SCOPE:
            return {
                "query": query,
                "rephrased_query": query,
                "answer": NO_CONTEXT_FALLBACK,
                "sources": [],
                "scores": [],
                "has_context": False,
            }

        # Route 3: Refold Methodology Query (RAG Pipeline)
        standalone_query = await self.contextualize_query(query, chat_history)
        scored_docs = self.retrieve_relevant_chunks(standalone_query)

        if not scored_docs:
            return {
                "query": query,
                "rephrased_query": standalone_query,
                "answer": NO_CONTEXT_FALLBACK,
                "sources": [],
                "scores": [],
                "has_context": False,
            }

        context = self.build_context_string(scored_docs)
        sources_md, sources_meta = self.format_sources_section(scored_docs)

        system_message = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
        messages: list[BaseMessage] = [system_message]

        for msg in chat_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        messages.append(HumanMessage(content=query))

        response = await self.llm.ainvoke(messages)
        answer_text = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        full_answer = f"{answer_text}{sources_md}"

        return {
            "query": query,
            "rephrased_query": standalone_query,
            "answer": full_answer,
            "sources": sources_meta,
            "scores": [s["score"] for s in sources_meta],
            "has_context": True,
        }

    async def stream_chat_completion(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream response tokens with intent routing and optional source citations."""
        if not messages:
            yield {"type": "token", "content": "No prompt provided."}
            return

        latest_user_message = ""
        chat_history: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role")
            if role == "user":
                latest_user_message = msg.get("content", "")
            if msg != messages[-1]:
                chat_history.append(msg)

        intent = await self.classify_intent(latest_user_message)

        # Route 1: Casual Greetings & Persona Chat
        if intent == QueryIntent.GREETING:
            system_message = SystemMessage(content=PERSONA_SYSTEM_PROMPT)
            llm_messages: list[BaseMessage] = [system_message]
            for msg in chat_history:
                if msg.get("role") == "user":
                    llm_messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    llm_messages.append(AIMessage(content=msg.get("content", "")))
            llm_messages.append(HumanMessage(content=latest_user_message))

            async for chunk in self.llm.astream(llm_messages):
                token_text = (
                    chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                )
                yield {"type": "token", "content": token_text}

            yield {
                "type": "token",
                "content": "",
                "metadata": {
                    "rephrased_query": latest_user_message,
                    "sources": [],
                    "has_context": True,
                },
            }
            return

        # Route 2: Explicitly Out of Scope
        if intent == QueryIntent.OUT_OF_SCOPE:
            yield {
                "type": "token",
                "content": NO_CONTEXT_FALLBACK,
                "metadata": {
                    "rephrased_query": latest_user_message,
                    "sources": [],
                    "has_context": False,
                },
            }
            return

        # Route 3: Refold Methodology Query (RAG Pipeline)
        standalone_query = await self.contextualize_query(
            latest_user_message, chat_history
        )
        scored_docs = self.retrieve_relevant_chunks(standalone_query)

        if not scored_docs:
            yield {
                "type": "token",
                "content": NO_CONTEXT_FALLBACK,
                "metadata": {
                    "rephrased_query": standalone_query,
                    "sources": [],
                    "has_context": False,
                },
            }
            return

        context = self.build_context_string(scored_docs)
        sources_md, sources_meta = self.format_sources_section(scored_docs)

        system_message = SystemMessage(content=RAG_SYSTEM_PROMPT.format(context=context))
        llm_messages: list[BaseMessage] = [system_message]

        for msg in chat_history:
            if msg.get("role") == "user":
                llm_messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                llm_messages.append(AIMessage(content=msg.get("content", "")))

        llm_messages.append(HumanMessage(content=latest_user_message))

        async for chunk in self.llm.astream(llm_messages):
            token_text = (
                chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            )
            yield {"type": "token", "content": token_text}

        # Append source citations at the end of the stream
        yield {
            "type": "token",
            "content": sources_md,
            "metadata": {
                "rephrased_query": standalone_query,
                "sources": sources_meta,
                "has_context": True,
            },
        }
