# Architecture & Design Philosophy

## 1. Overview & System Purpose

`refold-agent` is a production-oriented Retrieval-Augmented Generation (RAG) backend designed to index and query the complete Refold language learning methodology (~45 markdown documents).

The application serves as an OpenAI-compatible backend for chat frontends such as **OpenWebUI**, while also offering direct structured API endpoints for programmatic integrations.

```
+-------------------------------------------------------------+
|                      OpenWebUI Frontend                     |
+-------------------------------------------------------------+
                              |
                     HTTP / SSE Streaming
                              v
+-------------------------------------------------------------+
|                   FastAPI Application Engine                |
|  - GET /health                                              |
|  - GET /v1/models                                           |
|  - POST /v1/chat/completions (OpenAI SSE Stream / JSON)     |
|  - POST /query (Direct Structured Q&A)                      |
+-------------------------------------------------------------+
                              |
                              v
                   [Fast Intent Classifier]
                   (Pydantic QueryIntent Enum)
                   /          |             \
       GREETING   /           | REFOLD_QUERY \  OUT_OF_SCOPE
                 v            v               v
    +-----------------+  +-----------------+  +------------------+
    | Direct Persona  |  | [Contextualizer]|  |  Strict Refusal  |
    | Reply (Refold.la|  +-----------------+  |  Fallback (0 RAG)|
    | AI Coach, 0 RAG)|           |           +------------------+
    +-----------------+           v
                         [Hybrid Retrieval]
                         ├── BM25 (Sparse Top-10)
                         └── Chroma (Dense Top-10)
                                  |
                                  v
                         [Deduplicate Pool] (15 chunks)
                                  |
                                  v
                         [FlashRank Cross-Encoder]
                         (ms-marco-TinyBERT, score >= 0.5)
                                  |
                                  v
                         [LLM Generation + Citations]
                                  |
                                  v
                         [SQLite Audit Log] (data/chat.db)
```

---

## 2. Core Architectural Decisions

### A. Typed Intent Routing & Persona Anchoring
- **Zero-RAG Casual Greetings**: Using a typed Pydantic enum (`QueryIntent.GREETING`), greetings and introductions are routed directly to the LLM with `PERSONA_SYSTEM_PROMPT`. This delivers instant sub-second responses without unnecessary vector search or false refusals.
- **Explicit Provenance**: Prompts explicitly anchor the assistant's persona to `refold.la`, guiding users on phases, immersion, and sentence mining.

### B. Two-Stage Document Chunking
- **Stage 1: Markdown Header Splitting**: Refold documents are organized hierarchically by phases and sub-phases (e.g., `# 0A — Method Overview`, `## Phase 1: Foundations`). `MarkdownHeaderTextSplitter` preserves this hierarchy as metadata on each chunk.
- **Stage 2: Recursive Character Splitting**: Long sections are split using `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200) ensuring semantic coherence across boundary cuts.

### C. Hybrid Search (BM25 + Chroma) & Cross-Encoder Reranking
- **BM25 Sparse Retrieval**: Dense embeddings often blur short exact acronyms (`CARA`, `0A`, `4X`, `CEFR`). BM25 indexes exact tokens to guarantee acronym discovery.
- **Chroma Dense Vectors**: Captures broad semantic intent and conceptual questions.
- **FlashRank Reranking**: Merges top-10 BM25 and top-10 Chroma candidates (up to 15 unique chunks) and re-ranks them using `ms-marco-TinyBERT-L-2-v2` (3.2MB local ONNX model). Chunks scoring `< 0.5` are discarded.

### D. Multi-Turn Query Contextualization
- In multi-turn chat sessions, follow-up questions often contain implicit references (e.g., "What about phase 2?").
- The chat history and latest question are passed through a contextualizer LLM prompt to generate an independent search query before hybrid retrieval.

### E. 12-Factor & Cloud Deployment Readiness
- **Zero Hardcoding**: All configuration is managed via `pydantic-settings` (`Settings` class) driven by environment variables.
- **Volume Consolidation**: All persistent mutable state is contained under `./data/` (`./data/chroma/` and `./data/chat.db`). In Docker / AWS ECS deployments, mounting a single volume (`-v /host/path:/app/data`) guarantees zero data loss across container lifecycle events.
- **Health Checks**: `GET /health` provides instant status on vector index load and model configuration for AWS ALB target group healthchecks.
