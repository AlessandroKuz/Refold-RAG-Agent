# Architecture & Design Philosophy

## 1. Overview & System Purpose

`refold-agent` is a production-oriented Retrieval-Augmented Generation (RAG) backend designed to index and query the complete Refold language learning methodology (~45 markdown documents).

The application is structured to serve as an OpenAI-compatible backend for chat frontends such as **OpenWebUI**, while also offering direct structured API endpoints for programmatic integrations.

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
       |                           |                     |
       v                           v                     v
+---------------+         +------------------+   +-------------------+
| Multi-turn    |         | ChromaDB Index   |   | SQLite Audit DB   |
| Query         |         | (Cosine Sim >0.5)|   | (data/chat.db)    |
| Rephraser     |         | (data/chroma/)   |   +-------------------+
+---------------+         +------------------+
       |                           |
       +-------------+-------------+
                     |
                     v
+-------------------------------------------------------------+
|                       Ollama Service                        |
|  - Chat: gpt-oss                                            |
|  - Embeddings: qwen3-embedding                              |
+-------------------------------------------------------------+
```

---

## 2. Core Architectural Decisions

### A. Two-Stage Document Chunking

- **Stage 1: Markdown Header Splitting**: Refold documents are organized hierarchically by phases and sub-phases (e.g., `# 0A — Method Overview`, `## Phase 1: Foundations`). `MarkdownHeaderTextSplitter` preserves this hierarchy as metadata on each chunk.
- **Stage 2: Recursive Character Splitting**: Long sections are split using `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200) ensuring semantic coherence across boundary cuts.

### B. Similarity Threshold & Hallucination Defense

- Queries undergo cosine similarity scoring in ChromaDB (`hnsw:space: "cosine"`).
- Only chunks scoring `> 0.5` are admitted to the prompt context.
- **Strict Refold Refusal**: If no chunks satisfy the threshold, the system rejects out-of-scope questions immediately, preventing hallucination of non-existent methodology guidelines.

### C. Conversational Query Contextualization

- In multi-turn chat sessions, follow-up questions often contain implicit references (e.g., "What should I do in phase 2?").
- Prior to vector search, the chat history and latest question are passed through a contextualizer LLM prompt to generate an independent search query.

### D. 12-Factor & Cloud Deployment Readiness

- **Zero Hardcoding**: All configuration is managed via `pydantic-settings` (`Settings` class) driven by environment variables.
- **Volume Consolidation**: All persistent mutable state is contained under `./data/` (`./data/chroma/` and `./data/chat.db`). In Docker / AWS ECS deployments, mounting a single volume (`-v /host/path:/app/data`) guarantees zero data loss across container lifecycle events.
- **Health Checks**: `GET /health` provides instant status on vector index load and model configuration for AWS ALB target group healthchecks.
