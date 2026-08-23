# SPEC.md

## §G — Goal
Build modular, cloud-ready local RAG API for ~45 Refold language-learning markdown documents in `resources/`. FastAPI exposes OpenAI-compatible SSE endpoints for OpenWebUI frontend, backed by ChromaDB + BM25 hybrid search, FlashRank cross-encoder reranking, intent routing, and Ollama. Embed Refold.la identity and methodology coaching persona in system prompts.

## §C — Constraints
- **Python**: ≥3.13, managed with `pyproject.toml`
- **Config**: 12-Factor App compliant using `pydantic-settings` (all paths, ports, model names overridable via env vars)
- **Ollama Models**:
  - Chat & Rephrasing: `gpt-oss`
  - Embedding: `qwen3-embedding`
  - Base URL: `http://localhost:11434` (`OLLAMA_BASE_URL`)
- **Storage Consolidation** (Docker-ready single volume mount):
  - ChromaDB: `./data/chroma/` (`CHROMA_DIR`)
  - SQLite Audit Log: `./data/chat.db` (`SQLITE_DB_PATH`)
- **Document Source**: `resources/*.md` (read-only, ~45 files)
- **Chunking Pipeline**:
  - `MarkdownHeaderTextSplitter` (`#`, `##`, `###`) to preserve topic hierarchy
  - `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- **Retrieval & Reranking**:
  - Fast Intent Router: Pydantic `Enum` (`QueryIntent`: `GREETING`, `REFOLD_QUERY`, `OUT_OF_SCOPE`) via `with_structured_output`
  - Multi-turn contextual query rephrasing for conversation continuity
  - Hybrid retrieval: BM25 (sparse keyword match for `CARA`, `0A`, `4X`, `CEFR`) + Chroma (dense vectors)
  - Reranker: `FlashRank` (`ms-marco-TinyBERT-L-2-v2`, local ONNX CPU) re-scores top-15 candidate pool down to top-6
  - Cosine/cross-encoder score threshold filter: score ≥ 0.5
  - Out-of-scope fallback: strict refusal without hallucination
  - Citations: Markdown `### Sources:` section appended at the end of RAG responses
- **Persona & Identity**:
  - System prompts explicitly establish Refold Assistant identity and `refold.la` as knowledge origin
- **Frontend / Cloud**:
  - OpenAI-compatible `/v1/` contract for OpenWebUI with SSE token streaming
  - `GET /health` endpoint for Docker and AWS ALB / ECS target group healthchecks

## §I — Interfaces
| id | surface | desc |
|---|---|---|
| I1 | `GET /health` | Healthcheck returning service & Ollama/Chroma connectivity status |
| I2 | `GET /v1/models` | OpenAI-compatible model discovery (`id: "refold-agent"`) |
| I3 | `POST /v1/chat/completions` | OpenAI chat completion endpoint (supports `stream: true` SSE & JSON) |
| I4 | `POST /query` | Direct Q&A endpoint: `{query}` → `{answer, sources, scores}` |
| I5 | ChromaDB `./data/chroma/` | Persistent vector store with chunk text and header metadata |
| I6 | SQLite `./data/chat.db` | Audit log table (`timestamp, session_id, prompt, retrieved_chunks, answer`) |
| I7 | CLI `--reindex` | Force rebuild vector store from `resources/` on startup |
| I8 | `docs/architecture.md` | Architecture rationale, chunking & RAG design decisions |
| I9 | `README.md` | Complete onboarding, API docs, and OpenWebUI setup guide |

## §V — Invariants
| id | invariant |
|---|---|
| V1 | Ingestion indexes all markdown files in `resources/` with header hierarchy metadata |
| V2 | `qwen3-embedding` used consistently for both indexing and query embedding |
| V3 | Chunks with similarity/rerank score < 0.5 excluded from prompt context |
| V4 | In multi-turn chat, follow-up queries rephrased before retrieval when history present |
| V5 | Queries with no matching chunks (score < 0.5) trigger strict out-of-scope refusal without hallucination |
| V6 | Generated RAG answers include markdown `### Sources:` section with filenames and headers |
| V7 | `/v1/chat/completions` handles both `stream: true` (SSE chunks) and `stream: false` (JSON) |
| V8 | All requests, retrieved contexts, and answers logged to SQLite `./data/chat.db` |
| V9 | Index loaded from disk on startup; only re-indexed if empty or `--reindex` passed |
| V10 | Service config strictly driven by `pydantic-settings` with zero hardcoded environment paths |
| V11 | `main.py` adds `src` to `sys.path` dynamically for standalone execution without editable install |
| V12 | Intent router outputs typed Pydantic `QueryIntent` enum before retrieval: `GREETING` → direct stream, `REFOLD_QUERY` → RAG, `OUT_OF_SCOPE` → refusal |
| V13 | `GREETING` queries bypass vector search and return warm Refold Assistant persona response with zero citations |
| V14 | `REFOLD_QUERY` uses Hybrid Search (BM25 + Chroma) ensuring exact acronyms (`CARA`, `0A`, `4X`, `CEFR`) are retrieved |
| V15 | FlashRank cross-encoder re-scores merged candidate pool, filtering chunks by score threshold |
| V16 | System prompts explicitly embed Refold.la provenance and methodology coaching persona |

## §T — Tasks
| id | status | desc | cites |
|---|---|---|---|
| T1 | x | Update `pyproject.toml` dependencies (`chromadb`, `langchain-chroma`, `langchain-community`, `pydantic-settings`, `sse-starlette`, `pytest`, `httpx`) | C.python |
| T2 | x | Implement `src/refold_agent/config.py` using `pydantic-settings` | V10,C.config |
| T3 | x | Implement `src/refold_agent/db.py` SQLite audit logger | I6,V8 |
| T4 | x | Implement `src/refold_agent/ingest.py` (Markdown header splitter + recursive chunker) | V1,I5 |
| T5 | x | Implement `src/refold_agent/rag.py` (vector store manager, query rephrase, similarity filter, prompt & citation generator) | V2,V3,V4,V5,V6 |
| T6 | x | Implement `src/refold_agent/api.py` (`/health`, `/v1/models`, `/v1/chat/completions` streaming/non-streaming, `/query`) | I1,I2,I3,I4,V7,V8 |
| T7 | x | Implement `main.py` entrypoint (lifespan startup/indexing, CLI `--reindex` arg parsing, uvicorn runner) | I7,V9 |
| T8 | x | Implement tests (`tests/test_rag.py`, `tests/test_api.py`) verifying V1-V10 | V1..V10 |
| T9 | x | Create `docs/architecture.md` (philosophy, chunking rationale, OpenWebUI architecture) | I8 |
| T10 | x | Populate `README.md` (overview, quickstart, OpenWebUI setup, Docker roadmap) | I9 |
| T11 | x | Update `config.py` with FlashRank model and hybrid pool hyperparameters | V10,C.rag |
| T12 | x | Update `ingest.py` to build `BM25Retriever` from loaded chunks | V1,V14 |
| T13 | x | Implement `classify_intent()` and Refold persona prompts in `rag.py` | V12,V13,V16 |
| T14 | x | Implement `hybrid_search_and_rerank()` with FlashRank in `rag.py` | V14,V15 |
| T15 | x | Add tests verifying intent routing, "CARA" keyword retrieval, and reranking | V12..V16 |
| T16 | x | Update `docs/architecture.md` and `README.md` with hybrid search & routing diagram | I8,I9 |

## §B — Bugs
| id | date | cause | fix |
|---|---|---|---|
| B1 | 2026-08-23 | ModuleNotFoundError when running root main.py without editable install | V11 |
