# SPEC.md

## §G — Goal
Build modular, cloud-ready local RAG API for ~45 Refold language-learning markdown documents in `resources/`. FastAPI exposes OpenAI-compatible SSE endpoints for OpenWebUI frontend, backed by ChromaDB and Ollama. Document architecture and design decisions in `docs/` and `README.md`.

## §C — Constraints
- **Python**: ≥3.13, managed with `pyproject.toml`
- **Config**: 12-Factor App compliant using `pydantic-settings` (all paths, ports, model names overridable via env vars)
- **Ollama Models**:
  - Chat: `gpt-oss`
  - Embedding: `qwen3-embedding`
  - Base URL: `http://localhost:11434` (`OLLAMA_BASE_URL`)
- **Storage Consolidation** (Docker-ready single volume mount):
  - ChromaDB: `./data/chroma/` (`CHROMA_DIR`)
  - SQLite Audit Log: `./data/chat.db` (`SQLITE_DB_PATH`)
- **Document Source**: `resources/*.md` (read-only, ~45 files)
- **Chunking Pipeline**:
  - `MarkdownHeaderTextSplitter` (`#`, `##`, `###`) to preserve topic hierarchy
  - `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)
- **Retrieval & Generation**:
  - Multi-turn contextual query rephrasing for conversation continuity
  - Retrieval `top_k = 8`
  - Cosine similarity threshold: strict score > 0.5 filter
  - Out-of-scope fallback: strict refusal without hallucination
  - Citations: Markdown `### Sources:` section appended at the end
- **Frontend / Cloud**:
  - OpenAI-compatible `/v1/` contract for OpenWebUI with SSE token streaming
  - `GET /health` endpoint for Docker and AWS ALB / ECS target group healthchecks
- **Documentation**:
  - `docs/architecture.md` explaining RAG design decisions & 12-factor cloud principles
  - `README.md` with full setup, OpenWebUI connection guide, and API reference

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
| V3 | Chunks with cosine similarity ≤ 0.5 excluded from prompt context |
| V4 | In multi-turn chat, follow-up queries rephrased before retrieval when history present |
| V5 | Queries with no matching chunks (>0.5) trigger strict out-of-scope refusal without hallucination |
| V6 | Generated answers include markdown `### Sources:` section with filenames and headers |
| V7 | `/v1/chat/completions` handles both `stream: true` (SSE chunks) and `stream: false` (JSON) |
| V8 | All requests, retrieved contexts, and answers logged to SQLite `./data/chat.db` |
| V9 | Index loaded from disk on startup; only re-indexed if empty or `--reindex` passed |
| V10 | Service config strictly driven by `pydantic-settings` with zero hardcoded environment paths |
| V11 | `main.py` adds `src` to `sys.path` dynamically for standalone execution without editable install |

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

## §B — Bugs
| id | date | cause | fix |
|---|---|---|---|
| B1 | 2026-08-23 | ModuleNotFoundError when running root main.py without editable install | V11 |
