# Refold RAG Agent

A production-ready, local Retrieval-Augmented Generation (RAG) backend indexing the complete [Refold](https://refold.la) language acquisition methodology (~45 markdown documents).

Exposes an **OpenAI-compatible API** with SSE streaming for **OpenWebUI**, along with direct structured query endpoints, SQLite audit logging, and persistent ChromaDB vector storage.

---

## Key Features

- **Typed Intent Router**: Fast Pydantic enum routing (`GREETING` → instant persona reply, `REFOLD_QUERY` → RAG, `OUT_OF_SCOPE` → refusal).
- **Hybrid Search (BM25 + Chroma)**: Captures exact keyword acronyms (`CARA`, `0A`, `4X`, `CEFR`) alongside dense semantic concepts.
- **FlashRank Cross-Encoder Reranking**: Re-scores merged candidate pool locally via `ms-marco-TinyBERT-L-2-v2`.
- **Hierarchical Markdown Chunking**: Preserves phase and section header breadcrumbs (`#`, `##`, `###`).
- **OpenAI-Compatible `/v1/` Endpoints**: Drop-in backend for OpenWebUI with real-time SSE token streaming.
- **Strict Anti-Hallucination & Score Filtering**: Rerank score threshold (`>= 0.5`) prevents hallucinated advice on out-of-scope queries.
- **Contextual Follow-up Rephrasing**: Automatically reformulates conversational follow-ups into standalone vector queries.
- **Source Citations**: Every RAG response includes source filenames and section headers.
- **SQLite Audit Logging**: Persists all prompts, rephrased queries, retrieved chunk metadata, and completions in `./data/chat.db`.
- **12-Factor Cloud & Docker Ready**: Single `./data/` volume mount for AWS ECS / ALB deployment in Phase 2.

---

## Prerequisites

1. **Python 3.13+** (or `uv` package manager)
2. **Ollama** running locally with required models:
   ```bash
   ollama pull gpt-oss
   ollama pull qwen3-embedding
   ```

---

## Quickstart

### 1. Install Dependencies

Using `uv`:
```bash
uv sync
```

Or using `pip`:
```bash
pip install -e .
```

### 2. Start the Server

```bash
python main.py
```

To force a re-indexing of all markdown documents from `resources/`:
```bash
python main.py --reindex
```

The API will start at `http://localhost:8000`.

---

## Connecting OpenWebUI

1. Open **OpenWebUI** (`http://localhost:3000` or your deployment).
2. Navigate to **Admin Panel** > **Settings** > **Connections** > **OpenAI API**.
3. Add a new connection:
   - **API Base URL**: `http://localhost:8000/v1` (or `http://host.docker.internal:8000/v1` if OpenWebUI is in Docker)
   - **API Key**: Any string (e.g., `local`)
4. Save and select the **`refold-agent`** model in the chat interface.

---

## API Endpoints

### 1. Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "chroma_documents_count": 320,
  "models": {
    "chat": "gpt-oss",
    "embedding": "qwen3-embedding"
  }
}
```

### 2. Direct Q&A Query
```http
POST /query
Content-Type: application/json

{
  "query": "What is the primary goal of Phase 1 Foundations?"
}
```

### 3. OpenAI Chat Completions (Streaming & Non-Streaming)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "refold-agent",
  "messages": [
    {"role": "user", "content": "How do I start sentence mining in Phase 2?"}
  ],
  "stream": true
}
```

### 4. Audit Logs
```http
GET /v1/audit/logs?limit=20
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `8000` | Server port binding |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama service endpoint |
| `OLLAMA_CHAT_MODEL` | `gpt-oss` | Chat & rephrasing model |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding` | Embedding model |
| `RESOURCES_DIR` | `resources` | Path to markdown source documents |
| `DATA_DIR` | `data` | Root directory for state persistence |
| `CHROMA_DIR` | `data/chroma` | ChromaDB vector store directory |
| `SQLITE_DB_PATH` | `data/chat.db` | SQLite audit database path |
| `TOP_K` | `8` | Maximum chunks retrieved |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity threshold |

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for detailed technical decisions, chunking pipeline rationale, and AWS/Docker deployment design.
