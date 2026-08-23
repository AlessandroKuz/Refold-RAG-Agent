# Quickstart & Learning Guide

This guide walks you through running the Refold RAG agent, connecting it to the OpenWebUI frontend, and exploring the codebase step by step.

---

## 1. Quickstart Commands

A `justfile` is provided with preconfigured recipes. You can use `just <recipe>` or the direct CLI commands.

### All-In-One One-Liner

If you have `just`, `docker`, and `ollama` installed, run:

```bash
just run-everything
```

*(Creates `.env` → pulls models → runs tests → launches OpenWebUI in Docker → indexes documents → starts server at port 8000)*

---

### Step-by-Step Recipes

| Task | `just` Command | Direct CLI Command |
|---|---|---|
| **1. Create `.env`** | `just env` | `cp .env.example .env` |
| **2. Pull Models** | `just models` | `ollama pull gpt-oss && ollama pull qwen3-embedding` |
| **3. Run Tests** | `just test` | `pytest` |
| **4. Ingest & Index** | `just index` | `python main.py --reindex` |
| **5. Start Backend** | `just run` | `python main.py` |
| **6. Start OpenWebUI** | `just openwebui` | *(Docker command below)* |
| **7. Check Health** | `just health` | `curl -s http://localhost:8000/health` |
| **8. View Audit Logs** | `just logs 10` | `curl -s http://localhost:8000/v1/audit/logs?limit=10` |
| **9. Stop OpenWebUI** | `just openwebui-stop` | `docker stop open-webui && docker rm open-webui` |
| **10. Reset OpenWebUI Account** | `just openwebui-reset` | `docker stop open-webui && docker rm open-webui && docker volume rm open-webui && just openwebui` |

#### Launch OpenWebUI via Docker (Direct command)

```bash
docker run -d \
  --network=host \
  -e PORT=3000 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

---

## 2. Connecting OpenWebUI to the RAG Agent

1. Open your browser and navigate to **`http://localhost:3000`**.
2. Complete the quick initial admin account setup.
3. Click the **User Icon** (bottom-left) → **Admin Panel** → **Settings** → **Connections**.
4. In the **OpenAI API** section:
   - Click the **+** button or edit the existing connection.
   - Set **API Base URL**: `http://localhost:8000/v1` (or `http://127.0.0.1:8000/v1`)
   - Set **API Key**: `local` (or any dummy token)
5. Click **Verify Connection** (refresh icon 🔄) and then **Save**.
6. Return to the main chat interface, select **`refold-agent`** from the model dropdown.
7. Start asking questions about the Refold method (e.g. *"What is the goal of Phase 1 Foundations?"* or *"How do I start sentence mining in Phase 2?"*).
8. Notice real-time token streaming and the markdown `### Sources:` block appended to the response.

---

## 3. Direct API & Swagger Interface

If you want to interact directly with the backend without OpenWebUI:

- **Interactive Swagger Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **Direct Q&A Endpoint (`POST /query`)**:

  ```bash
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the CARA model in Phase 5?"}'
  ```

---

## 4. Configuration & Environment Variables

All settings are driven by `pydantic-settings` and can be customized in `.env` (copied from `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Server host binding |
| `PORT` | `8000` | Server port binding |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama connection endpoint |
| `OLLAMA_CHAT_MODEL` | `gpt-oss` | Chat model for generation and rephrasing |
| `OLLAMA_EMBEDDING_MODEL` | `qwen3-embedding` | Embedding model for ChromaDB vectors |
| `OPENWEBUI_MODEL_NAME` | `refold-agent` | Model ID exposed in `/v1/models` |
| `DATA_DIR` | `data` | Root storage directory (Docker mount target) |
| `CHROMA_DIR` | `data/chroma` | Persistent ChromaDB vector index path |
| `SQLITE_DB_PATH` | `data/chat.db` | SQLite audit database path |
| `TOP_K` | `8` | Number of chunks retrieved before filtering |
| `SIMILARITY_THRESHOLD` | `0.5` | Minimum cosine similarity score required |

---

## 5. Codebase Learning & Exploration Guide

To understand how this RAG application works end-to-end so you can build one yourself, read the files in this logical order:

```
Step 1: System Spec & Dependencies
  ├── SPEC.md                      <- Contract: invariants, interfaces, and architecture rules
  └── pyproject.toml               <- Dependencies: langchain-chroma, langchain-ollama, fastapi

Step 2: 12-Factor Configuration
  └── src/refold_agent/config.py   <- Type-safe settings, environment overrides, hyperparameters

Step 3: Data Ingestion & Semantic Chunking
  └── src/refold_agent/ingest.py   <- MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter

Step 4: Persistence & Audit Logging
  └── src/refold_agent/db.py       <- SQLite database logging queries, sources, and completions

Step 5: RAG Core Engine (The Heart)
  └── src/refold_agent/rag.py      <- Embeddings, Chroma vector store, contextual query rephrasing,
                                      cosine similarity filter (>0.5), strict refusal, and SSE streaming

Step 6: OpenAI-Compatible Web API
  └── src/refold_agent/api.py      <- FastAPI routes: /health, /v1/models, /v1/chat/completions, /query

Step 7: Lifecycle & Server Entrypoint
  └── main.py                      <- FastAPI lifespan (startup index check, CLI --reindex flag)

Step 8: Testing & Mocking
  ├── tests/conftest.py            <- Mock fixtures, isolated temporary databases
  ├── tests/test_ingest.py         <- Validates header metadata preservation
  ├── tests/test_rag.py            <- Validates threshold filtering and citations
  └── tests/test_api.py            <- Validates OpenAI SSE streaming and non-streaming responses
```

### Why this design works

1. **Decoupled layers**: Ingestion (`ingest.py`), RAG logic (`rag.py`), and Web routes (`api.py`) are independent and easily testable without a live database or LLM.
2. **OpenAI API Contract**: By implementing `/v1/models` and `/v1/chat/completions` with SSE streaming, any UI that supports OpenAI (OpenWebUI, LibreChat, Continue.dev, Cursor) works out of the box.
3. **Auditability**: Every single request, the standalone rephrased query, the retrieved chunks, and the generated response are preserved in `./data/chat.db` for review and evaluation.
4. **Cloud-Readiness**: Persistent state is consolidated in `./data/` for a single volume mount in Docker and AWS ECS deployments.
