"""FastAPI routes implementing OpenAI-compatible endpoints, /query, and /health."""

import json
import logging
import time
import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from refold_agent.config import Settings, get_settings
from refold_agent.db import AuditLogger
from refold_agent.rag import RAGEngine

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic Schemas
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="refold-agent")
    messages: list[ChatMessage]
    stream: bool = Field(default=False)
    temperature: float | None = None
    session_id: str | None = None


class QueryRequest(BaseModel):
    query: str
    chat_history: list[ChatMessage] | None = None
    session_id: str | None = None


class QueryResponse(BaseModel):
    query: str
    rephrased_query: str
    answer: str
    sources: list[dict[str, Any]]
    scores: list[float]
    has_context: bool


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    chroma_documents_count: int
    models: dict[str, str]


def get_rag_engine(request: Request) -> RAGEngine:
    """Retrieve RAGEngine from app state."""
    return request.app.state.rag_engine


def get_audit_logger(request: Request) -> AuditLogger:
    """Retrieve AuditLogger from app state."""
    return request.app.state.audit_logger


@router.get("/health", response_model=HealthResponse)
async def health_check(
    rag: RAGEngine = Depends(get_rag_engine),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Healthcheck endpoint for container orchestration and AWS ALBs."""
    doc_count = 0
    try:
        doc_count = rag.vector_store._collection.count()
    except Exception as e:
        logger.warning("Error fetching collection count during health check: %s", e)

    return HealthResponse(
        status="healthy",
        chroma_documents_count=doc_count,
        models={
            "chat": settings.ollama_chat_model,
            "embedding": settings.ollama_embedding_model,
        },
    )


@router.get("/v1/models")
async def list_models(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """OpenAI-compatible model list for OpenWebUI model discovery."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.openwebui_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "refold-agent",
            }
        ],
    }


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    rag: RAGEngine = Depends(get_rag_engine),
    audit: AuditLogger = Depends(get_audit_logger),
    settings: Settings = Depends(get_settings),
) -> QueryResponse:
    """Direct Q&A endpoint returning answer, citations, and scores."""
    if not body.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    history = [m.model_dump() for m in body.chat_history] if body.chat_history else []
    result = await rag.answer_query(body.query, chat_history=history)

    audit.log(
        endpoint="/query",
        session_id=body.session_id,
        prompt=body.query,
        rephrased_query=result.get("rephrased_query"),
        retrieved_sources=result.get("sources"),
        answer=result.get("answer", ""),
        model=settings.ollama_chat_model,
    )

    return QueryResponse(
        query=result["query"],
        rephrased_query=result["rephrased_query"],
        answer=result["answer"],
        sources=result["sources"],
        scores=result["scores"],
        has_context=result["has_context"],
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    rag: RAGEngine = Depends(get_rag_engine),
    audit: AuditLogger = Depends(get_audit_logger),
    settings: Settings = Depends(get_settings),
) -> Any:
    """OpenAI-compatible chat completion endpoint supporting streaming (SSE) and JSON."""
    if not body.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Messages list cannot be empty.",
        )

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_ts = int(time.time())
    messages_payload = [m.model_dump() for m in body.messages]
    latest_user_prompt = messages_payload[-1].get("content", "")

    if body.stream:
        async def event_generator():
            full_response_text = []
            final_metadata: dict[str, Any] = {}

            async for chunk in rag.stream_chat_completion(messages_payload):
                token = chunk.get("content", "")
                full_response_text.append(token)

                if "metadata" in chunk:
                    final_metadata = chunk["metadata"]

                sse_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": settings.openwebui_model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(sse_chunk)}\n\n"

            # Final stop chunk
            final_sse = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": settings.openwebui_model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_sse)}\n\n"
            yield "data: [DONE]\n\n"

            # Log completed stream interaction to audit database
            complete_text = "".join(full_response_text)
            audit.log(
                endpoint="/v1/chat/completions (stream)",
                session_id=body.session_id,
                prompt=latest_user_prompt,
                rephrased_query=final_metadata.get("rephrased_query"),
                retrieved_sources=final_metadata.get("sources"),
                answer=complete_text,
                model=settings.ollama_chat_model,
            )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming execution
    history = messages_payload[:-1]
    result = await rag.answer_query(latest_user_prompt, chat_history=history)

    audit.log(
        endpoint="/v1/chat/completions",
        session_id=body.session_id,
        prompt=latest_user_prompt,
        rephrased_query=result.get("rephrased_query"),
        retrieved_sources=result.get("sources"),
        answer=result.get("answer", ""),
        model=settings.ollama_chat_model,
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": settings.openwebui_model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("answer", ""),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@router.get("/v1/audit/logs")
async def get_audit_logs(
    limit: int = 50,
    audit: AuditLogger = Depends(get_audit_logger),
) -> list[dict[str, Any]]:
    """Retrieve recent queries and answers recorded in the audit database."""
    return audit.get_recent_logs(limit=limit)
