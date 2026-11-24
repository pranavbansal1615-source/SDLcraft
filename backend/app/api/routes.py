"""
FastAPI route definitions for the SDLcraft backend API.

Endpoints
---------
POST /api/ingest/text        — Ingest free-text project description
POST /api/ingest/pdf         — Ingest a PDF document (multipart upload)
POST /api/requirements/query — Query requirements from indexed document
POST /api/sdlc/recommend     — SDLC recommendation from requirement strings
POST /api/pipeline/text      — Full pipeline: text → requirements → SDLC
POST /api/pipeline/pdf       — Full pipeline: PDF → requirements → SDLC
GET  /api/sdlc/catalogue     — List all supported SDLC models
GET  /api/health             — Health check
"""
from __future__ import annotations

import uuid
import re
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.agents.sdlc_agent import SDLCAgent
from app.core.config import OUTPUT_DIR, UPLOAD_DIR
from app.rag.rag import (
    answer_query,
    answer_query_pdf_rag,
    answer_query_from_text,
    generate_pdf_emb,
    generate_text_emb,
    generate_requirements,
    generate_requirements_direct,
)
from app.schemas.models import (
    FullPipelineResponse,
    FullPipelineTextRequest,
    IngestionResponse,
    RequirementsQueryRequest,
    RequirementsResponse,
    SDLCRecommendRequest,
    SDLCResponse,
    TextIngestionRequest,
)

router = APIRouter()
_agent = SDLCAgent()

TEXT_REQUIREMENTS_INSTRUCTION = (
    "Generate the complete list of functional, security, compliance, performance, "
    "and usability requirements for the project described below. "
    "Only include security and compliance requirements that are clearly relevant to this specific project — "
    "do not inject generic finance boilerplate for projects that are simple internal tools, dashboards, or prototypes. "
    "Mark any requirement sourced from general domain knowledge (not explicitly in the description) as ASSUMPTION."
)

PDF_REQUIREMENTS_INSTRUCTION = (
    "Using the uploaded PDF as the primary evidence, generate specific, testable, "
    "IEEE-830 compliant finance software requirements. "
    "Only extract requirements that are supported by or directly implied by the document content. "
    "Mark anything added from general finance knowledge (not in the PDF) as ASSUMPTION, not INFERRED. "
    "Do not inject regulatory or security boilerplate unless the document explicitly addresses those domains."
)


def _safe_output_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._") or "finance_project"


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "SDLcraft API"}


# ── Ingestion ────────────────────────────────────────────────────────────────

@router.post("/ingest/text", response_model=IngestionResponse, tags=["Ingestion"])
def ingest_text(body: TextIngestionRequest):
    """Index a free-text project description into the vector store."""
    try:
        doc_id = body.document_id or f"text_{uuid.uuid4().hex[:8]}"
        msg = generate_text_emb(body.text, body.user_id, doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {exc}")
    return IngestionResponse(status="success", document_id=doc_id, message=msg)


@router.post("/ingest/pdf", response_model=IngestionResponse, tags=["Ingestion"])
async def ingest_pdf(
    file: UploadFile = File(...),
    user_id: str = Form("default_user"),
    document_id: str = Form(None),
):
    """Upload and index a PDF requirements document."""
    try:
        filename = Path(file.filename or "upload.pdf").name
        doc_id = document_id or Path(filename).stem
        save_path = UPLOAD_DIR / f"{doc_id}_{filename}"
        save_path.write_bytes(await file.read())
        msg = generate_pdf_emb(str(save_path), user_id, doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {exc}")

    return IngestionResponse(status="success", document_id=doc_id, message=msg)


# ── Requirements Query ───────────────────────────────────────────────────────

@router.post("/requirements/query", response_model=RequirementsResponse, tags=["Requirements"])
def query_requirements(body: RequirementsQueryRequest):
    """Query requirements from an already-indexed document."""
    try:
        result = answer_query(body.query, body.document_id, body.user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Requirements query failed: {exc}")

    return RequirementsResponse(
        document_id=body.document_id,
        query=body.query,
        requirements_markdown=result,
    )


# ── SDLC Recommendation ──────────────────────────────────────────────────────

@router.post("/sdlc/recommend", response_model=SDLCResponse, tags=["SDLC"])
def recommend_sdlc(body: SDLCRecommendRequest):
    """Run SDLC recommendation from a list of requirement strings."""
    try:
        report = _agent.recommend_from_list(body.requirements, project_name=body.project_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SDLC recommendation failed: {exc}")
    top3 = [
        {"rank": m.rank, "model": m.model_name, "score": m.score, "suitability": m.suitability}
        for m in report.ranked_models[:3]
    ]
    return SDLCResponse(
        project_name=body.project_name,
        recommended_model=report.recommended_model,
        report_markdown=report.to_markdown(),
        top_3=top3,
    )


@router.get("/sdlc/catalogue", tags=["SDLC"])
def get_catalogue():
    """Return the full SDLC model catalogue."""
    return {"catalogue": _agent.print_catalogue()}


# ── Full Pipeline ────────────────────────────────────────────────────────────

@router.post("/pipeline/text", response_model=FullPipelineResponse, tags=["Pipeline"])
def full_pipeline_text(body: FullPipelineTextRequest):
    """Full pipeline: text description → direct LLM requirements (no database/search) → SDLC recommendation."""
    try:
        doc_id = body.document_id or f"text_{uuid.uuid4().hex[:8]}"
        user_inst = body.query.strip() if body.query else ""
        requirements_md = generate_requirements_direct(
            project_description=body.project_description,
            user_instruction=user_inst,
            inbuilt_instruction=TEXT_REQUIREMENTS_INSTRUCTION,
        )

        # Step 2: SDLC recommendation
        combined = f"{body.project_description}\n\n--- EXTRACTED REQUIREMENTS ---\n{requirements_md}"
        report = _agent.recommend_from_text(
            combined,
            project_name=body.project_name,
            project_description=body.project_description,
        )

        # Step 3: Persist outputs
        safe = _safe_output_name(body.project_name)
        (OUTPUT_DIR / f"{safe}_requirements.md").write_text(requirements_md, encoding="utf-8")
        (OUTPUT_DIR / f"{safe}_sdlc_report.md").write_text(report.to_markdown(), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text pipeline failed: {exc}")

    return FullPipelineResponse(
        project_name=body.project_name,
        document_id=doc_id,
        requirements_markdown=requirements_md,
        recommended_model=report.recommended_model,
        sdlc_report_markdown=report.to_markdown(),
    )


@router.post("/pipeline/pdf", response_model=FullPipelineResponse, tags=["Pipeline"])
async def full_pipeline_pdf(
    file: UploadFile = File(...),
    project_name: str = Form("Finance Project"),
    user_id: str = Form("default_user"),
    query: str = Form("Extract all functional, security, compliance, and performance requirements from this document."),
):
    """Full pipeline: PDF upload → vector index → user prompt + inbuilt prompt vector search → LLM with evidence → SDLC."""
    try:
        filename = Path(file.filename or "upload.pdf").name
        doc_id = Path(filename).stem
        save_path = UPLOAD_DIR / f"{doc_id}_{filename}"
        save_path.write_bytes(await file.read())

        # Index PDF chunks into vector store
        generate_pdf_emb(str(save_path), user_id, doc_id)

        requested_functionality = (query or "").strip()
        if not requested_functionality:
            requested_functionality = "Extract all core functional, security, compliance, and performance requirements."

        # Vector search + RAG generation with user intent + inbuilt prompt
        requirements_md = answer_query_pdf_rag(
            user_intent=requested_functionality,
            document_id=doc_id,
            user_id=user_id,
            inbuilt_instruction=PDF_REQUIREMENTS_INSTRUCTION,
        )

        report = _agent.recommend_from_text(
            requirements_md,
            project_name=project_name,
            project_description=requested_functionality,
        )

        safe = _safe_output_name(project_name)
        (OUTPUT_DIR / f"{safe}_requirements.md").write_text(requirements_md, encoding="utf-8")
        (OUTPUT_DIR / f"{safe}_sdlc_report.md").write_text(report.to_markdown(), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF pipeline failed: {exc}")

    return FullPipelineResponse(
        project_name=project_name,
        document_id=doc_id,
        requirements_markdown=requirements_md,
        recommended_model=report.recommended_model,
        sdlc_report_markdown=report.to_markdown(),
    )
