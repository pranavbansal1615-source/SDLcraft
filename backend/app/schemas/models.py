"""
Pydantic request / response schemas for the SDLcraft API.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────────────────

class TextIngestionRequest(BaseModel):
    """Ingest free-text project description for RAG indexing."""
    text: str = Field(..., description="Raw text describing the software / requirements.")
    project_name: str = Field(..., description="Human-readable project name.")
    document_id: Optional[str] = Field(None, description="Optional stable document ID. Auto-generated if omitted.")
    user_id: str = Field("default_user", description="User / session identifier.")


class RequirementsQueryRequest(BaseModel):
    """Query requirements from an already-indexed document."""
    query: str = Field(..., description="Natural language question about the requirements.")
    document_id: str = Field(..., description="The document_id used during ingestion.")
    user_id: str = Field("default_user", description="User or session identifier.")


class SDLCRecommendRequest(BaseModel):
    """Request SDLC recommendation from requirement strings."""
    requirements: List[str] = Field(..., description="List of requirement strings.")
    project_name: str = Field("Finance Project", description="Project name for the report.")


class FullPipelineTextRequest(BaseModel):
    """Run the full pipeline: text → requirements → SDLC recommendation."""
    project_description: str = Field(..., description="Free-text project description.")
    project_name: str = Field("Finance Project")
    query: str = Field(
        "Generate the complete requirement set for this project.",
        description="Optional additional instruction for the requirements LLM."
    )
    user_id: str = Field("default_user")
    document_id: Optional[str] = None


# ── Response Schemas ─────────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    status: str
    document_id: str
    message: str


class RequirementsResponse(BaseModel):
    document_id: str
    query: str
    requirements_markdown: str


class SDLCResponse(BaseModel):
    project_name: str
    recommended_model: str
    report_markdown: str
    top_3: List[dict]


class FullPipelineResponse(BaseModel):
    project_name: str
    document_id: str
    requirements_markdown: str
    recommended_model: str
    sdlc_report_markdown: str
