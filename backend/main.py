"""
main.py — SDLcraft Entry Point
==============================

Demonstrates both capabilities of the SDLcraft pipeline:

  MODE 1 — From Document (PDF)
    • Upload a requirements document (PDF) → embed → query for requirements
    → pipe the extracted requirements into the SDLC agent.

  MODE 2 — From Text Description
    • Provide a free-text description of the software / project
    → embed → query for requirements
    → pipe the extracted requirements into the SDLC agent.

Run:
    python main.py

Or import individual components for integration into a FastAPI / Flask app.
"""

import os
import sys
from pathlib import Path

# ─── Import RAG module ───────────────────────────────────────────────────────
from app.rag.rag import (
    answer_query_from_pdf,
    answer_query_from_text,
    generate_pdf_emb,
    generate_text_emb,
    answer_query,
    generate_requirements,
)

# ─── Import SDLC Agent ───────────────────────────────────────────────────────
from app.agents.sdlc_agent import SDLCAgent


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline: Requirements RAG → SDLC Agent
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline_from_text(
    project_description: str,
    project_name: str = "Finance Project",
    query: str = "List all functional, security, and compliance requirements for this finance software project.",
    document_id: str | None = None,
    user_id: str = "sdlcraft_user",
    output_dir: str = "output",
) -> dict:
    """
    Full pipeline: Text description → RAG requirements extraction → SDLC recommendation.

    Returns
    -------
    dict with keys: 'requirements_text', 'sdlc_report', 'sdlc_markdown'
    """
    print(f"\n{'='*70}")
    print(f"  SDLcraft — Finance SDLC Pipeline (TEXT MODE)")
    print(f"  Project: {project_name}")
    print(f"{'='*70}\n")

    # Step 1: Send the complete text brief directly to the requirements LLM
    print("▶ Step 1: Generating requirements directly from the text brief...")
    requirements_text = generate_requirements(
        brief=project_description,
        question=f"{query}\n\nAdditional project context:\n{project_description}",
    )
    print("✔ Requirements extracted.\n")

    # Step 2: SDLC recommendation
    print("▶ Step 2: Running SDLC Recommendation Agent...")
    agent = SDLCAgent()
    # Feed BOTH the original description + extracted requirements for richer analysis
    combined_input = f"{project_description}\n\n--- EXTRACTED REQUIREMENTS ---\n{requirements_text}"
    report = agent.recommend_from_text(combined_input, project_name=project_name)
    print(f"✔ Recommended SDLC Model: {report.recommended_model}\n")

    # Step 3: Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    safe_name = project_name.lower().replace(" ", "_")

    req_path = out_path / f"{safe_name}_requirements.md"
    sdlc_path = out_path / f"{safe_name}_sdlc_report.md"

    req_path.write_text(requirements_text, encoding="utf-8")
    sdlc_path.write_text(report.to_markdown(), encoding="utf-8")

    print(f"▶ Outputs saved:")
    print(f"   Requirements : {req_path.resolve()}")
    print(f"   SDLC Report  : {sdlc_path.resolve()}")

    return {
        "requirements_text": requirements_text,
        "sdlc_report": report,
        "sdlc_markdown": report.to_markdown(),
    }


def run_pipeline_from_pdf(
    pdf_path: str,
    project_name: str = "Finance Project",
    query: str = "List all functional, security, and compliance requirements for this finance software project.",
    document_id: str | None = None,
    user_id: str = "sdlcraft_user",
    output_dir: str = "output",
) -> dict:
    """
    Full pipeline: PDF document → RAG requirements extraction → SDLC recommendation.
    """
    print(f"\n{'='*70}")
    print(f"  SDLcraft — Finance SDLC Pipeline (PDF MODE)")
    print(f"  Project: {project_name}")
    print(f"  PDF    : {pdf_path}")
    print(f"{'='*70}\n")

    doc_id = document_id or Path(pdf_path).stem

    # Step 1: Index PDF
    print("▶ Step 1: Indexing PDF document...")
    result = generate_pdf_emb(pdf_path, user_id, doc_id)
    print(f"   {result}")

    # Step 2: Extract requirements via RAG
    print("▶ Step 2: Extracting requirements via RAG...")
    requirements_text = answer_query(query, doc_id)
    print("✔ Requirements extracted.\n")

    # Step 3: SDLC recommendation
    print("▶ Step 3: Running SDLC Recommendation Agent...")
    agent = SDLCAgent()
    report = agent.recommend_from_text(requirements_text, project_name=project_name)
    print(f"✔ Recommended SDLC Model: {report.recommended_model}\n")

    # Step 4: Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)
    safe_name = project_name.lower().replace(" ", "_")

    req_path = out_path / f"{safe_name}_requirements.md"
    sdlc_path = out_path / f"{safe_name}_sdlc_report.md"

    req_path.write_text(requirements_text, encoding="utf-8")
    sdlc_path.write_text(report.to_markdown(), encoding="utf-8")

    print(f"▶ Outputs saved:")
    print(f"   Requirements : {req_path.resolve()}")
    print(f"   SDLC Report  : {sdlc_path.resolve()}")

    return {
        "requirements_text": requirements_text,
        "sdlc_report": report,
        "sdlc_markdown": report.to_markdown(),
    }


def sdlc_only(
    requirements_text: str,
    project_name: str = "Finance Project",
) -> dict:
    """
    Skip the RAG step — run only the SDLC agent on already-extracted requirements.
    Useful when you have requirement strings from another source.
    """
    agent = SDLCAgent()
    report = agent.recommend_from_text(requirements_text, project_name=project_name)
    return {
        "recommended_model": report.recommended_model,
        "sdlc_report": report,
        "sdlc_markdown": report.to_markdown(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DESCRIPTION = """
Project: Digital Lending & KYC Onboarding Platform

We are building a digital loan origination system for a regulated NBFC in India.
The platform will handle the full lifecycle from customer onboarding to loan disbursal.

Key requirements:
- KYC verification via Aadhaar OTP and Video KYC (VCIP) as per RBI guidelines.
- Credit bureau integration (CIBIL / Equifax) for instant credit scoring.
- Loan application and eligibility assessment within 60 seconds.
- AML screening against OFAC and domestic watchlists at the point of onboarding.
- PII data must be encrypted at rest (AES-256) and in transit (TLS 1.3).
- Multi-factor authentication (MFA) for all customer and agent login flows.
- RBAC with field-level masking — agents cannot see full Aadhaar / PAN numbers.
- Complete audit trail with tamper-evident, immutable logs retained for 7 years (DPDP Act).
- Loan disbursal via UPI / NEFT with transaction confirmation in under 30 seconds.
- System must achieve 99.9% uptime SLA with sub-500ms p95 API latency.
- Adverse action notices must be generated and sent within 24 hours of rejection (ECOA).
- SOX-compliant financial reporting for all loan book entries.
- The system must support concurrent access by 10,000 agents without performance degradation.
- Mobile-first responsive UI for customers; portal for operations team.
- Real-time fraud detection engine with anomaly scoring on each transaction.
"""


if __name__ == "__main__":
    # Run the full text-based pipeline with the sample description
    result = run_pipeline_from_text(
        project_description=SAMPLE_DESCRIPTION,
        project_name="Digital Lending & KYC Platform",
        output_dir="output",
    )

    print("\n" + "="*70)
    print("  SDLC Recommendation Summary")
    print("="*70)
    print(f"  Recommended Model : {result['sdlc_report'].recommended_model}")
    print(f"  Top 3 Models      :")
    for m in result["sdlc_report"].ranked_models[:3]:
        print(f"    {m.rank}. {m.model_name}  —  {m.score:.1f}  ({m.suitability})")
    print("="*70)
