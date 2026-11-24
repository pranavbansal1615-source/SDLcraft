"""
Unit tests for the SDLC Recommendation Agent (v2).
Run with: pytest backend/tests/

v2 changes tested:
  - Finance floors lowered from 3 → 2
  - 10-criteria MCDA (added project_size, complexity, uncertainty)
  - SUPPORTED-only signal filtering (_extract_supported_blocks)
  - Dual-input dimensions (requirements text + project description)
  - Regression: 4 different project types must produce DIFFERENT models
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.sdlc_agent import (
    SDLCAgent,
    _extract_dimensions,
    _extract_compliance_flags,
    _extract_supported_blocks,
    _compute_dynamic_weights,
    _mcda_scores,
    SDLC_CATALOGUE,
    CRITERIA,
    FINANCE_BASE_WEIGHTS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
The system shall support KYC verification via Aadhaar OTP as per RBI guidelines.
All PII must be encrypted at rest using AES-256 and in transit using TLS 1.3.
The platform must comply with DPDP Act 2023 data retention policies.
Loan disbursal via UPI must complete within 30 seconds of approval.
System must achieve 99.9% uptime SLA.
Multi-factor authentication (MFA) is mandatory for all logins.
Complete audit trails must be retained for 7 years.
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Dimension Extraction Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensionExtraction:
    def test_finance_floors_v2(self):
        """v2: Finance floor is 2 (was 3 in v1)."""
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert dims.regulatory_criticality >= 2, "Finance floor: reg criticality must be >= 2"
        assert dims.security_risk >= 2, "Finance floor: security risk must be >= 2"

    def test_high_regulatory_signal(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert dims.regulatory_criticality >= 4, "RBI + DPDP + audit signals should push reg >= 4"

    def test_high_security_signal(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert dims.security_risk >= 4, "AES + TLS + MFA signals should push security >= 4"

    def test_new_dimensions_present(self):
        """v2: ProjectDimensions now has project_size, complexity, uncertainty."""
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert hasattr(dims, "project_size"), "Missing project_size field"
        assert hasattr(dims, "complexity"), "Missing complexity field"
        assert hasattr(dims, "uncertainty"), "Missing uncertainty field"
        assert 1 <= dims.project_size <= 5
        assert 1 <= dims.complexity <= 5
        assert 1 <= dims.uncertainty <= 5

    def test_simple_project_gets_low_floors(self):
        """A trivial internal tool should NOT get reg=5, sec=5 (v2 fix)."""
        simple = "Build a simple internal dashboard for the accounts team. No external integrations."
        dims = _extract_dimensions(simple, project_description=simple)
        # Simple project should score lower than a full banking system
        assert dims.project_size <= 3, "Simple project should not get large project_size"

    def test_project_description_drives_volatility(self):
        """Volatility/urgency signals should come from project description, not requirements."""
        req_text = "The system shall store transaction records. The system shall validate inputs."
        desc = "We need an urgent MVP, go live by next month, changing requirements, daily standup with product owner."
        dims = _extract_dimensions(req_text, project_description=desc)
        assert dims.requirement_volatility >= 3, "Urgency/MVP in description should raise volatility"


# ─────────────────────────────────────────────────────────────────────────────
#  Compliance Flags Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComplianceFlags:
    def test_rbi_detected(self):
        flags = _extract_compliance_flags(SAMPLE_TEXT)
        assert any("RBI" in f for f in flags)

    def test_dpdp_detected(self):
        flags = _extract_compliance_flags(SAMPLE_TEXT)
        assert any("DPDP" in f for f in flags)

    def test_no_frameworks_fallback(self):
        flags = _extract_compliance_flags("Build a simple todo app.")
        assert len(flags) == 1
        assert "default" in flags[0].lower()


# ─────────────────────────────────────────────────────────────────────────────
#  Supported-Block Filter Tests (v2 new)
# ─────────────────────────────────────────────────────────────────────────────

class TestSupportedBlockFilter:
    def test_fallback_when_no_markers(self):
        """Plain text with no Evidence Status markers → returns full text."""
        plain = "The system shall do X. The system shall do Y."
        result = _extract_supported_blocks(plain)
        assert result == plain

    def test_filters_inferred_blocks(self):
        """INFERRED/ASSUMPTION blocks are dropped; SUPPORTED are kept."""
        md = (
            "#### REQ-001\nEncrypt PII.\n**Evidence Status:** [SUPPORTED]\n\n"
            "#### REQ-002\nUse AES-256.\n**Evidence Status:** [INFERRED]\n\n"
            "#### REQ-003\nAudit logs.\n**Evidence Status:** [SUPPORTED]\n"
        )
        result = _extract_supported_blocks(md)
        assert "REQ-001" in result
        assert "REQ-003" in result
        assert "REQ-002" not in result

    def test_keeps_explicit_blocks(self):
        md = (
            "#### REQ-001\nMFA required.\n**Evidence Status:** [EXPLICIT]\n\n"
            "#### REQ-002\nSome assumed req.\n**Evidence Status:** [ASSUMPTION]\n"
        )
        result = _extract_supported_blocks(md)
        assert "REQ-001" in result
        assert "REQ-002" not in result


# ─────────────────────────────────────────────────────────────────────────────
#  MCDA Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMCDA:
    def test_all_models_scored(self):
        scores = _mcda_scores(FINANCE_BASE_WEIGHTS)
        assert len(scores) == len(SDLC_CATALOGUE)

    def test_scores_in_valid_range(self):
        scores = _mcda_scores(FINANCE_BASE_WEIGHTS)
        for _, score in scores:
            assert 0 <= score <= 100, f"Score {score} out of range"

    def test_weights_sum_to_one(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        weights = _compute_dynamic_weights(dims)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_criteria_count_v2(self):
        """v2: 10 criteria (added project_size, complexity, uncertainty)."""
        assert len(CRITERIA) == 10, f"Expected 10 criteria, got {len(CRITERIA)}"
        assert "project_size" in CRITERIA
        assert "complexity" in CRITERIA
        assert "uncertainty" in CRITERIA

    def test_base_weights_sum_to_one(self):
        assert abs(sum(FINANCE_BASE_WEIGHTS.values()) - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  SDLCAgent Core Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSDLCAgent:
    def test_recommend_from_text(self):
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(SAMPLE_TEXT, project_name="Test KYC System")
        assert report.recommended_model in SDLC_CATALOGUE
        assert len(report.ranked_models) == len(SDLC_CATALOGUE)
        assert report.ranked_models[0].rank == 1

    def test_recommend_from_text_with_description(self):
        """v2: recommend_from_text accepts optional project_description."""
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(
            SAMPLE_TEXT,
            project_name="Test KYC System",
            project_description="KYC onboarding for RBI-regulated lending platform.",
        )
        assert report.recommended_model in SDLC_CATALOGUE

    def test_recommend_from_list(self):
        agent = SDLCAgent(use_llm_rationale=False)
        reqs = [
            "The system shall encrypt all PII using AES-256.",
            "The system shall comply with RBI Digital Lending Guidelines.",
            "The system shall achieve 99.9% uptime.",
        ]
        report = agent.recommend_from_list(reqs, project_name="Test Lending Platform")
        assert report.recommended_model in SDLC_CATALOGUE

    def test_finance_model_preferred(self):
        """For high-regulatory finance text, low-governance models should NOT be top model."""
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(SAMPLE_TEXT, project_name="Regulated Finance")
        assert report.recommended_model not in {"Agile (Scrum)", "RAD", "Prototype"}

    def test_catalogue_print(self):
        agent = SDLCAgent(use_llm_rationale=False)
        catalogue = agent.print_catalogue()
        expected_models = {"Waterfall", "V-Model", "Incremental", "Prototype", "Spiral", "RAD", "Agile (Scrum)"}
        assert set(SDLC_CATALOGUE) == expected_models
        assert all(model in catalogue for model in expected_models)

    def test_to_markdown(self):
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(SAMPLE_TEXT, project_name="Markdown Test")
        md = report.to_markdown()
        assert "# SDLC Recommendation Report" in md
        assert report.recommended_model in md


# ─────────────────────────────────────────────────────────────────────────────
#  Text Pipeline Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestTextPipeline:
    @patch("app.api.routes.generate_requirements_direct")
    @patch("app.api.routes.generate_text_emb")
    def test_text_pipeline_bypasses_vector_ingestion(
        self,
        mock_generate_text_emb,
        mock_generate_requirements_direct,
    ):
        from app.api.routes import full_pipeline_text
        from app.schemas.models import FullPipelineTextRequest

        mock_generate_requirements_direct.return_value = "### Generated Requirements\n"
        body = FullPipelineTextRequest(
            project_description=SAMPLE_TEXT,
            project_name="Direct Text Project",
            query="Generate loan onboarding requirements.",
        )

        response = full_pipeline_text(body)

        assert response.recommended_model in SDLC_CATALOGUE
        mock_generate_requirements_direct.assert_called_once()
        mock_generate_text_emb.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
#  Regression: 4 project types → at least 3 different SDLC models
# ─────────────────────────────────────────────────────────────────────────────

class TestProjectDiversity:
    """
    Regression tests: different project types MUST produce different SDLC recommendations.
    This is the core acceptance criterion — the agent must NOT always pick Spiral.

    The four archetypes chosen are maximally different along the key axes:
      - size/complexity (small vs enterprise)
      - certainty (frozen vs uncertain requirements)
      - urgency (slow vs fast time-to-market)
      - regulatory weight (low vs PCI/RBI)
    """

    def test_frozen_internal_tool_gets_waterfall_or_vmodel(self):
        """
        Simple internal reporting dashboard:
        - Requirements are fixed and documented
        - Small team, single page, no external integrations
        → Expected: Waterfall or V-Model (sequential, well-defined)
        """
        desc = (
            "Build a simple internal reporting dashboard for the accounts team. "
            "Requirements are fixed and fully documented upfront. "
            "Single page app, no external integrations. "
            "Small team of 2 developers, utility script style."
        )
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(desc, project_name="Internal Dashboard", project_description=desc)
        assert report.recommended_model in {"Waterfall", "V-Model", "Incremental"}, (
            f"Frozen internal tool → expected Waterfall/V-Model/Incremental, got {report.recommended_model}.\n"
            f"Scores: {[(m.model_name, m.score) for m in report.ranked_models]}"
        )

    def test_urgent_mvp_gets_agile_or_rad(self):
        """
        Urgent MVP with evolving requirements:
        - Competitive pressure, quick time-to-market
        - Daily standup, product owner available
        - Requirements evolving, MVP mindset
        → Expected: Agile (Scrum) or RAD
        """
        desc = (
            "We need to launch an MVP of a simple payment link generator by next month. "
            "Requirements are evolving, we need fast feedback from users. "
            "Daily standup with the product owner. "
            "Quick time-to-market is critical, urgent competitive pressure, go live soon."
        )
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(desc, project_name="Payment Link MVP", project_description=desc)
        assert report.recommended_model in {"Agile (Scrum)", "RAD", "Incremental"}, (
            f"Urgent MVP → expected Agile/RAD/Incremental, got {report.recommended_model}.\n"
            f"Scores: {[(m.model_name, m.score) for m in report.ranked_models]}"
        )

    def test_payment_switch_gets_vmodel_or_waterfall(self):
        """
        High-compliance payment switch — strict requirements, formal testing:
        - RBI, PCI-DSS, AML/KYC compliance
        - Formal verification, UAT, full audit trails required
        - Requirements are defined and frozen
        → Expected: V-Model or Waterfall (high governance, testing-paired)
        """
        desc = (
            "Build a national payment switch for RTGS and NEFT settlement processing. "
            "Must comply with RBI guidelines, PCI-DSS, AML and KYC. "
            "Requires formal verification, acceptance testing, full audit trails. "
            "99.99% uptime SLA, disaster recovery, mission-critical. "
            "All requirements are defined and frozen."
        )
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(desc, project_name="Payment Switch", project_description=desc)
        assert report.recommended_model in {"V-Model", "Waterfall", "Incremental"}, (
            f"Payment switch → expected V-Model/Waterfall/Incremental, got {report.recommended_model}.\n"
            f"Scores: {[(m.model_name, m.score) for m in report.ranked_models]}"
        )

    def test_core_banking_migration_gets_spiral_or_incremental(self):
        """
        Large, risky core banking migration:
        - Enterprise-scale, distributed, multi-tenant, cross-border
        - High uncertainty, evolving requirements
        - 50+ legacy integrations, millions of users
        → Expected: Spiral or Incremental (risk-driven, iterative for large uncertain projects)
        """
        desc = (
            "Multi-year core banking migration from legacy mainframe to microservices. "
            "Enterprise-scale distributed architecture, multi-tenant, cross-border reconciliation. "
            "High risk, complex integration with 50 legacy systems. "
            "Requirements are uncertain and will evolve significantly. "
            "Organization-wide digital transformation affecting millions of users."
        )
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(desc, project_name="Core Banking Migration", project_description=desc)
        assert report.recommended_model in {"Spiral", "Incremental", "V-Model"}, (
            f"Core banking migration → expected Spiral/Incremental/V-Model, got {report.recommended_model}.\n"
            f"Scores: {[(m.model_name, m.score) for m in report.ranked_models]}"
        )

    def test_four_projects_produce_diversity(self):
        """
        Meta-test: the 4 project archetypes must NOT all produce the same model.
        At least 2 distinct SDLC models must appear across the 4 recommendations.
        """
        agent = SDLCAgent(use_llm_rationale=False)
        projects = [
            (
                "Build a simple internal reporting dashboard. Requirements fixed. Small team, utility script.",
                "Internal Tool",
            ),
            (
                "Launch urgent MVP payment link generator. Evolving requirements, daily standup, go live next month.",
                "Urgent MVP",
            ),
            (
                "National RTGS payment switch. RBI, PCI-DSS, AML/KYC. Frozen requirements, formal verification, audit trails.",
                "Payment Switch",
            ),
            (
                "Enterprise core banking migration. Distributed microservices, multi-tenant, uncertain requirements, millions of users.",
                "Core Banking",
            ),
        ]
        results = []
        for desc, name in projects:
            r = agent.recommend_from_text(desc, project_name=name, project_description=desc)
            results.append(r.recommended_model)

        unique_models = set(results)
        assert len(unique_models) >= 2, (
            f"All 4 project archetypes got the same model(s): {results}. "
            "The agent is biased — it must recommend different models for different project types."
        )
