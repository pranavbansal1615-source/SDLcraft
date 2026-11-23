"""
Unit tests for the SDLC Recommendation Agent.
Run with: pytest backend/tests/
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
    _compute_dynamic_weights,
    _mcda_scores,
    SDLC_CATALOGUE,
    CRITERIA,
)


SAMPLE_TEXT = """
The system shall support KYC verification via Aadhaar OTP as per RBI guidelines.
All PII must be encrypted at rest using AES-256 and in transit using TLS 1.3.
The platform must comply with DPDP Act 2023 data retention policies.
Loan disbursal via UPI must complete within 30 seconds of approval.
System must achieve 99.9% uptime SLA.
Multi-factor authentication (MFA) is mandatory for all logins.
Complete audit trails must be retained for 7 years.
"""


class TestDimensionExtraction:
    def test_finance_floors(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert dims.regulatory_criticality >= 3, "Finance floor: reg criticality must be >= 3"
        assert dims.security_risk >= 3, "Finance floor: security risk must be >= 3"

    def test_high_regulatory_signal(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert dims.regulatory_criticality >= 4, "RBI + DPDP + audit signals should push reg >= 4"

    def test_high_security_signal(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        assert dims.security_risk >= 4, "AES + TLS + MFA signals should push security >= 4"


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


class TestMCDA:
    def test_all_models_scored(self):
        from app.agents.sdlc_agent import FINANCE_BASE_WEIGHTS
        scores = _mcda_scores(FINANCE_BASE_WEIGHTS)
        assert len(scores) == len(SDLC_CATALOGUE)

    def test_scores_in_valid_range(self):
        from app.agents.sdlc_agent import FINANCE_BASE_WEIGHTS
        scores = _mcda_scores(FINANCE_BASE_WEIGHTS)
        for _, score in scores:
            assert 0 <= score <= 100, f"Score {score} out of range"

    def test_weights_sum_to_one(self):
        dims = _extract_dimensions(SAMPLE_TEXT)
        weights = _compute_dynamic_weights(dims)
        assert abs(sum(weights.values()) - 1.0) < 0.01


class TestSDLCAgent:
    def test_recommend_from_text(self):
        agent = SDLCAgent(use_llm_rationale=False)
        report = agent.recommend_from_text(SAMPLE_TEXT, project_name="Test KYC System")
        assert report.recommended_model in SDLC_CATALOGUE
        assert len(report.ranked_models) == len(SDLC_CATALOGUE)
        assert report.ranked_models[0].rank == 1

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
