"""
sdlc_agent.py — SDLC Recommendation Agent for SDLcraft (Finance Projects)

Given a set of extracted software requirements (plain text or structured list),
this agent:
  1. Analyses project characteristics (dimensions) from the requirements text.
  2. Runs Multi-Criteria Decision Analysis (MCDA) with finance-aware dynamic weights.
  3. Performs sensitivity analysis (+/- 20 %) to validate robustness.
  4. Selects the best-fit SDLC model from a comprehensive catalogue.
  5. Generates a tailored workflow for the chosen model.
  6. Returns a fully-structured SDLCReport (Pydantic) and a Markdown summary.

Usage
-----
    from sdlc_agent import SDLCAgent

    agent = SDLCAgent()

    # From plain-text requirements / project description
    report = agent.recommend_from_text(
        requirements_text=\"\"\"
        - The system shall support KYC verification via Aadhaar OCI.
        - Loan disbursal must complete within 30 seconds of approval.
        - All PII must be encrypted at rest using AES-256.
        - The system shall comply with RBI Digital Lending Guidelines 2025.
        \"\"\",
        project_name="Digital Lending Platform"
    )

    print(report.to_markdown())

    # From a structured list of requirement strings
    report2 = agent.recommend_from_list(
        requirements=["REQ-01: ...", "REQ-02: ..."],
        project_name="Payments Gateway"
    )
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

from app.core.llm import invoke_llm

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
#  SDLC Model Catalogue
#  Each model carries: description, criteria scores (1-100), strengths, risks,
#  and finance-domain applicability notes.
# ─────────────────────────────────────────────────────────────────────────────

SDLC_CATALOGUE: Dict[str, Dict[str, Any]] = {
    # ── Sequential Models ────────────────────────────────────────────────────
    "Waterfall": {
        "iterative": False,
        "description": (
            "The first model to formalize the software development process. Phases run "
            "strictly one after another, with emphasis on requirements, design and "
            "documentation in the early stages. Each phase has defined deliverables and "
            "a review before the next begins."
        ),
        "criteria": {
            "compliance":    80.0,
            "security":      60.0,
            "speed":         30.0,
            "adaptability":  20.0,
            "safety":        70.0,
            "testing_rigor": 70.0,
            "stakeholder":   50.0,
        },
        "strengths": [
            "Easy to understand, use and manage because of its rigid structure.",
            "Clearly defined stages, milestones and deliverables.",
            "Process and results are well documented, which helps auditors.",
        ],
        "risks": [
            "No working software until late in the life cycle.",
            "Cannot accommodate changing requirements; changing scope mid-project can end it.",
            "Integration happens in a single \"big bang\" at the end, so technical and "
            "business bottlenecks surface late.",
            "Progress is difficult to measure within a stage.",
        ],
        "best_when": (
            "Requirements are clear, documented and fixed; the product definition is "
            "stable; the technology is well understood; and the project is short."
        ),
        "finance_note": "Limited applicability; only for simple internal tools with frozen requirements.",
        "phases": [
            "Requirements",
            "System Design",
            "Implementation",
            "Testing",
            "Deployment",
            "Maintenance",
        ],
    },
    "V-Model": {
        "iterative": False,
        "description": (
            "A sequential model, also called the Verification and Validation model, in "
            "which every development phase is paired with a testing phase. Tests are "
            "planned and designed early, during the corresponding development stage, "
            "and executed in a V-shaped sequence."
        ),
        "criteria": {
            "compliance":    95.0,
            "security":      80.0,
            "speed":         40.0,
            "adaptability":  30.0,
            "safety":        98.0,
            "testing_rigor": 98.0,
            "stakeholder":   60.0,
        },
        "strengths": [
            "Strong traceability: each requirement maps to a test activity.",
            "Requirements must be defined before design begins.",
            "Easy to use, and progress can be tracked accurately.",
        ],
        "risks": [
            "Cannot handle concurrent events or repeated iterations of a phase.",
            "Poor at absorbing requirement changes once work is under way.",
            "No built-in risk analysis activities.",
            "Working software is delivered only at the end.",
        ],
        "best_when": (
            "Requirements are well defined and fixed; the product definition is stable; "
            "the technology is well understood; and the project is short."
        ),
        "finance_note": "Suitable for RTGS/NEFT infrastructure and regulatory reporting engines.",
        "phases": [
            "Requirements Analysis",
            "System Design",
            "Architecture Design",
            "Detailed Design",
            "Coding",
            "Unit Testing",
            "Integration Testing",
            "System Testing",
            "Acceptance Testing",
        ],
    },
    # ── Iterative / Incremental Models ───────────────────────────────────────
    "Incremental": {
        "iterative": True,
        "description": (
            "Builds a partial implementation of the system first and adds functionality "
            "in successive increments. Each increment passes through requirements, "
            "design, implementation and testing, and the process repeats until the "
            "complete system is delivered. Major requirements are defined upfront; "
            "details may evolve. Can be combined with other models."
        ),
        "criteria": {
            "compliance":    78.0,
            "security":      72.0,
            "speed":         78.0,
            "adaptability":  75.0,
            "safety":        75.0,
            "testing_rigor": 80.0,
            "stakeholder":   82.0,
        },
        "strengths": [
            "Working functionality is available early, and results arrive periodically.",
            "Testing and debugging are easier in small iterations.",
            "High-risk parts can be built first; lessons carry into later increments.",
            "Scope changes are less costly than in sequential models.",
        ],
        "risks": [
            "Needs more resources and more management attention.",
            "Not all requirements are gathered upfront, so architecture or design issues may emerge.",
            "Defining increments may require defining the whole system.",
            "The end of the project may be hard to predict.",
        ],
        "best_when": (
            "Major requirements are clear but some details will evolve; early time-to-market "
            "matters; new technology is involved; or some features are high-risk."
        ),
        "finance_note": "Suitable for phased delivery of finance platforms with clear release boundaries.",
        "phases": [
            "Increment Planning",
            "Requirements & Design",
            "Development & Testing",
            "Increment Release",
            "Feedback & Next Increment Planning",
        ],
    },
    "Prototype": {
        "iterative": True,
        "description": (
            "Builds a working prototype that demonstrates the functionality of the "
            "intended product, so customer requirements are understood early. The "
            "prototype is reviewed by users and refined repeatedly. Common variants are "
            "rapid, evolutionary, incremental and extreme prototyping. In practice it is "
            "often used as a technique alongside another model."
        ),
        "criteria": {
            "compliance":    60.0,
            "security":      55.0,
            "speed":         85.0,
            "adaptability":  92.0,
            "safety":        55.0,
            "testing_rigor": 60.0,
            "stakeholder":   92.0,
        },
        "strengths": [
            "Users are involved before implementation and understand the system better.",
            "Missing or confusing functions are identified early.",
            "Defects are found sooner, reducing time and cost.",
            "Well suited to interaction-heavy systems such as web interfaces.",
        ],
        "risks": [
            "Over-reliance on the prototype can lead to insufficient requirements analysis.",
            "Users may confuse the prototype with the final system.",
            "Scope can expand beyond the original plan.",
            "Developers may reuse prototype code that is not technically fit for production.",
            "Effort spent on prototypes can balloon if not monitored.",
        ],
        "best_when": (
            "The system involves heavy end-user interaction and requirements are unclear "
            "(for example online systems and user interfaces)."
        ),
        "finance_note": (
            "Best for validating user journeys and requirements before a production model "
            "is chosen. Security, performance and compliance need separate validation."
        ),
        "phases": [
            "Basic Requirement Identification",
            "Initial Prototype Development",
            "Prototype Review",
            "Revise & Enhance Prototype",
        ],
    },
    "RAD": {
        "iterative": True,
        "description": (
            "Rapid Application Development is an iterative, incremental model built on "
            "prototyping with minimal upfront planning. Users take part in every phase, "
            "requirements are gathered through workshops or focus groups, and small teams "
            "produce reusable components. The system must be modularizable."
        ),
        "criteria": {
            "compliance":    58.0,
            "security":      55.0,
            "speed":         96.0,
            "adaptability":  88.0,
            "safety":        55.0,
            "testing_rigor": 62.0,
            "stakeholder":   90.0,
        },
        "strengths": [
            "Reduced development time and short iterations with RAD tools.",
            "Changing requirements can be accommodated, and progress can be measured.",
            "Encourages customer feedback and increases component reuse.",
            "Integrating from the start avoids many late integration issues.",
        ],
        "risks": [
            "Depends on technically strong team members and highly skilled developers and designers.",
            "Only modularizable systems qualify.",
            "Modeling and automated code generation are costly, so it is unsuited to cheaper projects.",
            "Requires user involvement throughout; management complexity is higher.",
        ],
        "best_when": (
            "The system is component-based and can be modularized in about 2-3 months; "
            "skilled designers and business-knowledgeable users are available; and the "
            "budget covers modeling and code-generation tools."
        ),
        "finance_note": "Useful for time-critical, lower-risk finance applications with accessible stakeholders.",
        "phases": [
            "Requirements Planning",
            "User Design Workshops",
            "Rapid Construction",
            "Testing & Cutover",
        ],
    },
    # ── Risk-Driven Model ────────────────────────────────────────────────────
    "Spiral": {
        "iterative": True,
        "description": (
            "A risk-driven model proposed by Barry Boehm. It keeps the strengths of "
            "Waterfall while adding risk analysis, risk management and support processes. "
            "Each cycle covers planning, risk analysis, engineering and evaluation, and "
            "can use prototyping or RAD to build the product."
        ),
        "criteria": {
            "compliance":    85.0,
            "security":      88.0,
            "speed":         55.0,
            "adaptability":  80.0,
            "safety":        90.0,
            "testing_rigor": 85.0,
            "stakeholder":   70.0,
        },
        "strengths": [
            "Explicit risk analysis in every cycle; risky parts are built first.",
            "Changing requirements can be accommodated.",
            "Extensive use of prototypes captures requirements more accurately.",
            "Users see the system early.",
        ],
        "risks": [
            "Complex process and management; needs experienced risk analysts.",
            "The project end may not be known early, and the spiral can continue indefinitely.",
            "Many intermediate stages require heavy documentation.",
            "Expensive for small or low-risk projects.",
        ],
        "best_when": (
            "The project is medium to high risk, long-term, and requirements are complex or "
            "uncertain; significant changes are expected; and risk evaluation matters."
        ),
        "finance_note": "Suited to large digital transformation or core banking migration projects.",
        "phases": [
            "Planning (Objectives, Constraints)",
            "Risk Analysis & Prototyping",
            "Engineering & Testing",
            "Evaluation & Next Cycle Planning",
        ],
    },
    # ── Agile ────────────────────────────────────────────────────────────────
    # Note: Agile is not covered in the slides; this entry draws on general knowledge.
    "Agile (Scrum)": {
        "iterative": True,
        "description": (
            "Agile is a set of principles for iterative, collaborative delivery; Scrum is "
            "a common framework for applying it, using time-boxed sprints, daily standups, "
            "sprint reviews and retrospectives. It prioritizes speed, stakeholder feedback "
            "and responsiveness to change. It has no built-in compliance gates, so "
            "regulated projects must add them explicitly."
        ),
        "criteria": {
            "compliance":    50.0,
            "security":      55.0,
            "speed":         95.0,
            "adaptability":  95.0,
            "safety":        50.0,
            "testing_rigor": 65.0,
            "stakeholder":   95.0,
        },
        "strengths": [
            "Fast delivery and high customer responsiveness.",
            "High stakeholder visibility and short feedback cycles.",
            "Easy to adopt for small cross-functional teams.",
        ],
        "risks": [
            "No built-in compliance gates; regulated finance work needs added controls.",
            "Security work can be deprioritized under sprint velocity pressure.",
            "Audit trail and traceability need additional tooling or process.",
        ],
        "requires_added_controls": True,
        "best_when": (
            "Requirements evolve quickly and the team can layer explicit compliance, "
            "security and audit controls onto the process."
        ),
        "finance_note": "Use for fast-changing products where explicit finance controls can be added.",
        "phases": [
            "Product Backlog Refinement",
            "Sprint Planning",
            "Sprint Execution (typically 1-4 weeks)",
            "Sprint Review & Demo",
            "Sprint Retrospective",
        ],
    },
}

# MCDA criteria with finance-domain baseline weights
CRITERIA = ["compliance", "security", "speed", "adaptability", "safety", "testing_rigor", "stakeholder"]

FINANCE_BASE_WEIGHTS: Dict[str, float] = {
    "compliance":    0.22,
    "security":      0.20,
    "safety":        0.18,
    "testing_rigor": 0.15,
    "adaptability":  0.10,
    "speed":         0.09,
    "stakeholder":   0.06,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic-free dataclasses for lightweight portability
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelScore:
    model_name: str
    score: float
    rank: int
    suitability: str
    justification: str
    strengths: List[str]
    risks: List[str]
    finance_note: str


@dataclass
class ProjectDimensions:
    regulatory_criticality: int   # 1-5
    security_risk: int            # 1-5
    requirement_volatility: int   # 1-5
    system_failure_impact: int    # 1-5
    testing_rigor_needed: int     # 1-5
    stakeholder_availability: Optional[int]  # 1-5 or None
    time_to_market_urgency: Optional[int]    # 1-5 or None

    @staticmethod
    def label(val: Optional[int]) -> str:
        if val is None:
            return "Unknown"
        return {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Very High"}.get(val, str(val))


@dataclass
class SDLCReport:
    project_name: str
    input_summary: str
    dimensions: ProjectDimensions
    dynamic_weights: Dict[str, float]
    ranked_models: List[ModelScore]
    recommended_model: str
    recommendation_rationale: str
    tailored_workflow: List[Dict[str, Any]]
    sensitivity_analysis: Dict[str, Any]
    finance_compliance_flags: List[str]
    llm_rationale: str = ""   # optional LLM-generated narrative

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ─────────────────────────────────────────────────────────────────────────────
#  Dimension Extraction (rule-based, finance-aware)
# ─────────────────────────────────────────────────────────────────────────────

_REGULATORY_SIGNALS = [
    "rbi", "sebi", "irdai", "pci-dss", "pci dss", "aml", "kyc", "gdpr", "dpdp",
    "fcra", "ecoa", "glba", "bsa", "sox", "audit", "compliance", "regulatory",
    "statutory", "legal", "consent", "retention", "disclosure", "ofac", "pep",
    "sanctions", "anti-money", "finra", "cfpb",
]
_SECURITY_SIGNALS = [
    "encrypt", "aes", "tls", "ssl", "mfa", "multi-factor", "rbac", "token",
    "session timeout", "masking", "pii", "zero-trust", "siem", "sast", "dast",
    "penetration", "firewall", "vault", "hsm", "authentication", "authorization",
    "password", "credential", "mtls", "intrusion", "ddos",
]
_VOLATILITY_SIGNALS = [
    "user story", "prototype", "feedback", "iterative", "mvp", "discovery",
    "pivot", "changing requirements", "evolving", "exploratory",
]
_SAFETY_SIGNALS = [
    "critical", "mission-critical", "failure", "downtime", "disaster recovery",
    "sla", "99.9", "uptime", "resilience", "fault tolerance", "high availability",
    "data loss", "catastrophic", "irrecoverable",
]
_TESTING_SIGNALS = [
    "unit test", "integration test", "regression", "qa", "quality assurance",
    "test coverage", "uam", "uat", "acceptance test", "formal verification",
    "v-model", "test plan", "test case",
]
_URGENCY_SIGNALS = [
    "urgent", "asap", "immediately", "deadline", "fast", "quick", "time-to-market",
    "launch by", "go live", "competitive pressure", "market window",
]
_STAKEHOLDER_SIGNALS = [
    "daily standup", "daily sync", "product owner", "dedicated stakeholder",
    "continuous collaboration", "business analyst", "full-time", "available",
]


def _count_signals(text: str, signals: List[str]) -> int:
    low = text.lower()
    return sum(1 for s in signals if s in low)


def _extract_dimensions(text: str) -> ProjectDimensions:
    reg   = _count_signals(text, _REGULATORY_SIGNALS)
    sec   = _count_signals(text, _SECURITY_SIGNALS)
    vol   = _count_signals(text, _VOLATILITY_SIGNALS)
    safe  = _count_signals(text, _SAFETY_SIGNALS)
    test  = _count_signals(text, _TESTING_SIGNALS)
    urg   = _count_signals(text, _URGENCY_SIGNALS)
    stake = _count_signals(text, _STAKEHOLDER_SIGNALS)

    def scale(count: int, low: int, high: int) -> int:
        """Map raw count to 1-5 scale."""
        if count == 0:
            return 1
        if count >= high:
            return 5
        return max(1, min(5, 1 + round((count - low) / max(high - low, 1) * 4)))

    # Finance default: regulatory and security always have elevated floors
    reg_criticality = max(3, scale(reg, 1, 5))   # finance floor = 3
    security_risk   = max(3, scale(sec, 1, 6))
    volatility      = scale(vol, 0, 4)
    failure_impact  = max(3, scale(safe, 1, 4))
    testing_rigor   = max(3, scale(test, 1, 5))
    urgency         = scale(urg, 1, 4) if urg > 0 else None
    stakeholder     = scale(stake, 1, 3) if stake > 0 else None

    return ProjectDimensions(
        regulatory_criticality=reg_criticality,
        security_risk=security_risk,
        requirement_volatility=volatility,
        system_failure_impact=failure_impact,
        testing_rigor_needed=testing_rigor,
        time_to_market_urgency=urgency,
        stakeholder_availability=stakeholder,
    )


def _extract_compliance_flags(text: str) -> List[str]:
    """Return a list of detected regulatory frameworks from the requirements text."""
    patterns = {
        "RBI Guidelines": r"\brbi\b",
        "SEBI Regulations": r"\bsebi\b",
        "DPDP Act 2023": r"\bdpdp\b",
        "PCI-DSS": r"\bpci[-\s]?dss\b",
        "AML / KYC": r"\b(aml|kyc)\b",
        "GDPR": r"\bgdpr\b",
        "FCRA": r"\bfcra\b",
        "ECOA": r"\becoa\b",
        "GLBA": r"\bglba\b",
        "SOX": r"\bsox\b",
        "BSA": r"\bbsa\b",
        "OFAC Sanctions": r"\bofac\b",
        "CFPB": r"\bcfpb\b",
        "ISO 27001": r"\biso\s*27001\b",
    }
    flags = []
    for label, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            flags.append(label)
    return flags if flags else ["No specific regulatory framework detected (apply finance-domain defaults)"]


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamic Weight Computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dynamic_weights(dims: ProjectDimensions) -> Dict[str, float]:
    """
    Adjust MCDA weights dynamically based on extracted project dimensions.
    Finance domain always applies elevated floors for compliance and security.
    """
    w = dict(FINANCE_BASE_WEIGHTS)

    # Boost compliance weight if regulatory criticality is very high
    if dims.regulatory_criticality >= 5:
        w["compliance"] += 0.05
        w["speed"] -= 0.03
        w["stakeholder"] -= 0.02

    # Boost security if security risk is very high
    if dims.security_risk >= 5:
        w["security"] += 0.04
        w["adaptability"] -= 0.02
        w["stakeholder"] -= 0.02

    # Boost speed if urgency is known and high
    if dims.time_to_market_urgency and dims.time_to_market_urgency >= 4:
        w["speed"] += 0.04
        w["compliance"] -= 0.02
        w["safety"] -= 0.02

    # Boost adaptability for volatile requirements
    if dims.requirement_volatility >= 4:
        w["adaptability"] += 0.03
        w["testing_rigor"] -= 0.03

    # Normalize to sum = 1.0
    total = sum(w.values())
    return {k: round(v / total, 4) for k, v in w.items()}


# ─────────────────────────────────────────────────────────────────────────────
#  MCDA Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _suitability_label(score: float) -> str:
    if score >= 85.0:
        return "Highly Recommended"
    if score >= 72.0:
        return "Recommended"
    if score >= 58.0:
        return "Moderate"
    return "Unsuitable"


def _mcda_scores(weights: Dict[str, float]) -> List[Tuple[str, float]]:
    scores = []
    for model, data in SDLC_CATALOGUE.items():
        raw = data["criteria"]
        score = sum(raw.get(c, 50.0) * weights.get(c, 0.0) for c in CRITERIA)
        scores.append((model, round(score, 2)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def _build_ranked_models(scores: List[Tuple[str, float]]) -> List[ModelScore]:
    ranked = []
    for rank, (name, score) in enumerate(scores, start=1):
        data = SDLC_CATALOGUE[name]
        ranked.append(ModelScore(
            model_name=name,
            score=score,
            rank=rank,
            suitability=_suitability_label(score),
            justification=data["description"],
            strengths=data["strengths"],
            risks=data["risks"],
            finance_note=data["finance_note"],
        ))
    return ranked


# ─────────────────────────────────────────────────────────────────────────────
#  Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _sensitivity_analysis(base_weights: Dict[str, float], delta: float = 0.20) -> Dict[str, Any]:
    base_scores = dict(_mcda_scores(base_weights))
    base_top = max(base_scores, key=base_scores.get)
    variations = []
    all_stable = True

    for criterion in CRITERIA:
        for factor, label in [(1 + delta, f"+{int(delta*100)}%"), (1 - delta, f"-{int(delta*100)}%")]:
            new_w = dict(base_weights)
            new_w[criterion] = base_weights[criterion] * factor
            # Re-normalise others proportionally
            others_sum = sum(base_weights[c] for c in CRITERIA if c != criterion)
            remainder = 1.0 - new_w[criterion]
            if others_sum > 0:
                for c in CRITERIA:
                    if c != criterion:
                        new_w[c] = base_weights[c] * (remainder / others_sum)

            p_scores = dict(_mcda_scores(new_w))
            top_m = max(p_scores, key=p_scores.get)
            is_stable = (top_m == base_top)
            if not is_stable:
                all_stable = False

            variations.append({
                "criterion": criterion,
                "variation": label,
                "top_model": top_m,
                "top_score": round(p_scores[top_m], 1),
                "stable": is_stable,
            })

    return {
        "base_recommended": base_top,
        "is_robust": all_stable,
        "delta_pct": int(delta * 100),
        "summary": (
            f"'{base_top}' remains #1 across all ±{int(delta*100)}% weight perturbations."
            if all_stable else
            f"Ranking shifts when weights are perturbed by ±{int(delta*100)}%."
        ),
        "variations": variations,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Tailored Workflow Generator
# ─────────────────────────────────────────────────────────────────────────────

def _build_workflow(
    model_name: str,
    compliance_flags: List[str],
    dims: ProjectDimensions
) -> List[Dict[str, Any]]:
    """Build a finance-tailored workflow for the selected SDLC model."""
    phases_raw = SDLC_CATALOGUE[model_name]["phases"]
    frameworks_str = ", ".join(compliance_flags[:4]) if compliance_flags else "Finance Domain Standards"

    workflow = []
    for idx, phase_name in enumerate(phases_raw, start=1):
        is_first = idx == 1
        is_last  = idx == len(phases_raw)

        entry = "Previous phase deliverables signed off; no unresolved critical issues."
        if is_first:
            entry = "Approved requirements specification; stakeholder alignment complete."

        exit_c = "All deliverables reviewed; evidence archived; no open critical defects."
        if is_last:
            exit_c = "Production live; zero compliance incidents; compliance officer sign-off."

        sec_controls = [
            "Threat model reviewed and signed off.",
            "SAST/DAST scan passing with zero critical findings.",
            "RBAC and MFA enforced on all privileged functions.",
        ]
        comp_checks = [
            f"Regulatory obligations traced to evidence ({frameworks_str}).",
            "Immutable audit logs verified for all state-changing operations.",
        ]

        workflow.append({
            "phase_number": idx,
            "phase_name": phase_name,
            "entry_criteria": entry,
            "exit_criteria": exit_c,
            "security_controls": sec_controls,
            "compliance_checkpoints": comp_checks,
            "human_approval_gate": is_first or is_last or idx == len(phases_raw) // 2,
        })

    return workflow


# ─────────────────────────────────────────────────────────────────────────────
#  Markdown Renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(report: SDLCReport) -> str:
    dims = report.dimensions
    weights = report.dynamic_weights
    lines = [
        "# SDLC Recommendation Report",
        f"**Project**: {report.project_name}",
        f"**Recommended SDLC Model**: **{report.recommended_model}**",
        "",
        "---",
        "",
        "## 1. Project Dimensions (Finance-Aware Analysis)",
        "",
        "| Dimension | Score | Assessment |",
        "|---|:---:|:---:|",
        f"| Regulatory Criticality | {dims.regulatory_criticality}/5 | {ProjectDimensions.label(dims.regulatory_criticality)} |",
        f"| Security Risk | {dims.security_risk}/5 | {ProjectDimensions.label(dims.security_risk)} |",
        f"| Requirement Volatility | {dims.requirement_volatility}/5 | {ProjectDimensions.label(dims.requirement_volatility)} |",
        f"| System Failure Impact | {dims.system_failure_impact}/5 | {ProjectDimensions.label(dims.system_failure_impact)} |",
        f"| Testing Rigor Needed | {dims.testing_rigor_needed}/5 | {ProjectDimensions.label(dims.testing_rigor_needed)} |",
        f"| Time-to-Market Urgency | {dims.time_to_market_urgency}/5 | {ProjectDimensions.label(dims.time_to_market_urgency)} |",
        f"| Stakeholder Availability | {dims.stakeholder_availability}/5 | {ProjectDimensions.label(dims.stakeholder_availability)} |",
        "",
        "---",
        "",
        "## 2. Detected Regulatory Frameworks",
        "",
    ]

    for flag in report.finance_compliance_flags:
        lines.append(f"- {flag}")

    lines += [
        "",
        "---",
        "",
        "## 3. Dynamic MCDA Weights",
        "",
        "| Criterion | Weight |",
        "|---|:---:|",
    ]
    for c, w in sorted(weights.items(), key=lambda x: -x[1]):
        lines.append(f"| {c.replace('_', ' ').title()} | **{w*100:.1f}%** |")

    lines += [
        "",
        "---",
        "",
        "## 4. SDLC Model Rankings",
        "",
        "| Rank | Model | Score | Suitability |",
        "|:---:|---|:---:|:---:|",
    ]
    badges = {
        "Highly Recommended": "⭐ Highly Recommended",
        "Recommended": "✅ Recommended",
        "Moderate": "⚠️ Moderate",
        "Unsuitable": "❌ Unsuitable",
    }
    for m in report.ranked_models:
        badge = badges.get(m.suitability, m.suitability)
        lines.append(f"| {m.rank} | **{m.model_name}** | {m.score:.1f} | {badge} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Sensitivity Analysis",
        "",
        f"**Stability**: {'✅ Robust' if report.sensitivity_analysis.get('is_robust') else '⚠️ Sensitive'}",
        f"> {report.sensitivity_analysis.get('summary', '')}",
        "",
        "| Criterion | Variation | Top Model | Score | Stable? |",
        "|---|:---:|---|:---:|:---:|",
    ]
    for v in report.sensitivity_analysis.get("variations", []):
        stab = "✅" if v["stable"] else "⚠️ Shifted"
        lines.append(f"| {v['criterion'].title()} | {v['variation']} | {v['top_model']} | {v['top_score']} | {stab} |")

    lines += [
        "",
        "---",
        "",
        f"## 6. Recommended Model: {report.recommended_model}",
        "",
        f"> {SDLC_CATALOGUE[report.recommended_model]['finance_note']}",
        "",
        "### Strengths",
    ]
    top = next(m for m in report.ranked_models if m.rank == 1)
    for s in top.strengths:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("### Risks")
    for r in top.risks:
        lines.append(f"- {r}")

    lines += [
        "",
        "---",
        "",
        f"## 7. Tailored Workflow: {report.recommended_model}",
        "",
    ]
    for phase in report.tailored_workflow:
        gate = " 🔐 **[Human Approval Gate]**" if phase.get("human_approval_gate") else ""
        lines += [
            f"### Phase {phase['phase_number']}: {phase['phase_name']}{gate}",
            f"**Entry Criteria:** {phase['entry_criteria']}",
            f"**Exit Criteria:** {phase['exit_criteria']}",
            "",
            "**Security Controls:**",
        ]
        for sc in phase["security_controls"]:
            lines.append(f"- 🔒 {sc}")
        lines.append("")
        lines.append("**Compliance Checkpoints:**")
        for cc in phase["compliance_checkpoints"]:
            lines.append(f"- ⚖️ {cc}")
        lines.append("")

    if report.llm_rationale:
        lines += [
            "---",
            "",
            "## 8. AI-Generated Recommendation Rationale",
            "",
            report.llm_rationale,
            "",
        ]

    lines += [
        "---",
        "",
        "## 9. All SDLC Models — Catalogue Reference",
        "",
    ]
    for model_name, data in SDLC_CATALOGUE.items():
        lines += [
            f"### {model_name}",
            f"{data['description']}",
            "",
            f"**Finance Applicability:** {data['finance_note']}",
            "",
            "**Workflow Phases:**",
        ]
        for p in data["phases"]:
            lines.append(f"  - {p}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  LLM Rationale Generator (Optional local LLM call)
# ─────────────────────────────────────────────────────────────────────────────

_LLM_RATIONALE_PROMPT = PromptTemplate(
    input_variables=[
        "project_name", "requirements_summary", "recommended_model",
        "top3_models", "compliance_flags", "dimensions_summary"
    ],
    template="""
You are a senior software engineering consultant specializing in FINANCE domain projects.

## Project: {project_name}

## Requirements Summary (first 1500 chars):
{requirements_summary}

## Project Dimensions:
{dimensions_summary}

## Detected Regulatory Frameworks:
{compliance_flags}

## SDLC MCDA Rankings (Top 3):
{top3_models}

## Recommended Model: {recommended_model}

---

Write a concise (3-5 paragraphs) **professional SDLC selection rationale** that explains:
1. WHY the recommended model is the best fit for this finance project.
2. What specific regulatory, security, and delivery characteristics drove the selection.
3. What the second-ranked model offers and why it was not selected.
4. Any critical risks or implementation recommendations for the recommended model.

Be specific, use the project dimensions and detected frameworks in your reasoning.
Do NOT repeat the rankings table verbatim — write in professional prose.
"""
)


def _generate_llm_rationale(
    llm,
    report: SDLCReport,
    requirements_text: str
) -> str:
    dims = report.dimensions
    dims_summary = (
        f"Regulatory Criticality={dims.regulatory_criticality}/5, "
        f"Security Risk={dims.security_risk}/5, "
        f"Volatility={dims.requirement_volatility}/5, "
        f"Failure Impact={dims.system_failure_impact}/5, "
        f"Testing Rigor={dims.testing_rigor_needed}/5"
    )
    top3 = "\n".join(
        f"{m.rank}. {m.model_name} — Score: {m.score:.1f} ({m.suitability})"
        for m in report.ranked_models[:3]
    )
    prompt = _LLM_RATIONALE_PROMPT.format(
        project_name=report.project_name,
        requirements_summary=requirements_text[:1500],
        recommended_model=report.recommended_model,
        top3_models=top3,
        compliance_flags=", ".join(report.finance_compliance_flags),
        dimensions_summary=dims_summary,
    )
    try:
        return invoke_llm(prompt, max_tokens=3000).strip()
    except Exception as e:
        return f"[LLM rationale unavailable: {e}]"


# ─────────────────────────────────────────────────────────────────────────────
#  Main Agent Class
# ─────────────────────────────────────────────────────────────────────────────

class SDLCAgent:
    """
    Finance-domain SDLC Recommendation Agent.

    Supports two ingestion modes:
      - recommend_from_text()   : plain text (project description / requirements notes)
      - recommend_from_list()   : structured list of requirement strings
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_llm_rationale: bool = True,
    ):
        # Configuration is handled by app.core.llm for the LM Studio endpoint.
        if api_key:
            os.environ["LLM_API_KEY"] = api_key
        if model_name:
            os.environ["LLM_MODEL"] = model_name
        self._llm = None
        self._use_llm = use_llm_rationale

    # ── Public API ────────────────────────────────────────────────────────────

    def recommend_from_text(
        self,
        requirements_text: str,
        project_name: str = "Finance Project",
    ) -> SDLCReport:
        """
        Analyse plain text (project description, meeting notes, stakeholder inputs,
        or extracted requirements) and produce a full SDLC recommendation.
        """
        return self._run_pipeline(requirements_text, project_name)

    def recommend_from_list(
        self,
        requirements: List[str],
        project_name: str = "Finance Project",
    ) -> SDLCReport:
        """
        Accept a structured list of requirement strings, join them into a single
        analysis corpus, and produce a full SDLC recommendation.
        """
        combined = "\n".join(requirements)
        return self._run_pipeline(combined, project_name)

    # ── Internal Pipeline ─────────────────────────────────────────────────────

    def _run_pipeline(self, text: str, project_name: str) -> SDLCReport:
        # 1. Extract project dimensions
        dims = _extract_dimensions(text)

        # 2. Detect compliance frameworks
        compliance_flags = _extract_compliance_flags(text)

        # 3. Compute dynamic MCDA weights
        weights = _compute_dynamic_weights(dims)

        # 4. MCDA scoring & ranking
        raw_scores = _mcda_scores(weights)
        ranked = _build_ranked_models(raw_scores)

        # 5. Best model
        recommended = ranked[0].model_name

        # 6. Sensitivity analysis
        sensitivity = _sensitivity_analysis(weights)

        # 7. Tailored workflow
        workflow = _build_workflow(recommended, compliance_flags, dims)

        # 8. Build rationale summary
        top3_names = ", ".join(m.model_name for m in ranked[:3])
        rationale = (
            f"{recommended} was selected with a score of {ranked[0].score:.1f}. "
            f"Runner-up models: {top3_names}. "
            f"Key drivers: regulatory criticality={dims.regulatory_criticality}/5, "
            f"security risk={dims.security_risk}/5, "
            f"testing rigor={dims.testing_rigor_needed}/5."
        )

        # 9. Input summary
        input_summary = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")

        report = SDLCReport(
            project_name=project_name,
            input_summary=input_summary,
            dimensions=dims,
            dynamic_weights=weights,
            ranked_models=ranked,
            recommended_model=recommended,
            recommendation_rationale=rationale,
            tailored_workflow=workflow,
            sensitivity_analysis=sensitivity,
            finance_compliance_flags=compliance_flags,
        )

        # 10. Optional LLM-generated narrative
        if self._use_llm:
            report.llm_rationale = _generate_llm_rationale(self._llm, report, text)

        return report

    def print_catalogue(self) -> str:
        """Return a formatted catalogue of all supported SDLC models."""
        lines = [
            "# SDLcraft — Supported SDLC Models Catalogue",
            f"**Total models**: {len(SDLC_CATALOGUE)}",
            "",
        ]
        for name, data in SDLC_CATALOGUE.items():
            lines += [
                f"## {name}",
                data["description"],
                "",
                f"**Finance Applicability:** {data['finance_note']}",
                "",
                "**Workflow Phases:**",
            ]
            for p in data["phases"]:
                lines.append(f"  - {p}")
            lines.append("")
            lines.append("**Criteria Scores (raw):**")
            for c, v in data["criteria"].items():
                lines.append(f"  - {c.replace('_', ' ').title()}: {v}")
            lines.append("")
        return "\n".join(lines)
