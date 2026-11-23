# SDLcraft 🏗️

**Finance-domain Software Requirements Engineering + SDLC Recommendation System**

SDLcraft is an AI-powered pipeline that:
1. **Extracts software requirements** from either a PDF document or a free-text project description using a RAG (Retrieval-Augmented Generation) engine.
2. **Recommends the optimal SDLC model** (from a catalogue of 9 finance-aware models) using Multi-Criteria Decision Analysis (MCDA) with dynamic, project-dimension-driven weights.

> Built specifically for **finance / fintech projects** — compliance, security, and regulatory rigor are first-class citizens.

---

## Repository Structure

```
SDLcraft/
├── backend/                        # Python FastAPI backend
│   ├── app/
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   └── sdlc_agent.py       # SDLC Recommendation Agent (9 models, MCDA)
│   │   ├── rag/
│   │   │   ├── __init__.py
│   │   │   └── rag.py              # RAG engine (PDF + free-text ingestion)
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── app.py              # FastAPI app factory + CORS
│   │   │   └── routes.py           # All API route handlers
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── models.py           # Pydantic request / response schemas
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   └── config.py           # Centralised config (env vars)
│   │   └── __init__.py
│   ├── data/
│   │   └── chroma_store/           # ChromaDB vector store (auto-created)
│   ├── uploads/                    # Uploaded PDFs (auto-created)
│   ├── output/                     # Generated requirement & SDLC reports
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_sdlc_agent.py      # Pytest unit tests
│   ├── main.py                     # CLI pipeline runner (text & PDF modes)
│   ├── server.py                   # Uvicorn entry-point
│   ├── requirements.txt
│   └── .env.example
│
├── frontend/                       # React + Vite frontend (UI)
│   ├── src/
│   │   ├── components/             # Reusable UI components
│   │   ├── pages/                  # Page-level views
│   │   ├── services/
│   │   │   └── api.js              # Backend API service layer
│   │   └── assets/                 # Static assets (icons, images)
│   ├── public/
│   ├── package.json
│   └── .env.example
│
├── .gitignore
└── README.md                       # ← You are here
```

---

## Features

| Feature | Details |
|---|---|
| **Dual-mode requirements pipeline** | Free-text briefs go directly to the LLM; PDFs use OCR, hybrid retrieval, and the LLM |
| **Finance-aware prompting** | Auto-applies RBI, DPDP, PCI-DSS, AML/KYC, GDPR context |
| **7 SDLC models** | Waterfall, V-Model, Incremental, Prototype, Spiral, RAD, Agile |
| **Dynamic MCDA weights** | Weights shift based on project dimensions (compliance, security, speed, …) |
| **Sensitivity analysis** | ±20% weight perturbation stability check |
| **Tailored workflow** | Phase-by-phase plan with security controls and compliance gates |
| **LLM rationale** | Local LM Studio-powered prose explanation for the recommendation |
| **REST API** | FastAPI with Swagger UI at `/api/docs` |
| **IEEE-830 requirements** | "The system shall …" format with SUPPORTED / INFERRED / ASSUMPTION tags |

---

## Quick Start

### Backend

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Configure environment
cp backend/.env.example backend/.env
# Ensure LM Studio is running with a model loaded at http://127.0.0.1:1234

# 4a. Run the CLI pipeline (text mode demo)
cd backend
python main.py

# 4b. Start the API server
uvicorn server:app --reload --port 8000
# Swagger UI → http://localhost:8000/api/docs
```

### Frontend

```bash
cd frontend
cp .env.example .env       # set VITE_API_URL if backend is not on :8000
npm install
npm run dev                # → http://localhost:5173
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/ingest/text` | Index free-text description |
| `POST` | `/api/ingest/pdf` | Upload and index a PDF |
| `POST` | `/api/requirements/query` | Query requirements from indexed doc |
| `POST` | `/api/sdlc/recommend` | SDLC recommendation from requirement list |
| `GET` | `/api/sdlc/catalogue` | Full SDLC model catalogue |
| `POST` | `/api/pipeline/text` | Full pipeline — text mode |
| `POST` | `/api/pipeline/pdf` | Full pipeline — PDF mode |

Interactive docs: **`http://localhost:8000/api/docs`**

---

## Python Usage (without API)

```python
from backend.app.rag.rag import answer_query_from_text
from backend.app.agents.sdlc_agent import SDLCAgent

# Extract requirements from a text description
requirements = answer_query_from_text(
    text_description="We are building a digital KYC platform compliant with RBI VCIP guidelines...",
    query="List all functional, security, and compliance requirements.",
)

# Get SDLC recommendation
agent = SDLCAgent()
report = agent.recommend_from_text(requirements, project_name="KYC Platform")

print(report.recommended_model)   # e.g. "Agile"
print(report.to_markdown())       # Full report
```

---

## Supported SDLC Models

| Model | Best For |
|---|---|
| **Waterfall** | Simple systems with frozen requirements |
| **V-Model** | Mission-critical systems requiring strict verification |
| **Incremental** | Phased delivery of a larger system |
| **Prototype** | Validating unclear requirements and user workflows |
| **Spiral** | Large, complex, high-risk projects |
| **RAD** | Time-critical applications with available stakeholders |
| **Agile** | Evolving requirements and frequent stakeholder feedback |

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | LM Studio OpenAI-compatible base URL |
| `LLM_API_KEY` | `lm-studio` | LM Studio API key placeholder |
| `LLM_MODEL` | `local-model` | Model identifier exposed by LM Studio |
| `LLM_TEMPERATURE` | `0.2` | LLM temperature |
| `CHROMA_PERSIST_DIR` | `./data/chroma_store` | ChromaDB storage path |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `API_PORT` | `8000` | Backend server port |

---

## Tech Stack

**Backend:** Python 3.11+, FastAPI, LangChain, ChromaDB, Sentence-Transformers, PyMuPDF, LM Studio  
**Frontend:** React 18, Vite, Tailwind CSS  
**AI:** LM Studio OpenAI-compatible API (LLM inference), `all-MiniLM-L6-v2` (embeddings)
