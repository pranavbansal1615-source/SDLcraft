"""
rag_requirements.py - Hybrid RAG requirements generator (SDLcraft, finance).

Changes vs. the original script (inspired by pico-bot):
  1. Hybrid retrieval: BM25 keywords + ChromaDB vectors, merged by Reciprocal Rank Fusion.
  2. Two knowledge sources: the user's own document AND an optional reference corpus
     (RBI / SEBI / NIST / OWASP ...), searched separately and labelled differently.
  3. Evidence labels E1..En in the prompt; the model may cite only those labels, and code
     validates and maps them back to real chunk ids (anti-hallucinated-citation check).
  4. Chunks sized for the embedding model (~1000 chars, MiniLM reads ~256 tokens).
  5. Isolation: every query filters on document_id AND user_id; re-indexing deletes old chunks.
  6. Idempotent, deterministic chunk ids; cosine-space collection.
  7. OCR renders at 300 DPI; paths are cross-platform; models load lazily (no import side effects).
  8. BM25 index is cached per document and rebuilt only when that document changes.

Install:  pip install chromadb rank_bm25 sentence-transformers pymupdf pytesseract pillow \
                      langchain-core langchain-text-splitters openai python-dotenv torch
NOTE: the collection is created with cosine distance. If you already have a collection with
the same name created by the old script (L2 distance), change CHROMA_COLLECTION or delete
the old chroma_store folder.
"""
from __future__ import annotations

import io
import os
import re
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from dotenv import load_dotenv

load_dotenv()

import chromadb
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from rank_bm25 import BM25Okapi

# ─────────────────────────────────────────────────────────────
#  Configuration (all overridable by environment variables)
# ─────────────────────────────────────────────────────────────

TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if not TESSERACT_CMD:
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        shutil.which("tesseract"),
    ]
    TESSERACT_CMD = next((path for path in candidates if path and os.path.isfile(path)), None)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

DEFAULT_CHROMA_DIR = str(Path(__file__).resolve().parent / "data" / "chroma_store")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", DEFAULT_CHROMA_DIR)
COLLECTION = os.getenv("CHROMA_COLLECTION", "sdlcraft_hybrid")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # was 2000 (too long for MiniLM)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))   # was 200
CANDIDATES = int(os.getenv("RAG_CANDIDATES", "30"))      # per retriever before fusion
TOP_K_PROJECT = int(os.getenv("RAG_TOP_K_PROJECT", "6"))  # was 7
TOP_K_REFERENCE = int(os.getenv("RAG_TOP_K_REFERENCE", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
RRF_K = 60  # standard reciprocal-rank-fusion constant
OCR_TEXT_THRESHOLD = 50

PROJECT_KIND = "project"
REFERENCE_KIND = "reference"
CORPUS_USER = "__corpus__"


# ─────────────────────────────────────────────────────────────
#  PDF processing and text cleaning (same behaviour as before)
# ─────────────────────────────────────────────────────────────

def process_pdf_hybrid(pdf_path: str, text_threshold: int = OCR_TEXT_THRESHOLD) -> List[Document]:
    """Extract text per page; fall back to OCR (300 DPI) for image-only pages."""
    docs: List[Document] = []
    name = Path(pdf_path).name
    with fitz.open(pdf_path) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text()
            if len(text.strip()) < text_threshold:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                text = pytesseract.image_to_string(img)
            docs.append(Document(
                page_content=text.strip(),
                metadata={
                    "source_file": name,
                    "page_number": i + 1,
                    "file_type": "pdf",
                    "input_mode": "document",
                },
            ))
    return docs


def clean_page_text(text: str) -> str:
    """Remove page-number artifacts and normalize whitespace."""
    cleaned = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if re.fullmatch(r"(page|p\.)\s*\d+(\s*(/|of)\s*\d+)?", s, flags=re.IGNORECASE):
            continue
        if len(s) <= 4 and not re.search(r"[A-Za-z]", s):
            continue
        cleaned.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def split_docs(documents: List[Document], chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    # Separators suited to requirements/regulation text (not source code).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n- ", "\n* ", ". ", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────────────────────
#  Tokenizer for BM25
# ─────────────────────────────────────────────────────────────

_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "will with shall must should system".split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP and len(t) > 1]


# ─────────────────────────────────────────────────────────────
#  Embeddings + Vector store (lazy singletons)
# ─────────────────────────────────────────────────────────────

class EmbeddingManager:
    def __init__(self, model_name: str = EMBED_MODEL):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True)


@dataclass
class Hit:
    chunk_id: str
    text: str
    meta: Dict
    score: float
    found_by: List[str] = field(default_factory=list)  # ["bm25", "vector"]


class HybridStore:
    """ChromaDB (vectors) + in-memory BM25 (keywords), fused with RRF."""

    def __init__(self):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.col = self.client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine", "description": "SDLcraft hybrid RAG"},
        )
        self._embedder: Optional[EmbeddingManager] = None
        self._bm25_cache: Dict[str, Tuple[List[str], List[str], List[Dict], Optional[BM25Okapi]]] = {}

    @property
    def embedder(self) -> EmbeddingManager:
        if self._embedder is None:
            self._embedder = EmbeddingManager()
        return self._embedder

    # ---- filters ---------------------------------------------------------
    @staticmethod
    def project_where(document_id: str, user_id: str) -> Dict:
        return {"$and": [{"kind": PROJECT_KIND}, {"document_id": document_id}, {"user_id": user_id}]}

    @staticmethod
    def reference_where() -> Dict:
        return {"kind": REFERENCE_KIND}

    # ---- write -----------------------------------------------------------
    def exists(self, document_id: str, user_id: str) -> bool:
        res = self.col.get(where=self.project_where(document_id, user_id), limit=1)
        return len(res["ids"]) > 0

    def delete(self, document_id: str, user_id: str) -> None:
        self.col.delete(where=self.project_where(document_id, user_id))
        self._bm25_cache.pop(self._project_key(document_id, user_id), None)

    def add_chunks(self, chunks: List[Document], id_prefix: str) -> int:
        if not chunks:
            return 0
        texts = [c.page_content for c in chunks]
        vectors = self.embedder.embed(texts)
        ids, metas = [], []
        for i, c in enumerate(chunks):
            cid = f"{id_prefix}#chunk-{i + 1}"
            ids.append(cid)
            m = dict(c.metadata)
            m["chunk_id"] = cid
            m["chunk_index"] = i + 1
            m["content_length"] = len(c.page_content)
            metas.append(m)
        self.col.upsert(ids=ids, documents=texts, embeddings=[v.tolist() for v in vectors], metadatas=metas)
        return len(ids)

    # ---- BM25 index (cached; rebuilt when a document is (re)indexed) ------
    @staticmethod
    def _project_key(document_id: str, user_id: str) -> str:
        return f"project::{user_id}::{document_id}"

    def invalidate(self, key: str) -> None:
        self._bm25_cache.pop(key, None)

    def _bm25(self, key: str, where: Dict):
        if key not in self._bm25_cache:
            res = self.col.get(where=where, include=["documents", "metadatas"])
            ids, docs, metas = res["ids"], res["documents"], res["metadatas"]
            tokens = [tokenize(d) for d in docs]
            bm25 = BM25Okapi(tokens) if any(tokens) else None
            self._bm25_cache[key] = (ids, docs, metas, bm25)
        return self._bm25_cache[key]

    # ---- read ------------------------------------------------------------
    def search(self, query: str, where: Dict, cache_key: str, top_k: int) -> List[Hit]:
        ids, docs, metas, bm25 = self._bm25(cache_key, where)
        if not ids:
            return []
        by_id = {cid: (d, m) for cid, d, m in zip(ids, docs, metas)}
        rankings: List[Tuple[str, List[str]]] = []

        # 1) keyword ranking
        if bm25 is not None:
            q_tokens = tokenize(query)
            if q_tokens:
                scores = bm25.get_scores(q_tokens)
                order = np.argsort(scores)[::-1][:CANDIDATES]
                rankings.append(("bm25", [ids[i] for i in order if scores[i] > 0]))

        # 2) meaning ranking
        try:
            q_vec = self.embedder.embed([query])[0].tolist()
            res = self.col.query(query_embeddings=[q_vec], n_results=min(CANDIDATES, len(ids)), where=where)
            rankings.append(("vector", res["ids"][0]))
        except Exception as e:  # keyword search alone still works
            print(f"[rag] vector search unavailable, using BM25 only: {e}")

        # 3) Reciprocal Rank Fusion
        fused: Dict[str, float] = defaultdict(float)
        found_by: Dict[str, List[str]] = defaultdict(list)
        for name, ranking in rankings:
            for rank, cid in enumerate(ranking):
                fused[cid] += 1.0 / (RRF_K + rank + 1)
                found_by[cid].append(name)

        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [Hit(cid, by_id[cid][0], by_id[cid][1], round(score, 5), found_by[cid]) for cid, score in top if cid in by_id]


_store: Optional[HybridStore] = None
def get_store() -> HybridStore:
    global _store
    if _store is None:
        _store = HybridStore()
    return _store


def get_llm():
    from app.core.llm import get_llm_client

    return get_llm_client()


# ─────────────────────────────────────────────────────────────
#  Ingestion
# ─────────────────────────────────────────────────────────────

def _index_project_docs(docs: List[Document], user_id: str, document_id: str, source_label: str) -> int:
    store = get_store()
    chunks = split_docs(docs)
    chunks = [c for c in chunks if c.page_content.strip()]
    for c in chunks:
        c.metadata.update({
            "kind": PROJECT_KIND,
            "document_id": document_id,
            "user_id": user_id,
            "source_label": source_label,
        })
    n = store.add_chunks(chunks, id_prefix=f"{user_id}/{document_id}")
    store.invalidate(store._project_key(document_id, user_id))
    return n


def generate_pdf_emb(file_path: str, user_id: str, document_id: str, overwrite: bool = False) -> str:
    """Ingest a PDF. Idempotent unless overwrite=True (which deletes the old chunks first)."""
    store = get_store()
    if store.exists(document_id, user_id):
        if not overwrite:
            return f"Document '{document_id}' already indexed."
        store.delete(document_id, user_id)

    docs = process_pdf_hybrid(file_path)
    for d in docs:
        d.page_content = clean_page_text(d.page_content)
    n = _index_project_docs(docs, user_id, document_id, Path(file_path).name)
    return f"Indexed {n} chunks from '{Path(file_path).name}'."


def generate_text_emb(text_input: str, user_id: str, document_id: str,
                      source_label: str = "user_description", overwrite: bool = False) -> str:
    """Ingest free-form text (project description, notes, transcript)."""
    store = get_store()
    if store.exists(document_id, user_id):
        if not overwrite:
            return f"Text document '{document_id}' already indexed."
        store.delete(document_id, user_id)

    doc = Document(
        page_content=text_input.strip(),
        metadata={"source_file": source_label, "page_number": 1,
                  "file_type": "text", "input_mode": "text_description"},
    )
    n = _index_project_docs([doc], user_id, document_id, source_label)
    return f"Indexed {n} chunks from text description '{document_id}'."


def ingest_reference_corpus(corpus_dir: str, overwrite: bool = False) -> str:
    """
    Index trusted reference documents (RBI / SEBI / NIST / OWASP / ISO 20022 ... as .txt or .md)
    once. They are searched on every query, separately from the user's own document.
    """
    store = get_store()
    root = Path(corpus_dir)
    files = sorted([*root.rglob("*.txt"), *root.rglob("*.md")])
    if not files:
        return f"No .txt/.md files found in {corpus_dir}"

    total = 0
    for f in files:
        rel = f.relative_to(root).as_posix()
        existing = store.col.get(where={"$and": [{"kind": REFERENCE_KIND}, {"source_file": rel}]}, limit=1)
        if existing["ids"] and not overwrite:
            continue
        if existing["ids"]:
            store.col.delete(where={"$and": [{"kind": REFERENCE_KIND}, {"source_file": rel}]})
        text = f.read_text(encoding="utf-8", errors="ignore")
        doc = Document(page_content=text, metadata={
            "source_file": rel, "page_number": 0, "file_type": f.suffix.lstrip("."),
            "input_mode": "reference", "kind": REFERENCE_KIND,
            "document_id": f"corpus:{rel}", "user_id": CORPUS_USER,
        })
        chunks = [c for c in split_docs([doc]) if c.page_content.strip()]
        total += store.add_chunks(chunks, id_prefix=f"corpus/{rel}")
    store.invalidate("reference")
    return f"Reference corpus: {total} new chunks indexed from {len(files)} files."


# ─────────────────────────────────────────────────────────────
#  Retrieval + evidence labelling
# ─────────────────────────────────────────────────────────────

def retrieve(query: str, document_id: str, user_id: str) -> Tuple[List[Hit], List[Hit]]:
    store = get_store()
    project = store.search(
        query, store.project_where(document_id, user_id),
        store._project_key(document_id, user_id), TOP_K_PROJECT,
    )
    reference = store.search(query, store.reference_where(), "reference", TOP_K_REFERENCE)
    return project, reference


def build_context(project: List[Hit], reference: List[Hit],
                  max_chars: int = MAX_CONTEXT_CHARS) -> Tuple[str, Dict[str, Hit]]:
    """Label every chunk E1..En so the model can cite them and code can verify the citations."""
    labelled: Dict[str, Hit] = {}
    parts: List[str] = []
    used = 0
    n = 0

    def add(section: str, hits: List[Hit]) -> None:
        nonlocal used, n
        if not hits:
            return
        parts.append(f"=== {section} ===")
        for h in hits:
            n += 1
            label = f"E{n}"
            page = h.meta.get("page_number", "N/A")
            block = f"[{label} | {h.meta.get('source_file', '?')} | Page {page}]\n{h.text}"
            remaining = max_chars - used
            if remaining <= 0:
                return
            block = block[:remaining]
            parts.append(block)
            used += len(block)
            labelled[label] = h

    add("PROJECT CONTEXT (the user's own requirements / description)", project)
    add("REFERENCE CORPUS (regulations and standards - use for compliance and security grounding)", reference)
    return "\n\n".join(parts), labelled


# ─────────────────────────────────────────────────────────────
#  Prompt (kept from the original; only the evidence/citation rules changed)
# ─────────────────────────────────────────────────────────────

# ── PDF RAG Prompt: used when context comes from vector-searched PDF chunks ──

PDF_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "user_intent", "inbuilt_instruction"],
    template=(
        """
You are a Senior Software Requirements Engineer specializing in FINANCE and FINTECH domain projects.
You have been given retrieved passages from a PDF document (SRS, BRD, proposal, or RFP) and a
user-stated intent describing what they want to build.

---

## USER INTENT (What the user wants to build / extract)

```text
{user_intent}
```

---

## INBUILT EXTRACTION INSTRUCTION

{inbuilt_instruction}

---

## RETRIEVED DOCUMENT CONTEXT

The following passages were retrieved from the uploaded PDF using semantic vector search.
Each passage is labelled [E1], [E2], … — cite ONLY these labels; never invent new ones.

```text
{context}
```

---

## FINANCE DOMAIN CONSTRAINTS (apply SELECTIVELY — read this carefully)

Apply the constraints below **only** when the retrieved context or user intent explicitly mentions them.

| Constraint | Apply only when context or intent mentions… |
|---|---|
| RBI / SEBI / DPDP / PCI-DSS / AML / KYC / SOX | …a regulated product: lending, payments, banking, trading, insurance |
| AES-256 / TLS / MFA / RBAC | …user authentication, PII storage, or financial transactions |
| 99.9%+ uptime SLA | …production systems, transaction-critical paths, or SLA requirements |
| 7-year data retention | …financial transaction records or audit mandates |
| Fraud / anomaly detection | …payment flows, loan disbursals, or account operations |

**If the uploaded document is an internal tool spec, dashboard design, prototype, or MVP brief
with no mention of regulations, PII, or financial transactions — do NOT inject compliance boilerplate.**
Any requirement sourced from general finance knowledge (not from the retrieved context) must be
marked `ASSUMPTION`, not `INFERRED`.

NEVER fabricate regulatory clause numbers or thresholds not present in the context.

---

## YOUR TASK

1. Use the user intent to focus requirement extraction on the right functionality.
2. Use the retrieved context (evidence) as the primary source of truth.
3. Generate specific, testable, IEEE-830 compliant requirements.
4. Each requirement must be categorised and cited with evidence labels.
5. Distinguish SUPPORTED (from context), INFERRED (logical extension), and ASSUMPTION (not in context).

---

## REQUIREMENT QUALITY RULES

Each requirement MUST be:
- Written as **"The system shall ..."** or **"The system must ..."**
- Atomic, unambiguous, testable
- Tagged: **Functional / Security / Compliance / Performance / Usability**
- Non-duplicative — merge if two requirements are equivalent

---

## OUTPUT FORMAT

### Generated Requirements

#### REQ-01

**Category:** [Functional / Security / Compliance / Performance / Usability]

**Requirement:**
The system shall ...

**Reasoning:**
[Why this requirement is needed given the user intent and context]

**Supporting Evidence:**
[E1, E3 — labels from the context above ONLY]

**Evidence Status:**
[SUPPORTED / INFERRED / ASSUMPTION]

---

[Repeat for REQ-02, REQ-03 … cover all significant requirements]

---

### Missing Information

[Gaps in the PDF context that the user should clarify]
If none: > None identified.

---

### Potential Issues

[Contradictions, ambiguities, or compliance gaps spotted in the PDF]
If none: > None identified.

---

## FINAL RULE

Ground every requirement in the retrieved context.
Prioritise accuracy and traceability. Quality over quantity.
"""
    ),
)


# ── Text Direct Prompt: text goes straight to LLM — no retrieval, no vector DB ──

TEXT_DIRECT_PROMPT = PromptTemplate(
    input_variables=["project_description", "inbuilt_instruction", "user_instruction"],
    template=(
        """
You are a Senior Software Requirements Engineering Consultant specialising in FINANCE and FINTECH domain projects.
The user has provided a plain-text description of their project. This description is small enough to send directly
to you — no document retrieval is needed. Analyse it in full and produce a comprehensive requirements specification.

---

## PROJECT DESCRIPTION (provided directly by the user)

```text
{project_description}
```

---

## EXTRACTION INSTRUCTIONS

**Inbuilt instruction (applied automatically):**
{inbuilt_instruction}

**User's additional instruction:**
{user_instruction}

---

## FINANCE DOMAIN CONSTRAINTS (apply SELECTIVELY — read this carefully)

This project may be in the finance / fintech domain. Apply the constraints below **only** when
the project description explicitly mentions or clearly implies them.

| Constraint | Apply only when the description mentions… |
|---|---|
| RBI / SEBI / DPDP / PCI-DSS / AML / KYC / SOX | …a regulated product: lending, payments, banking, trading, insurance |
| AES-256 / TLS / MFA / RBAC | …user authentication, PII storage, or financial transactions |
| 99.9%+ uptime SLA | …production systems, transaction-critical paths, or SLA requirements |
| 7-year data retention | …financial transaction records or audit mandates |
| Fraud / anomaly detection | …payment flows, loan disbursals, or account operations |

**If the project is an internal tool, dashboard, prototype, MVP, or admin utility with no mention
of regulations, PII, or financial transactions — do NOT inject compliance/security boilerplate.**
Mark any requirement you add from general finance knowledge as `ASSUMPTION`, not `INFERRED`.

Never fabricate regulatory clause numbers or thresholds not present in the description.

---

## YOUR TASK

1. Read and fully understand the project description.
2. Identify the core functionalities, users, integrations, and constraints described.
3. Generate a complete, structured set of software requirements covering:
   - Functional requirements (what the system does)
   - Security requirements (how the system protects data and users)
   - Compliance requirements (regulatory obligations)
   - Performance requirements (measurable SLAs)
   - Usability requirements (UX and accessibility)
4. For each requirement, state your reasoning and whether it was explicitly stated, inferred, or assumed.
5. Distinguish evidence status precisely:
   - **SUPPORTED** — explicitly stated in the project description
   - **INFERRED** — a direct, necessary logical consequence of something explicit (e.g. "supports payments" implies "needs idempotency")
   - **ASSUMPTION** — added from general finance knowledge NOT mentioned in the description; mark these clearly so they don't bias SDLC model selection

---

## REQUIREMENT QUALITY RULES

Each requirement MUST be:
- Written as **"The system shall ..."** or **"The system must ..."**
- Specific, atomic, and testable — no vague statements like "the system should be fast"
- Tagged: **Functional / Security / Compliance / Performance / Usability**
- Non-duplicative

---

## OUTPUT FORMAT

### Project Overview

[1-2 sentences summarising what the system does and who it serves]

---

### Generated Requirements

#### REQ-01

**Category:** [Functional / Security / Compliance / Performance / Usability]

**Requirement:**
The system shall ...

**Reasoning:**
[Why this requirement is needed — reference specific parts of the project description]

**Source:** [EXPLICIT — stated verbatim in description | INFERRED — necessary logical consequence of something explicit | ASSUMPTION — added from general finance/domain knowledge not mentioned in description]

---

[Repeat for REQ-02, REQ-03 … until all significant requirements are covered]

---

### Missing Information

[List gaps in the description that would affect requirements]
If nothing important is missing: > None identified.

---

### Potential Issues

[Contradictions, ambiguities, or design concerns in the description]
If none: > None identified.

---

## FINAL RULE

Work only from the project description provided. Be thorough — but be **accurate**.
Only include security, compliance, and regulatory requirements if the project description
actually warrants them. A simple internal tool, dashboard, or prototype does NOT need
AES-256, MFA, RBI, or audit-log requirements unless those are explicitly mentioned.
Mark everything not in the description as ASSUMPTION so downstream analysis is not biased.
Quality and accurate traceability matter more than quantity.
"""
    ),
)


# Keep old REQUIREMENT_PROMPT for backward compatibility (used by answer_query)
REQUIREMENT_PROMPT = PromptTemplate(
    input_variables=["context", "question", "input_mode"],
    template=(
        """
You are an expert Software Requirements Engineering Assistant specializing in FINANCE domain projects.
Your task is to generate specific, testable, IEEE-830 compliant software requirements.

---

## INPUT SOURCE TYPE

The context below was derived from: **{input_mode}**

---

## CONTEXT

The following is the retrieved context. Every passage has an evidence label such as [E1], [E2].
PROJECT CONTEXT is the primary source for what the system does. REFERENCE CORPUS passages are
regulations and standards you may cite for compliance and security requirements.

```text
{context}
```

Treat PROJECT CONTEXT as the primary source of information.

---

## QUESTION / REQUESTED FUNCTIONALITY

```text
{question}
```

Use the question to determine the functionality for which requirements must be generated.

---

## FINANCE DOMAIN CONSTRAINTS

Since this is a **finance / fintech project**, automatically consider the following compliance and security dimensions when applicable:
- Regulatory: RBI guidelines, SEBI, DPDP Act 2023, PCI-DSS, AML/KYC, FCRA, ECOA, GLBA, SOX
- Security: Encryption at rest/transit, MFA, RBAC, audit trails, session management, data masking
- Data retention and audit log requirements
- Fraud detection and anomaly flagging

Only add these if they are **relevant** to the requested functionality or supported by the context.
Never invent legal clauses, thresholds or certifications that do not appear in the context.

---

## TASK

Based on the context and question:

1. Understand the existing system and its capabilities from the context.
2. Identify existing requirements relevant to the requested functionality.
3. Identify dependencies and constraints established by those requirements.
4. Generate new, specific requirements needed to support the requested functionality.
5. Provide reasoning for every generated requirement.
6. Provide traceability to relevant requirements in the context.
7. Clearly distinguish directly supported information from logical inferences or assumptions.

Do not merely summarize or repeat the existing requirements.

---

## REQUIREMENT QUALITY

Every generated requirement should be:

- Written as: **"The system shall ..."** or **"The system must ..."**
- Specific and unambiguous
- Atomic where possible
- Testable
- Consistent with existing requirements
- Relevant to the requested functionality
- Non-duplicative
- Tagged with a category: Functional / Security / Compliance / Performance / Usability

Avoid vague requirements like "the system should be fast" without measurable criteria.

---

## EVIDENCE STATUS

For each generated requirement, assign one of:

**SUPPORTED** – Directly evidenced by the context.
**INFERRED** – Reasonable logical extension of context + requested functionality.
**ASSUMPTION** – Requires information not in the context.

Prefer SUPPORTED and INFERRED. Flag ASSUMPTION clearly.

---

## TRACEABILITY AND CITATIONS

For every generated requirement, cite the supporting passage(s) using ONLY the evidence labels
that appear in the context (for example "E2, E5"). Never cite a label that is not in the context
and never write file names or clause numbers from memory.

If no passage supports the requirement, write exactly:
> No direct supporting requirement found in the provided context.
and set the Evidence Status to ASSUMPTION.

---

## OUTPUT FORMAT

Return the answer using the following structure:

### Generated Requirements

#### REQ-01

**Category:** [Functional / Security / Compliance / Performance / Usability]

**Requirement:**
[IEEE-830 style: "The system shall ..."]

**Reasoning:**
[Why this requirement is needed based on the question and context]

**Supporting Evidence:**
[Evidence labels, e.g. E1, E4]

**Evidence Status:**
[SUPPORTED / INFERRED / ASSUMPTION]

---

#### REQ-02

[Continue for each requirement ...]

---

### Missing Information

[List information that is required but unavailable in the context.]

If nothing important is missing, state:
> None identified.

---

### Potential Issues

[List contradictions, ambiguities, duplicates, or other issues found in the context.]

If no issues are found, state:
> None identified.

---

## FINAL RULE

Always ground the response in the provided context.
Prioritize accuracy, consistency, reasoning, and traceability over generating a large number of requirements.
"""
    ),
)


# ─────────────────────────────────────────────────────────────
#  Citation validation + final answer
# ─────────────────────────────────────────────────────────────

_LABEL_RE = re.compile(r"\bE(\d{1,3})\b")


def _validate_and_index(answer: str, labelled: Dict[str, Hit]) -> str:
    """Flag invented labels and append an Evidence Index mapping labels to real chunk ids."""
    cited = {f"E{m}" for m in _LABEL_RE.findall(answer)}
    invalid = sorted(cited - set(labelled), key=lambda s: int(s[1:]))
    used = sorted(cited & set(labelled), key=lambda s: int(s[1:]))

    out = [answer.strip()]
    if invalid:
        out.append(
            "\n> ⚠️ **Citation check:** the model cited labels that were not retrieved "
            f"({', '.join(invalid)}). Treat those citations as NEEDS-VALIDATION."
        )
    if used:
        out.append("\n### Evidence Index\n")
        out.append("| Label | Source | Page | Chunk id | Found by |")
        out.append("|---|---|:---:|---|---|")
        for label in used:
            h = labelled[label]
            out.append(
                f"| {label} | {h.meta.get('source_file', '?')} | {h.meta.get('page_number', '-')} "
                f"| `{h.chunk_id}` | {'+'.join(h.found_by)} |"
            )
    return "\n".join(out)


def answer_query_pdf_rag(
    user_intent: str,
    document_id: str,
    user_id: str = "default_user",
    inbuilt_instruction: str = (
        "Extract specific, testable, IEEE-830 compliant finance software requirements "
        "covering functional, security, compliance, performance, and usability."
    ),
    return_hits: bool = False,
):
    """
    Vector search + RAG generation for uploaded PDF documents.
    1. Formulates search query combining user_intent and inbuilt_instruction.
    2. Performs hybrid retrieval on indexed PDF chunks (and reference corpus).
    3. Builds evidence context with [E1], [E2] labels.
    4. Invokes PDF_RAG_PROMPT with context, user_intent, and inbuilt_instruction.
    5. Validates evidence citations.
    """
    search_query = f"{user_intent}\n{inbuilt_instruction}"
    project, reference = retrieve(search_query, document_id, user_id)
    if not project and not reference:
        msg = f"No indexed content found for document_id='{document_id}' and user_id='{user_id}'."
        return (msg, []) if return_hits else msg

    context, labelled = build_context(project, reference)
    prompt = PDF_RAG_PROMPT.format(
        context=context,
        user_intent=user_intent,
        inbuilt_instruction=inbuilt_instruction,
    )
    from app.core.llm import invoke_llm

    response = invoke_llm(prompt)
    final = _validate_and_index(response.strip(), labelled)
    return (final, list(labelled.values())) if return_hits else final


def answer_query(query: str, document_id: str, user_id: str = "default_user",
                 return_hits: bool = False):
    """Answer a requirements query against an already-indexed document (PDF or text)."""
    return answer_query_pdf_rag(
        user_intent=query,
        document_id=document_id,
        user_id=user_id,
        return_hits=return_hits,
    )


def answer_query_from_text(text_description: str, query: str, user_id: str = "default_user",
                           document_id: Optional[str] = None) -> str:
    document_id = document_id or f"text_{uuid.uuid4().hex[:8]}"
    generate_text_emb(text_description, user_id, document_id)
    return answer_query(query, document_id, user_id)


def answer_query_from_pdf(pdf_path: str, query: str, user_id: str = "default_user",
                          document_id: Optional[str] = None) -> str:
    document_id = document_id or Path(pdf_path).stem
    generate_pdf_emb(pdf_path, user_id, document_id)
    return answer_query_pdf_rag(query, document_id, user_id)


def generate_requirements_direct(
    project_description: str,
    user_instruction: str = "",
    inbuilt_instruction: str = (
        "Extract a comprehensive, testable IEEE-830 requirement specification "
        "covering functional, security, compliance, performance, and usability requirements."
    ),
) -> str:
    """
    Direct LLM generation for small plain-text descriptions.
    No database indexing, no vector store lookup, and no retrieval needed.
    Data goes straight into the LLM.
    """
    desc = project_description.strip()
    if not desc:
        return "Please provide a project description."

    user_inst = user_instruction.strip() if user_instruction else "None provided."
    prompt = TEXT_DIRECT_PROMPT.format(
        project_description=desc,
        inbuilt_instruction=inbuilt_instruction,
        user_instruction=user_inst,
    )
    from app.core.llm import invoke_llm

    return invoke_llm(prompt)


def generate_requirements(
    brief: str,
    question: str = "Generate the software requirements for the project described above.",
) -> str:
    """
    For text briefs: sends straight to LLM without database or vector searching.
    """
    return generate_requirements_direct(project_description=brief, user_instruction=question)


# ─────────────────────────────────────────────────────────────
#  Example
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # One-time: index trusted regulations/standards (put .txt/.md files in ./corpus)
    # print(ingest_reference_corpus("corpus"))

    sample = """
    - The system shall support KYC verification via Aadhaar OCR.
    - Loan disbursal must complete within 30 seconds of approval.
    - All PII must be encrypted at rest using AES-256.
    """
    print(answer_query_from_text(sample, "Generate requirements for KYC and loan disbursal",
                                 user_id="demo", document_id="demo_lending"))