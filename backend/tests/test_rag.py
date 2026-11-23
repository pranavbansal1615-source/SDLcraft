"""
Unit tests for the Hybrid RAG module (app/rag/rag.py).
Tests document splitting, text cleaning, tokenization,
hybrid store operations, BM25 retrieval, citation extraction,
and evidence index generation.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from app.rag.rag import (
    clean_page_text,
    split_docs,
    tokenize,
    Hit,
    HybridStore,
    build_context,
    _validate_and_index,
    _LABEL_RE,
    PROJECT_KIND,
    REFERENCE_KIND,
    CORPUS_USER,
)


class TestTextProcessing:
    def test_clean_page_text_removes_page_numbers(self):
        raw = "Header text\n3\nPage 4 of 10\np. 5\nReal requirements content\n---\n1/3"
        cleaned = clean_page_text(raw)
        assert "3" not in cleaned.splitlines()
        assert "Page 4 of 10" not in cleaned
        assert "Real requirements content" in cleaned

    def test_clean_page_text_collapses_newlines(self):
        raw = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_page_text(raw)
        assert "\n\n\n" not in cleaned
        assert "Line 1\n\nLine 2" == cleaned

    def test_split_docs_splits_into_chunks(self):
        doc = Document(
            page_content="## Requirement 1\nThe system shall authenticate users.\n\n## Requirement 2\nThe system shall log audits.",
            metadata={"source_file": "test.txt", "page_number": 1}
        )
        chunks = split_docs([doc], chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 2
        for c in chunks:
            assert "source_file" in c.metadata


class TestTokenize:
    def test_tokenize_filters_stopwords_and_short(self):
        text = "The system shall encrypt all customer PII data with AES-256."
        tokens = tokenize(text)
        # Stopwords 'the', 'system', 'shall', 'all', 'with' should be removed
        assert "the" not in tokens
        assert "system" not in tokens
        assert "shall" not in tokens
        assert "encrypt" in tokens
        assert "customer" in tokens
        assert "pii" in tokens
        assert "data" in tokens
        assert "aes" in tokens
        assert "256" in tokens


class TestCitationValidation:
    def test_extract_evidence_labels(self):
        text = "According to [E1] and [E3], loan disbursals must finish in 30 seconds [E2]."
        labels = {f"E{m}" for m in _LABEL_RE.findall(text)}
        assert labels == {"E1", "E2", "E3"}

    def test_extract_evidence_labels_empty(self):
        text = "No evidence citations are present here."
        labels = {f"E{m}" for m in _LABEL_RE.findall(text)}
        assert labels == set()

    def test_validate_and_index_adds_table(self):
        labelled = {
            "E1": Hit(
                chunk_id="user1/doc1#chunk-1",
                text="Loan disbursals under 30s.",
                meta={"source_file": "loan_policy.pdf", "page_number": 2},
                score=0.95,
                found_by=["bm25", "vector"]
            )
        }
        answer = "The system shall disburse loans rapidly [E1]."
        final = _validate_and_index(answer, labelled)
        assert "### Evidence Index" in final
        assert "| E1 | loan_policy.pdf | 2 |" in final
        assert "`user1/doc1#chunk-1`" in final
        assert "bm25+vector" in final

    def test_validate_and_index_flags_hallucinated_citations(self):
        labelled = {
            "E1": Hit(
                chunk_id="user1/doc1#chunk-1",
                text="Valid chunk content",
                meta={"source_file": "file.pdf", "page_number": 1},
                score=0.9,
                found_by=["bm25"]
            )
        }
        answer = "Requirement based on [E1] and non-existent [E9]."
        final = _validate_and_index(answer, labelled)
        assert "⚠️ **Citation check:**" in final
        assert "E9" in final


class TestBuildContext:
    def test_build_context_labels_chunks(self):
        project_hits = [
            Hit(
                chunk_id="p1",
                text="Project specific requirement.",
                meta={"source_file": "project.md", "page_number": 1},
                score=0.9
            )
        ]
        ref_hits = [
            Hit(
                chunk_id="r1",
                text="RBI regulatory clause.",
                meta={"source_file": "rbi.txt", "page_number": 0},
                score=0.85
            )
        ]
        context, labelled = build_context(project_hits, ref_hits, max_chars=5000)
        assert "E1" in labelled
        assert "E2" in labelled
        assert "[E1 | project.md | Page 1]" in context
        assert "[E2 | rbi.txt | Page 0]" in context
        assert "PROJECT CONTEXT" in context
        assert "REFERENCE CORPUS" in context


class TestHybridStoreMocked:
    @patch("chromadb.PersistentClient")
    def test_store_initialization(self, mock_chroma_client, tmp_path):
        mock_col = MagicMock()
        mock_col.get.return_value = {"ids": []}
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_col

        with patch("app.rag.rag.CHROMA_DIR", str(tmp_path)):
            store = HybridStore()
            assert store.exists("doc123", "user123") is False
            assert store.project_where("doc1", "u1") == {
                "$and": [{"kind": PROJECT_KIND}, {"document_id": "doc1"}, {"user_id": "u1"}]
            }
            assert store.reference_where() == {"kind": REFERENCE_KIND}
