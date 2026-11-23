"""
Uvicorn entry-point.
Run: uvicorn server:app --reload --port 8000
"""
from app.api.app import app  # noqa: F401
