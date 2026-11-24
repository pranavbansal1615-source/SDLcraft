/**
 * SDLcraft API Service
 * Thin wrapper around the FastAPI backend.
 */

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

// ── Helper ────────────────────────────────────────────────────────────────────

async function request(endpoint, options = {}) {
  const res = await fetch(`${BASE_URL}${endpoint}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "API request failed");
  }
  return res.json();
}

// ── Ingestion ─────────────────────────────────────────────────────────────────

/**
 * Ingest a free-text project description.
 * @param {string} text
 * @param {string} projectName
 * @param {string} [documentId]
 * @param {string} [userId]
 */
export async function ingestText(text, projectName, documentId = null, userId = "default_user") {
  return request("/ingest/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      project_name: projectName,
      document_id: documentId,
      user_id: userId,
    }),
  });
}

/**
 * Upload and ingest a PDF document.
 * @param {File} file
 * @param {string} [userId]
 * @param {string} [documentId]
 */
export async function ingestPDF(file, userId = "default_user", documentId = null) {
  const form = new FormData();
  form.append("file", file);
  form.append("user_id", userId);
  if (documentId) form.append("document_id", documentId);
  return request("/ingest/pdf", { method: "POST", body: form });
}

// ── Requirements ──────────────────────────────────────────────────────────────

/**
 * Query requirements from an indexed document.
 * @param {string} query
 * @param {string} documentId
 */
export async function queryRequirements(query, documentId, userId = "default_user") {
  return request("/requirements/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, document_id: documentId, user_id: userId }),
  });
}

// ── SDLC ─────────────────────────────────────────────────────────────────────

/**
 * Get SDLC recommendation from a list of requirements.
 * @param {string[]} requirements
 * @param {string} projectName
 */
export async function recommendSDLC(requirements, projectName) {
  return request("/sdlc/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requirements, project_name: projectName }),
  });
}

/**
 * Fetch the full SDLC model catalogue.
 */
export async function getSDLCCatalogue() {
  return request("/sdlc/catalogue");
}

export async function checkBackendHealth() {
  return request("/health");
}

// ── Full Pipeline ─────────────────────────────────────────────────────────────

/**
 * Run the complete text → requirements → SDLC pipeline.
 * @param {string} projectDescription
 * @param {string} projectName
 * @param {string} [query]
 * @param {string} [userId]
 * @param {string} [documentId]
 */
export async function runPipelineFromText(
  projectDescription,
  projectName,
  query = "Generate the complete requirement set for this project.",
  userId = "default_user",
  documentId = null
) {
  return request("/pipeline/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_description: projectDescription,
      project_name: projectName,
      query,
      user_id: userId,
      document_id: documentId,
    }),
  });
}

/**
 * Run the complete PDF → requirements → SDLC pipeline.
 * @param {File} file
 * @param {string} projectName
 * @param {string} [userId]
 */
export async function runPipelineFromPDF(
  file,
  projectName,
  query,
  userId = "default_user"
) {
  const form = new FormData();
  form.append("file", file);
  form.append("project_name", projectName);
  form.append("user_id", userId);
  form.append("query", query);
  return request("/pipeline/pdf", { method: "POST", body: form });
}
