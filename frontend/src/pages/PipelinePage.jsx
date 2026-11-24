import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import ModeToggle from '../components/ModeToggle'
import TextInputPanel from '../components/TextInputPanel'
import PDFUploadPanel from '../components/PDFUploadPanel'
import RequirementsDisplay from '../components/RequirementsDisplay'
import SDLCReportDisplay from '../components/SDLCReportDisplay'
import LoadingOverlay from '../components/LoadingOverlay'
import ErrorBanner from '../components/ErrorBanner'
import { checkBackendHealth, runPipelineFromText, runPipelineFromPDF } from '../services/api'

export default function PipelinePage() {
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState(searchParams.get('mode') === 'pdf' ? 'pdf' : 'text')
  
  // Form state
  const [projectName, setProjectName] = useState('')
  const [description, setDescription] = useState('')
  const [query, setQuery] = useState('')
  const [file, setFile] = useState(null)
  const [pdfPrompt, setPdfPrompt] = useState('')

  // Pipeline state
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    let active = true
    const refreshBackendStatus = () => checkBackendHealth()
      .then(() => {
        if (active) setBackendStatus('connected')
      })
      .catch(() => {
        if (active) setBackendStatus('unavailable')
      })
    refreshBackendStatus()
    const intervalId = setInterval(refreshBackendStatus, 10000)
    return () => {
      active = false
      clearInterval(intervalId)
    }
  }, [])

  const isValid = mode === 'text'
    ? projectName.trim() && description.trim()
    : projectName.trim() && file

  const stepTimings = [0, 1500, 3000, 4500]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)
    setLoadingStep(0)

    // Animate loading steps
    const timers = stepTimings.map((t, i) =>
      setTimeout(() => setLoadingStep(i), t)
    )

    try {
      let res
      if (mode === 'text') {
        res = await runPipelineFromText(
          description,
          projectName,
          query || undefined,
          'default_user'
        )
      } else {
        const promptToSend = pdfPrompt.trim() || 'Extract all functional, security, compliance, and performance requirements from this document.'
        res = await runPipelineFromPDF(file, projectName, promptToSend, 'default_user')
      }
      setResult({ ...res, source: mode })
    } catch (err) {
      const msg = err?.response?.data?.detail || err?.message || 'Unknown error'
      setError(msg)
    } finally {
      timers.forEach(clearTimeout)
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setDescription('')
    setProjectName('')
    setQuery('')
    setFile(null)
    setPdfPrompt('')
  }

  return (
    <div className="min-h-screen bg-slate-50 page-enter">
      {loading && <LoadingOverlay step={loadingStep} />}

      {/* Page Header */}
      <div className="bg-gradient-to-r from-brand-950 to-indigo-900 text-white px-4 sm:px-6 lg:px-8 py-10">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-2 text-indigo-300 text-sm mb-3">
            <span>🏠</span> <span>/</span> <span>Pipeline</span>
          </div>
          <h1 className="text-3xl font-bold mb-2">SDLC Analysis Pipeline</h1>
          <p className="text-indigo-300 max-w-2xl">
            Describe your finance project or upload a document. The AI will extract requirements and recommend the optimal SDLC model.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!result ? (
          /* Input Form */
          <div className="max-w-3xl mx-auto">
            {/* Mode Toggle */}
            <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
              <ModeToggle mode={mode} onChange={(m) => { setMode(m); setError(null) }} />
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <div className={`w-2 h-2 rounded-full ${
                  backendStatus === 'connected'
                    ? 'bg-emerald-400'
                    : backendStatus === 'checking'
                      ? 'bg-amber-400 animate-pulse'
                      : 'bg-red-400'
                }`}></div>
                {backendStatus === 'checking'
                  ? 'Checking backend…'
                  : backendStatus === 'connected'
                    ? 'Backend connected'
                    : 'Backend unavailable — start the API server'}
              </div>
            </div>

            {/* Error */}
            {error && <div className="mb-5"><ErrorBanner message={error} onDismiss={() => setError(null)} /></div>}

            {/* Form Card */}
            <div className="card">
              <div className={`h-1.5 ${mode === 'text' ? 'bg-gradient-to-r from-indigo-500 to-violet-500' : 'bg-gradient-to-r from-violet-500 to-pink-500'}`} />
              <div className="p-7">
                <form onSubmit={handleSubmit}>
                  {mode === 'text' ? (
                    <TextInputPanel
                      projectName={projectName}
                      onProjectNameChange={setProjectName}
                      description={description}
                      onDescriptionChange={setDescription}
                      query={query}
                      onQueryChange={setQuery}
                    />
                  ) : (
                    <PDFUploadPanel
                      projectName={projectName}
                      onProjectNameChange={setProjectName}
                      file={file}
                      onFileChange={setFile}
                      prompt={pdfPrompt}
                      onPromptChange={setPdfPrompt}
                    />
                  )}

                  {/* Submit */}
                  <div className="mt-7 flex items-center gap-3">
                    <button
                      type="submit"
                      disabled={!isValid || loading}
                      className="btn-primary flex items-center gap-2 flex-1 justify-center text-base"
                    >
                      {loading ? (
                        <>
                          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                          Analysing...
                        </>
                      ) : (
                        <>🚀 Run Pipeline</>
                      )}
                    </button>
                  </div>

                  {!isValid && (
                    <p className="text-xs text-slate-400 mt-2 text-center">
                      {mode === 'text'
                        ? 'Enter a project name and description to continue'
                        : 'Enter a project name and upload a PDF to continue'
                      }
                    </p>
                  )}
                </form>
              </div>
            </div>

            {/* Info boxes */}
            <div className="mt-5 grid sm:grid-cols-3 gap-3">
              {[
                { icon: '⚡', title: 'Fast Analysis', desc: 'Local LM Studio inference in seconds' },
                { icon: '🔒', title: 'Finance Domain', desc: 'Compliance-aware requirements' },
                { icon: '📊', title: '7 SDLC Models', desc: 'MCDA-ranked recommendation' },
              ].map((info, i) => (
                <div key={i} className="bg-white border border-slate-100 rounded-xl p-4 flex gap-3 items-start">
                  <span className="text-xl">{info.icon}</span>
                  <div>
                    <div className="text-sm font-semibold text-slate-700">{info.title}</div>
                    <div className="text-xs text-slate-400">{info.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          /* Results View */
          <div className="animate-fade-in">
            {/* Results header */}
            <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="badge badge-emerald">✓ Analysis Complete</span>
                  <span className="badge badge-slate">{result.project_name || 'Project'}</span>
                </div>
                <h2 className="text-xl font-bold text-slate-800">Pipeline Results</h2>
              </div>
              <div className="flex gap-2">
                <button onClick={handleReset} className="btn-secondary text-sm flex items-center gap-1.5">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v1m0 14v1m8.66-9h-1M4.34 12h-1m15.07-6.93l-.71.71M5.64 18.36l-.71.71M18.36 18.36l-.71-.71M5.64 5.64l-.71-.71" />
                  </svg>
                  New Analysis
                </button>
              </div>
            </div>

            {/* Top recommendation highlight */}
            {result.recommended_model && (
              <div className="bg-gradient-to-r from-brand-950 to-indigo-900 text-white rounded-2xl p-5 mb-6 flex items-center gap-4">
                <div className="text-3xl">🏆</div>
                <div>
                  <div className="text-indigo-300 text-xs font-semibold uppercase tracking-wider">Top SDLC Recommendation for</div>
                  <div className="font-bold text-xl">{result.recommended_model}</div>
                  <div className="text-indigo-300 text-sm">{result.project_name}</div>
                </div>
              </div>
            )}

            {/* Two-column results */}
            <div className="grid lg:grid-cols-2 gap-6 items-start">
              {result.requirements_markdown && (
                <RequirementsDisplay
                  markdown={result.requirements_markdown}
                  source={result.source}
                />
              )}
              {result.sdlc_report_markdown && (
                <SDLCReportDisplay
                  markdown={result.sdlc_report_markdown}
                  recommendedModel={result.recommended_model}
                />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
