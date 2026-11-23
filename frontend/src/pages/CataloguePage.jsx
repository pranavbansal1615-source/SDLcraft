import { useState, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getSDLCCatalogue } from '../services/api'

const MODEL_ICONS = {
  'Waterfall': '🌊',
  'V-Model': '🔺',
  'Incremental': '📈',
  'Prototype': '🔬',
  'Spiral': '🌀',
  'RAD': '⚡',
  'Agile (Scrum)': '🏃',
  'Agile-DevSecOps': '🛡️',
  'DevSecOps': '🔐',
}

const MODEL_COLORS = {
  'Waterfall': 'from-blue-500 to-cyan-500',
  'V-Model': 'from-violet-500 to-purple-500',
  'Incremental': 'from-green-500 to-emerald-500',
  'Prototype': 'from-orange-500 to-amber-500',
  'Spiral': 'from-pink-500 to-rose-500',
  'RAD': 'from-yellow-500 to-orange-500',
  'Agile (Scrum)': 'from-indigo-500 to-blue-500',
}

const MODEL_BADGES = {
  'Waterfall': { label: 'Predictive', color: 'badge-blue' },
  'V-Model': { label: 'Verification', color: 'badge-indigo' },
  'Incremental': { label: 'Iterative', color: 'badge-emerald' },
  'Prototype': { label: 'Exploratory', color: 'badge-amber' },
  'Spiral': { label: 'Risk-Driven', color: 'badge-red' },
  'RAD': { label: 'Fast Delivery', color: 'badge-amber' },
  'Agile (Scrum)': { label: 'Adaptive', color: 'badge-emerald' },
}

function extractModels(markdown) {
  // Split catalogue by ## headers (model names)
  const sections = markdown.split(/\n## /).filter(Boolean)
  return sections.map(section => {
    const lines = section.split('\n')
    const name = lines[0].replace(/^#+ /, '').trim()
    const content = lines.slice(1).join('\n').trim()
    return { name, content }
  })
}

function ModelCard({ model }) {
  const [expanded, setExpanded] = useState(false)
  const icon = MODEL_ICONS[model.name] || '📋'
  const color = MODEL_COLORS[model.name] || 'from-slate-500 to-slate-600'
  const badge = MODEL_BADGES[model.name]

  // Extract first paragraph as summary
  const lines = model.content.split('\n').filter(Boolean)
  const summary = lines.find(l => !l.startsWith('#') && !l.startsWith('|') && !l.startsWith('-') && l.length > 20) || ''

  return (
    <div className="card overflow-hidden hover:shadow-md transition-all duration-200">
      {/* Color header */}
      <div className={`bg-gradient-to-r ${color} p-5 text-white`}>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="text-3xl">{icon}</div>
            <div>
              <h3 className="font-bold text-lg leading-tight">{model.name}</h3>
              {badge && (
                <span className="inline-block mt-1 text-xs font-semibold bg-white/20 text-white px-2 py-0.5 rounded-full">
                  {badge.label}
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="p-5">
        {summary && (
          <p className="text-slate-600 text-sm leading-relaxed mb-4">{summary.replace(/\*\*/g, '')}</p>
        )}

        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full text-sm text-indigo-600 hover:text-indigo-800 font-medium flex items-center justify-center gap-1.5 bg-indigo-50 hover:bg-indigo-100 py-2 rounded-lg transition-all"
        >
          {expanded ? 'Show less' : 'View full specification'}
          <svg className={`w-4 h-4 transition-transform ${expanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {expanded && (
          <div className="mt-4 prose-custom text-sm animate-fade-in border-t border-slate-100 pt-4">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {model.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  )
}

export default function CataloguePage() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [models, setModels] = useState([])
  const [rawMarkdown, setRawMarkdown] = useState('')
  const [viewMode, setViewMode] = useState('cards') // cards | raw

  useEffect(() => {
    getSDLCCatalogue()
      .then(data => {
        const md = data.catalogue || data
        setRawMarkdown(md)
        setModels(extractModels(md))
        setLoading(false)
      })
      .catch(err => {
        setError(err?.message || 'Failed to load catalogue')
        setLoading(false)
      })
  }, [])

  return (
    <div className="min-h-screen bg-slate-50 page-enter">
      {/* Header */}
      <div className="bg-gradient-to-r from-brand-950 to-violet-900 text-white px-4 sm:px-6 lg:px-8 py-10">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center gap-2 text-indigo-300 text-sm mb-3">
            <span>🏠</span> <span>/</span> <span>Catalogue</span>
          </div>
          <h1 className="text-3xl font-bold mb-2">SDLC Model Catalogue</h1>
          <p className="text-indigo-300 max-w-2xl">
            Explore all {models.length || 7} SDLC models available in the pipeline — their specifications, strengths, and finance-domain fit.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Controls */}
        <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
          <div>
            {!loading && (
              <span className="text-slate-500 text-sm">
                {models.length} models in catalogue
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="inline-flex bg-slate-100 p-1 rounded-xl gap-1">
              <button
                onClick={() => setViewMode('cards')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${viewMode === 'cards' ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                🗂️ Cards
              </button>
              <button
                onClick={() => setViewMode('raw')}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${viewMode === 'raw' ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}
              >
                📄 Raw
              </button>
            </div>
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {Array.from({ length: 7 }).map((_, i) => (
              <div key={i} className="card overflow-hidden">
                <div className="shimmer h-24" />
                <div className="p-5 space-y-3">
                  <div className="shimmer h-4 w-3/4 rounded" />
                  <div className="shimmer h-3 w-full rounded" />
                  <div className="shimmer h-3 w-5/6 rounded" />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-center">
            <div className="text-3xl mb-3">⚠️</div>
            <h3 className="font-bold text-red-800 mb-1">Failed to load catalogue</h3>
            <p className="text-red-600 text-sm">{error}</p>
            <p className="text-red-400 text-xs mt-2">Make sure the backend is running on port 8000.</p>
          </div>
        )}

        {/* Cards View */}
        {!loading && !error && viewMode === 'cards' && (
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5 animate-fade-in">
            {models.map((model, i) => (
              <ModelCard key={i} model={model} />
            ))}
          </div>
        )}

        {/* Raw markdown view */}
        {!loading && !error && viewMode === 'raw' && (
          <div className="card p-7 animate-fade-in">
            <div className="prose-custom max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {rawMarkdown}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {/* Finance context note */}
        {!loading && !error && (
          <div className="mt-8 bg-indigo-50 border border-indigo-100 rounded-2xl p-5 flex gap-4 items-start">
            <div className="text-2xl mt-0.5">🏦</div>
            <div>
              <h4 className="font-semibold text-indigo-800 mb-1">Finance-Domain Weighting</h4>
              <p className="text-indigo-600 text-sm leading-relaxed">
                When the pipeline runs, each model is scored using Multi-Criteria Decision Analysis (MCDA) with criteria
                weighted for financial services: regulatory compliance, auditability, security, risk management, and
                stakeholder alignment. The scores you see in the pipeline report reflect these finance-specific priorities.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
