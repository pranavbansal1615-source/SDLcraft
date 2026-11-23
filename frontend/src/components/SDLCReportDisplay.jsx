import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const getSuitabilityBadge = (label) => {
  if (!label) return 'badge-slate'
  const l = label.toLowerCase()
  if (l.includes('highly')) return 'badge-emerald'
  if (l.includes('recommended')) return 'badge-blue'
  if (l.includes('moderate')) return 'badge-amber'
  if (l.includes('unsuitable') || l.includes('not')) return 'badge-red'
  return 'badge-slate'
}

const getRankColor = (rank) => {
  if (rank === 1) return 'text-yellow-600 bg-yellow-50'
  if (rank === 2) return 'text-slate-500 bg-slate-50'
  if (rank === 3) return 'text-orange-600 bg-orange-50'
  return 'text-slate-400 bg-white'
}

function parseReport(markdown) {
  // Try to extract ranking table from markdown
  const lines = markdown.split('\n')
  const tableLines = lines.filter(l => l.includes('|') && (l.includes('Model') || /\d+\./.test(l) || /\d+\.\d+/.test(l)))
  return { raw: markdown, hasTable: tableLines.length > 2 }
}

export default function SDLCReportDisplay({ markdown, recommendedModel }) {
  const [showFullReport, setShowFullReport] = useState(false)
  const { raw } = parseReport(markdown)

  return (
    <div className="card animate-slide-up h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-gradient-to-r from-emerald-50 to-white">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-emerald-100 rounded-xl flex items-center justify-center text-lg">🏆</div>
          <div>
            <h3 className="font-bold text-slate-800 text-sm">SDLC Recommendation</h3>
            <span className="badge badge-emerald text-[10px]">MCDA Ranked</span>
          </div>
        </div>
      </div>

      {/* Recommended model banner */}
      {recommendedModel && (
        <div className="mx-6 mt-5 bg-gradient-to-r from-emerald-600 to-teal-600 rounded-xl p-4 text-white flex items-center gap-3 shadow-sm">
          <div className="text-2xl">🎯</div>
          <div>
            <div className="text-xs font-semibold text-emerald-200 uppercase tracking-wider mb-0.5">Recommended Model</div>
            <div className="font-bold text-lg leading-tight">{recommendedModel}</div>
          </div>
          <div className="ml-auto bg-white/20 text-white text-xs font-bold px-3 py-1.5 rounded-full">
            #1 Match
          </div>
        </div>
      )}

      {/* Full Report */}
      <div className="flex-1 overflow-auto px-6 py-5">
        <div className="mb-4 flex items-center justify-between">
          <span className="text-sm font-semibold text-slate-700">Full Analysis Report</span>
          <button
            onClick={() => setShowFullReport(!showFullReport)}
            className="text-xs text-indigo-600 hover:text-indigo-800 font-medium flex items-center gap-1 bg-indigo-50 hover:bg-indigo-100 px-3 py-1.5 rounded-lg transition-all"
          >
            {showFullReport ? 'Collapse' : 'Expand full report'}
            <svg className={`w-3.5 h-3.5 transition-transform ${showFullReport ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>

        {/* Show a preview by default */}
        <div className={`prose-custom overflow-hidden transition-all duration-300 ${showFullReport ? '' : 'max-h-96 relative'}`}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {raw}
          </ReactMarkdown>
          {!showFullReport && (
            <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-white to-transparent pointer-events-none" />
          )}
        </div>

        {!showFullReport && (
          <button
            onClick={() => setShowFullReport(true)}
            className="mt-3 w-full text-sm text-indigo-600 font-medium hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 py-2.5 rounded-xl transition-all"
          >
            Show full report ↓
          </button>
        )}
      </div>
    </div>
  )
}
