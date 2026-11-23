import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function RequirementsDisplay({ markdown, source }) {
  const [copied, setCopied] = useState(false)
  const [collapsed, setCollapsed] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(markdown).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="card animate-slide-up h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100 bg-gradient-to-r from-indigo-50 to-white">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-indigo-100 rounded-xl flex items-center justify-center text-lg">📋</div>
          <div>
            <h3 className="font-bold text-slate-800 text-sm">Extracted Requirements</h3>
            <div className="flex items-center gap-2 mt-0.5">
              <span className={`badge ${source === 'pdf' ? 'badge-violet' : 'badge-indigo'} text-[10px]`}
                style={{ backgroundColor: source === 'pdf' ? '#f3e8ff' : '#e0e7ff', color: source === 'pdf' ? '#6b21a8' : '#3730a3' }}>
                {source === 'pdf' ? '📄 PDF' : '📝 Text'}
              </span>
              <span className="badge badge-emerald text-[10px]">AI Generated</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-indigo-600 bg-white hover:bg-indigo-50 border border-slate-200 px-3 py-1.5 rounded-lg transition-all"
          >
            {copied ? (
              <><span>✓</span> Copied!</>
            ) : (
              <><svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10" />
              </svg> Copy</>
            )}
          </button>
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-100 transition-all"
          >
            <svg className={`w-4 h-4 transition-transform ${collapsed ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
          </button>
        </div>
      </div>

      {/* Content */}
      {!collapsed && (
        <div className="flex-1 overflow-auto px-6 py-5">
          <div className="prose-custom">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {markdown}
            </ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  )
}
