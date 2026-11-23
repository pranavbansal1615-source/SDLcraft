import { useState, useRef } from 'react'

export default function PDFUploadPanel({
  projectName,
  onProjectNameChange,
  file,
  onFileChange,
  prompt,
  onPromptChange,
}) {
  const [dragging, setDragging] = useState(false)
  const [showInbuiltPrompt, setShowInbuiltPrompt] = useState(false)
  const inputRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped && dropped.type === 'application/pdf') {
      onFileChange(dropped)
    }
  }

  const handleFileInput = (e) => {
    const selected = e.target.files[0]
    if (selected) onFileChange(selected)
  }

  const formatSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  const defaultPromptPlaceholder =
    "e.g. Extract requirements for a digital lending & loan disbursement module with Aadhaar KYC and RBI compliance."

  return (
    <div className="space-y-5">
      {/* Project Name */}
      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-1.5">
          Project Name <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          className="input-field"
          placeholder="e.g. Trade Finance Automation"
          value={projectName}
          onChange={e => onProjectNameChange(e.target.value)}
        />
      </div>

      {/* Drop Zone */}
      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-1.5">
          PDF Document <span className="text-red-500">*</span>
        </label>

        {!file ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current.click()}
            className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
              dragging
                ? 'border-violet-400 bg-violet-50'
                : 'border-slate-200 bg-slate-50 hover:border-violet-300 hover:bg-violet-50/50'
            }`}
          >
            <div className="text-4xl mb-3">{dragging ? '📂' : '📄'}</div>
            <p className="font-semibold text-slate-700 mb-1">
              {dragging ? 'Drop your PDF here' : 'Drag & drop your PDF'}
            </p>
            <p className="text-slate-400 text-sm mb-3">or click to browse files</p>
            <span className="inline-block bg-violet-100 text-violet-700 text-xs font-semibold px-3 py-1 rounded-full">
              PDF files only (SRS, Proposals, RFP, BRD)
            </span>
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              className="hidden"
              onChange={handleFileInput}
            />
          </div>
        ) : (
          <div className="border-2 border-emerald-200 bg-emerald-50 rounded-2xl p-5 flex items-center gap-4">
            <div className="w-11 h-11 bg-emerald-100 rounded-xl flex items-center justify-center text-2xl flex-shrink-0">
              📋
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-slate-800 truncate">{file.name}</p>
              <p className="text-sm text-slate-500 mt-0.5">{formatSize(file.size)}</p>
            </div>
            <button
              type="button"
              onClick={() => { onFileChange(null); if (inputRef.current) inputRef.current.value = '' }}
              className="text-slate-400 hover:text-red-500 transition-colors p-1"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* User Intent Prompt */}
      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-1.5">
          What do you want to build / extract? <span className="text-violet-600 font-normal">(User Prompt for Vector Search)</span>
        </label>
        <textarea
          rows={3}
          className="input-field resize-none text-sm"
          placeholder={defaultPromptPlaceholder}
          value={prompt}
          onChange={e => onPromptChange(e.target.value)}
        />
        <p className="text-xs text-slate-400 mt-1">
          💡 This prompt is combined with our inbuilt finance prompt to run semantic vector search over the PDF chunks.
        </p>
      </div>

      {/* Inbuilt Prompt Preview */}
      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5">
        <button
          type="button"
          onClick={() => setShowInbuiltPrompt(!showInbuiltPrompt)}
          className="w-full flex items-center justify-between text-left text-xs font-semibold text-slate-700 hover:text-slate-900"
        >
          <span className="flex items-center gap-1.5 text-violet-700 font-medium">
            <span>⚙️</span> Inbuilt Finance Prompt (Applied Automatically)
          </span>
          <span className="text-slate-400">{showInbuiltPrompt ? 'Hide' : 'View'}</span>
        </button>

        {showInbuiltPrompt && (
          <div className="mt-2.5 pt-2.5 border-t border-slate-200 text-xs text-slate-600 leading-relaxed font-mono bg-white p-3 rounded-lg border border-slate-100">
            "Using the uploaded PDF as primary evidence, extract specific, testable, IEEE-830 compliant finance requirements covering Functional, Security (AES-256/TLS/MFA/RBAC), Compliance (RBI/SEBI/DPDP/PCI-DSS), Performance, and Usability with strict evidence citations [E1..En]."
          </div>
        )}
      </div>
    </div>
  )
}
