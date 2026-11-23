import { useState } from 'react'

export default function TextInputPanel({ projectName, onProjectNameChange, description, onDescriptionChange, query, onQueryChange }) {
  const [showAdvanced, setShowAdvanced] = useState(false)

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
          placeholder="e.g. Digital Lending & KYC Platform"
          value={projectName}
          onChange={e => onProjectNameChange(e.target.value)}
        />
      </div>

      {/* Description */}
      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-1.5">
          Project Description <span className="text-red-500">*</span>
        </label>
        <textarea
          className="input-field resize-none font-mono text-sm"
          rows={10}
          placeholder={`Describe your finance project in detail. For example:\n\nWe are building a digital lending platform for a mid-size NBFC. The platform must support:\n- Loan origination and underwriting workflows\n- KYC document verification using Aadhaar & PAN APIs\n- Real-time credit bureau integrations (CIBIL, Experian)\n- RBI compliance reporting\n- Mobile-first borrower experience with multi-language support\n- Secure audit trails and data encryption at rest...`}
          value={description}
          onChange={e => onDescriptionChange(e.target.value)}
        />
        <p className="text-xs text-slate-400 mt-1.5">
          💡 Be specific about features, compliance needs, users, and integrations for best results.
        </p>
      </div>

      {/* Advanced: Custom Query */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-700 transition-colors"
        >
          <svg className={`w-4 h-4 transition-transform ${showAdvanced ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          Advanced: Custom extraction query
        </button>
        {showAdvanced && (
          <div className="mt-3 animate-fade-in">
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">
              Custom Query <span className="text-slate-400 font-normal">(optional)</span>
            </label>
            <input
              type="text"
              className="input-field"
              placeholder="e.g. Focus on security and compliance requirements only"
              value={query}
              onChange={e => onQueryChange(e.target.value)}
            />
            <p className="text-xs text-slate-400 mt-1.5">
              Override the default extraction query to focus on specific requirement categories.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
