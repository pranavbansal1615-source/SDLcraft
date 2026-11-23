const steps = [
  'Ingesting project data...',
  'Extracting requirements with RAG...',
  'Running MCDA analysis...',
  'Generating SDLC recommendation...',
]

export default function LoadingOverlay({ step = 0 }) {
  return (
    <div className="fixed inset-0 bg-slate-900/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl shadow-2xl p-10 max-w-sm w-full text-center animate-fade-in">
        {/* Spinner */}
        <div className="relative w-20 h-20 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full border-4 border-indigo-100" />
          <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-indigo-600 animate-spin" />
          <div className="absolute inset-3 rounded-full bg-indigo-50 flex items-center justify-center">
            <span className="text-2xl">🧠</span>
          </div>
        </div>

        <h3 className="font-bold text-slate-800 text-lg mb-1">Analysing requirements</h3>
        <p className="text-slate-400 text-sm mb-7">This may take a few seconds…</p>

        {/* Steps */}
        <div className="space-y-2.5 text-left">
          {steps.map((s, i) => (
            <div key={i} className={`flex items-center gap-3 text-sm transition-all duration-500 ${
              i < step ? 'text-emerald-600' : i === step ? 'text-indigo-600' : 'text-slate-300'
            }`}>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold ${
                i < step ? 'bg-emerald-100' : i === step ? 'bg-indigo-100 animate-pulse' : 'bg-slate-100'
              }`}>
                {i < step ? '✓' : i + 1}
              </div>
              <span className={i === step ? 'font-semibold' : ''}>{s}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
