export default function ModeToggle({ mode, onChange }) {
  return (
    <div className="inline-flex bg-slate-100 p-1 rounded-xl gap-1">
      <button
        onClick={() => onChange('text')}
        className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${
          mode === 'text'
            ? 'bg-white text-indigo-700 shadow-sm'
            : 'text-slate-500 hover:text-slate-700'
        }`}
      >
        <span>📝</span> Text Mode
      </button>
      <button
        onClick={() => onChange('pdf')}
        className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all duration-200 ${
          mode === 'pdf'
            ? 'bg-white text-violet-700 shadow-sm'
            : 'text-slate-500 hover:text-slate-700'
        }`}
      >
        <span>📄</span> PDF Upload
      </button>
    </div>
  )
}
