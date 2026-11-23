export default function ErrorBanner({ message, onDismiss }) {
  if (!message) return null

  return (
    <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3 animate-slide-up">
      <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0 text-lg">
        ⚠️
      </div>
      <div className="flex-1 min-w-0">
        <p className="font-semibold text-red-800 text-sm mb-0.5">Something went wrong</p>
        <p className="text-red-600 text-sm break-words">{message}</p>
        <p className="text-red-400 text-xs mt-1">Make sure the backend is running on port 8000.</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="text-red-400 hover:text-red-600 transition-colors p-1 flex-shrink-0"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  )
}
