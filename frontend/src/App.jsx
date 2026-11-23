import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import PipelinePage from './pages/PipelinePage'
import CataloguePage from './pages/CataloguePage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/pipeline" element={<PipelinePage />} />
            <Route path="/catalogue" element={<CataloguePage />} />
          </Routes>
        </main>
        <footer className="border-t border-slate-200 bg-white py-5 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 bg-indigo-600 rounded flex items-center justify-center">
                <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span className="text-sm font-semibold text-slate-700">SDLcraft</span>
            </div>
            <p className="text-xs text-slate-400">
              AI-Powered Finance SDLC Intelligence · Built with LM Studio &amp; RAG
            </p>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  )
}
