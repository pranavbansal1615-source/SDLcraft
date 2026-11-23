import { Link } from 'react-router-dom'

const steps = [
  {
    icon: '📋',
    title: 'Describe Your Project',
    desc: 'Enter a natural language description or upload a finance project PDF.',
    color: 'bg-indigo-50 border-indigo-100',
    iconBg: 'bg-indigo-100',
  },
  {
    icon: '🔍',
    title: 'AI Extracts Requirements',
    desc: 'RAG pipeline with BM25 + semantic search identifies key functional & non-functional requirements.',
    color: 'bg-violet-50 border-violet-100',
    iconBg: 'bg-violet-100',
  },
  {
    icon: '📊',
    title: 'SDLC Model Selection',
    desc: 'Multi-criteria decision analysis ranks all SDLC models for your finance domain needs.',
    color: 'bg-cyan-50 border-cyan-100',
    iconBg: 'bg-cyan-100',
  },
  {
    icon: '📄',
    title: 'Get Your Report',
    desc: 'Receive a detailed report with rankings, rationale, workflow phases, and risk assessment.',
    color: 'bg-emerald-50 border-emerald-100',
    iconBg: 'bg-emerald-100',
  },
]

const features = [
  { icon: '🏦', title: 'Finance-First', desc: 'Built exclusively for banking, fintech, and financial service projects.' },
  { icon: '🤖', title: 'RAG-Powered', desc: 'Hybrid BM25 + vector retrieval for precise requirement extraction.' },
  { icon: '⚖️', title: 'MCDA Ranking', desc: 'Multi-criteria analysis across 7 SDLC models with sensitivity analysis.' },
  { icon: '🔒', title: 'Compliance Aware', desc: 'Understands regulatory requirements like PCI-DSS, RBI, SEBI mandates.' },
  { icon: '📑', title: 'PDF Support', desc: 'Ingest SRS documents, project proposals, and RFP documents directly.' },
  { icon: '⚡', title: 'Local LM Studio', desc: 'Fast local inference powered by LM Studio for real-time analysis.' },
]

const sdlcModels = ['Waterfall', 'V-Model', 'Incremental', 'Prototype', 'Spiral', 'RAD', 'Agile (Scrum)']

export default function HomePage() {
  return (
    <div className="page-enter">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-brand-950 via-indigo-900 to-violet-900 text-white overflow-hidden">
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-violet-500/10 rounded-full blur-3xl" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-indigo-600/5 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 sm:py-28">
          <div className="text-center max-w-4xl mx-auto">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm border border-white/20 text-indigo-200 text-sm px-4 py-2 rounded-full mb-6 font-medium">
              <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
              AI-Powered · Finance Domain · 7 SDLC Models
            </div>

            {/* Title */}
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6">
              <span className="text-white">SDL</span>
              <span className="text-indigo-400">craft</span>
            </h1>
            <p className="text-xl sm:text-2xl text-indigo-200 font-medium mb-4">
              AI-Powered Finance SDLC Pipeline
            </p>
            <p className="text-indigo-300/80 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed mb-10">
              Automatically extract software requirements and select the optimal SDLC model
              for your financial services project — in seconds.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link
                to="/pipeline"
                className="w-full sm:w-auto bg-indigo-500 hover:bg-indigo-400 text-white font-semibold px-8 py-3.5 rounded-xl transition-all duration-200 shadow-lg hover:shadow-indigo-500/30 text-center"
              >
                🚀 Start Pipeline
              </Link>
              <Link
                to="/catalogue"
                className="w-full sm:w-auto bg-white/10 hover:bg-white/20 backdrop-blur-sm border border-white/20 text-white font-semibold px-8 py-3.5 rounded-xl transition-all duration-200 text-center"
              >
                📚 View SDLC Catalogue
              </Link>
            </div>

            {/* SDLC model chips */}
            <div className="mt-10 flex flex-wrap justify-center gap-2">
              {sdlcModels.map(m => (
                <span key={m} className="bg-white/10 text-indigo-200 text-xs px-3 py-1.5 rounded-full font-medium border border-white/10">
                  {m}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Input Mode Cards */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-slate-800 mb-3">Choose Your Input Mode</h2>
          <p className="text-slate-500 text-base max-w-xl mx-auto">
            Start with a project description or upload an existing document — the pipeline handles both.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 gap-6 max-w-3xl mx-auto">
          <Link to="/pipeline?mode=text" className="card-hover group p-8 text-center">
            <div className="w-16 h-16 bg-indigo-100 group-hover:bg-indigo-200 rounded-2xl flex items-center justify-center mx-auto mb-5 transition-colors text-3xl">
              📝
            </div>
            <h3 className="text-lg font-bold text-slate-800 mb-2">Text Mode</h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              Describe your finance project in plain English. Our AI will extract and structure requirements automatically.
            </p>
            <div className="mt-5 text-indigo-600 text-sm font-semibold group-hover:text-indigo-700 flex items-center justify-center gap-1.5">
              Start with text <span className="group-hover:translate-x-1 transition-transform">→</span>
            </div>
          </Link>

          <Link to="/pipeline?mode=pdf" className="card-hover group p-8 text-center">
            <div className="w-16 h-16 bg-violet-100 group-hover:bg-violet-200 rounded-2xl flex items-center justify-center mx-auto mb-5 transition-colors text-3xl">
              📄
            </div>
            <h3 className="text-lg font-bold text-slate-800 mb-2">PDF Upload</h3>
            <p className="text-slate-500 text-sm leading-relaxed">
              Upload an SRS document, project proposal, or RFP and get instant requirements + SDLC recommendation.
            </p>
            <div className="mt-5 text-violet-600 text-sm font-semibold group-hover:text-violet-700 flex items-center justify-center gap-1.5">
              Upload a PDF <span className="group-hover:translate-x-1 transition-transform">→</span>
            </div>
          </Link>
        </div>
      </section>

      {/* Pipeline Steps */}
      <section className="bg-white border-y border-slate-100 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-800 mb-3">How It Works</h2>
            <p className="text-slate-500 max-w-xl mx-auto">Four intelligent steps from project description to SDLC selection.</p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {steps.map((step, i) => (
              <div key={i} className={`border rounded-2xl p-6 ${step.color} relative`}>
                <div className="absolute -top-3 -left-3 w-7 h-7 bg-indigo-600 text-white text-xs font-bold rounded-full flex items-center justify-center shadow-sm">
                  {i + 1}
                </div>
                <div className={`w-12 h-12 ${step.iconBg} rounded-xl flex items-center justify-center text-2xl mb-4`}>
                  {step.icon}
                </div>
                <h3 className="font-bold text-slate-800 mb-2">{step.title}</h3>
                <p className="text-slate-500 text-sm leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-slate-800 mb-3">Built for Finance</h2>
          <p className="text-slate-500 max-w-xl mx-auto">Every feature is tailored for the unique demands of financial software projects.</p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f, i) => (
            <div key={i} className="card p-6 flex gap-4 items-start">
              <div className="text-2xl mt-0.5 flex-shrink-0">{f.icon}</div>
              <div>
                <h4 className="font-semibold text-slate-800 mb-1">{f.title}</h4>
                <p className="text-slate-500 text-sm leading-relaxed">{f.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Banner */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-3xl p-10 text-center text-white relative overflow-hidden">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute top-0 right-0 w-64 h-64 bg-white rounded-full -translate-x-1/4 -translate-y-1/4" />
            <div className="absolute bottom-0 left-0 w-64 h-64 bg-white rounded-full translate-x-1/4 translate-y-1/4" />
          </div>
          <div className="relative">
            <h2 className="text-3xl font-bold mb-3">Ready to find your SDLC match?</h2>
            <p className="text-indigo-100 mb-7 max-w-lg mx-auto">
              Run the AI pipeline on your finance project and get a data-driven SDLC recommendation in under a minute.
            </p>
            <Link
              to="/pipeline"
              className="inline-block bg-white text-indigo-700 font-bold px-8 py-3.5 rounded-xl hover:bg-indigo-50 transition-colors shadow-lg"
            >
              Launch Pipeline →
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
