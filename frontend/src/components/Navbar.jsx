import { NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'

const NavItem = ({ to, children }) => (
  <NavLink
    to={to}
    className={({ isActive }) =>
      `px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
        isActive
          ? 'bg-white/15 text-white'
          : 'text-indigo-200 hover:text-white hover:bg-white/10'
      }`
    }
  >
    {children}
  </NavLink>
)

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <nav className="bg-brand-950 border-b border-indigo-900/50 sticky top-0 z-50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 bg-indigo-500 rounded-lg flex items-center justify-center shadow-lg group-hover:bg-indigo-400 transition-colors">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <span className="text-white font-bold text-lg tracking-tight">SDLcraft</span>
              <div className="text-indigo-400 text-[10px] font-medium -mt-0.5 tracking-wide uppercase">Finance SDLC</div>
            </div>
          </NavLink>

          {/* Desktop Nav */}
          <div className="hidden sm:flex items-center gap-1">
            <NavItem to="/">Home</NavItem>
            <NavItem to="/pipeline">Pipeline</NavItem>
            <NavItem to="/catalogue">Catalogue</NavItem>
          </div>

          {/* CTA */}
          <div className="hidden sm:flex items-center gap-3">
            <div className="flex items-center gap-1.5 bg-emerald-500/20 text-emerald-300 text-xs px-3 py-1.5 rounded-full font-medium">
              <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse-slow"></span>
              Finance Ready
            </div>
            <NavLink
              to="/pipeline"
              className="bg-indigo-500 hover:bg-indigo-400 text-white text-sm font-semibold px-4 py-2 rounded-lg transition-all duration-200 shadow-sm"
            >
              Run Pipeline →
            </NavLink>
          </div>

          {/* Mobile menu */}
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="sm:hidden text-indigo-200 hover:text-white p-2 rounded-lg"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {menuOpen
                ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              }
            </svg>
          </button>
        </div>

        {/* Mobile Menu Dropdown */}
        {menuOpen && (
          <div className="sm:hidden pb-4 pt-2 flex flex-col gap-1">
            <NavItem to="/">Home</NavItem>
            <NavItem to="/pipeline">Pipeline</NavItem>
            <NavItem to="/catalogue">Catalogue</NavItem>
            <NavLink
              to="/pipeline"
              onClick={() => setMenuOpen(false)}
              className="mt-2 bg-indigo-500 text-white text-sm font-semibold px-4 py-2 rounded-lg text-center"
            >
              Run Pipeline →
            </NavLink>
          </div>
        )}
      </div>
    </nav>
  )
}
