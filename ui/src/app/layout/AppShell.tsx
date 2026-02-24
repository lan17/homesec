import { NavLink, Outlet } from 'react-router-dom'

import { useTheme } from '../providers/theme-context'

const NAV_LINKS = [
  { to: '/', label: 'Dashboard' },
  { to: '/cameras', label: 'Cameras' },
  { to: '/clips', label: 'Clips' },
]

function navLinkClassName({ isActive }: { isActive: boolean }): string {
  return isActive ? 'nav-link nav-link--active' : 'nav-link'
}

export function AppShell() {
  const { theme, toggleTheme } = useTheme()

  return (
    <div className="app-shell">
      <div className="app-shell__background" />
      <aside className="app-shell__sidebar">
        <p className="app-shell__brand">HOMESEC</p>
        <h1 className="app-shell__title">Control Plane</h1>
        <nav className="app-shell__nav" aria-label="Primary">
          {NAV_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} end={link.to === '/'} className={navLinkClassName}>
              {link.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="app-shell__main">
        <header className="app-shell__header">
          <p className="app-shell__header-text">MVP self-serve</p>
          <button type="button" className="button button--ghost" onClick={toggleTheme}>
            Theme: {theme === 'dark' ? 'Dark' : 'Light'}
          </button>
        </header>
        <main className="app-shell__content" role="main">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
