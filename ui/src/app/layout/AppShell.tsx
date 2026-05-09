import { NavLink, Outlet } from 'react-router-dom'

import { useHealthQuery } from '../../api/hooks/useHealthQuery'
import { MobileBottomNav, type MobileBottomNavLink } from '../../components/ui/MobileBottomNav'
import { useTheme } from '../providers/theme-context'

const DESKTOP_NAV_LINKS = [
  { to: '/live', label: 'Live' },
  { to: '/events', label: 'Events' },
  { to: '/cameras', label: 'Cameras' },
  { to: '/settings', label: 'Settings' },
  { to: '/system', label: 'System' },
]

const MOBILE_NAV_LINKS: readonly MobileBottomNavLink[] = [
  { to: '/live', label: 'Live' },
  { to: '/events', label: 'Events' },
  { to: '/cameras', label: 'Cameras' },
  { to: '/settings', label: 'Settings' },
]

function navLinkClassName({ isActive }: { isActive: boolean }): string {
  return isActive ? 'nav-link nav-link--active' : 'nav-link'
}

function systemStatusText(status: string | undefined, isError: boolean): string {
  if (isError) {
    return 'System unavailable'
  }
  if (!status) {
    return 'Checking system'
  }
  if (status === 'healthy') {
    return 'System OK'
  }
  return `System ${status}`
}

export function AppShell() {
  const { theme, toggleTheme } = useTheme()
  const healthQuery = useHealthQuery()

  return (
    <div className="app-shell">
      <div className="app-shell__background" />
      <aside className="app-shell__sidebar">
        <p className="app-shell__brand">HomeSec</p>
        <h1 className="app-shell__title">Home security</h1>
        <nav className="app-shell__nav" aria-label="Primary">
          {DESKTOP_NAV_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} className={navLinkClassName}>
              {link.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <div className="app-shell__main">
        <header className="app-shell__header">
          <NavLink className="app-shell__header-status" to="/system">
            {systemStatusText(healthQuery.data?.status, healthQuery.isError)}
          </NavLink>
          <button type="button" className="button button--ghost" onClick={toggleTheme}>
            Theme: {theme === 'dark' ? 'Dark' : 'Light'}
          </button>
        </header>
        <main className="app-shell__content" role="main">
          <Outlet />
        </main>
      </div>

      <MobileBottomNav links={MOBILE_NAV_LINKS} />
    </div>
  )
}
