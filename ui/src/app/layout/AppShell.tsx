import { NavLink, Outlet } from 'react-router-dom'

import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { useHealthQuery } from '../../api/hooks/useHealthQuery'
import { MobileBottomNav, type MobileBottomNavLink } from '../../components/ui/MobileBottomNav'
import { cameraIssueSummary } from '../../features/cameras/cameraHealth'
import { useTheme } from '../providers/theme-context'

const DESKTOP_NAV_LINKS = [
  { to: '/live', label: 'Live' },
  { to: '/events', label: 'Events' },
  { to: '/settings', label: 'Settings' },
  { to: '/system', label: 'System' },
]

const MOBILE_NAV_LINKS: readonly MobileBottomNavLink[] = [
  { to: '/live', label: 'Live' },
  { to: '/events', label: 'Events' },
  { to: '/settings', label: 'Settings' },
]

function navLinkClassName({ isActive }: { isActive: boolean }): string {
  return isActive ? 'nav-link nav-link--active' : 'nav-link'
}

function systemStatusText(status: string | undefined, isError: boolean): string {
  if (isError) {
    return 'System needs attention'
  }
  if (!status) {
    return 'Checking'
  }
  if (status === 'healthy') {
    return 'All systems normal'
  }
  return `System ${status}`
}

export function AppShell() {
  const { theme, toggleTheme } = useTheme()
  const healthQuery = useHealthQuery()
  const camerasQuery = useCamerasQuery()
  const cameraIssue = cameraIssueSummary(camerasQuery.data)
  const systemStatus = cameraIssue ?? systemStatusText(healthQuery.data?.status, healthQuery.isError)
  const systemStatusClassName = !cameraIssue && !healthQuery.isError && healthQuery.data?.status === 'healthy'
    ? 'app-shell__header-status app-shell__header-status--nominal'
    : 'app-shell__header-status'

  return (
    <div className="app-shell">
      <div className="app-shell__background" />
      <header className="app-shell__topbar">
        <NavLink className="app-shell__brand-link" to="/live" aria-label="HomeSec Live">
          <p className="app-shell__brand">HomeSec</p>
          <p className="app-shell__title">Home security</p>
        </NavLink>
        <nav className="app-shell__nav" aria-label="Primary">
          {DESKTOP_NAV_LINKS.map((link) => (
            <NavLink key={link.to} to={link.to} className={navLinkClassName}>
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="app-shell__header">
          <NavLink className={systemStatusClassName} to="/system">
            {systemStatus}
          </NavLink>
          <button
            type="button"
            className="button button--ghost app-shell__theme-button"
            onClick={toggleTheme}
          >
            {theme === 'dark' ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      <main className="app-shell__content" role="main">
        <Outlet />
      </main>

      <MobileBottomNav links={MOBILE_NAV_LINKS} />
    </div>
  )
}
