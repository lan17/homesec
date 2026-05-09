import { NavLink } from 'react-router-dom'

export interface MobileBottomNavLink {
  to: string
  label: string
  end?: boolean
}

interface MobileBottomNavProps {
  links: readonly MobileBottomNavLink[]
  ariaLabel?: string
}

function mobileNavLinkClassName({ isActive }: { isActive: boolean }): string {
  return isActive ? 'mobile-nav-link mobile-nav-link--active' : 'mobile-nav-link'
}

export function MobileBottomNav({
  links,
  ariaLabel = 'Mobile primary',
}: MobileBottomNavProps) {
  return (
    <nav className="mobile-bottom-nav" aria-label={ariaLabel}>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          end={link.end}
          className={mobileNavLinkClassName}
        >
          {link.label}
        </NavLink>
      ))}
    </nav>
  )
}
