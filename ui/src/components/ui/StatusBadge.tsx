type BadgeTone = 'healthy' | 'degraded' | 'unhealthy' | 'unknown'

interface StatusBadgeProps {
  tone: BadgeTone
  children: string
}

export function StatusBadge({ tone, children }: StatusBadgeProps) {
  return (
    <span className={`status-badge status-badge--${tone}`} role="status">
      {children}
    </span>
  )
}
