import type { HTMLAttributes, ReactNode } from 'react'

export type StatusBadgeTone = 'healthy' | 'degraded' | 'unhealthy' | 'unknown'

interface StatusBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone: StatusBadgeTone
  children: ReactNode
}

export function StatusBadge({ tone, children, className, role, ...props }: StatusBadgeProps) {
  const classes = className
    ? `status-badge status-badge--${tone} ${className}`
    : `status-badge status-badge--${tone}`

  return (
    <span className={classes} role={role ?? 'status'} {...props}>
      {children}
    </span>
  )
}
