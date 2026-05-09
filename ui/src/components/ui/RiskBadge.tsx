import type { HTMLAttributes } from 'react'

import { riskToneForLevel } from './riskTone'

interface RiskBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  level: string | null | undefined
}

function riskLabel(level: string | null | undefined): string {
  const trimmed = level?.trim()
  return trimmed ? trimmed : 'Risk unavailable'
}

export function RiskBadge({ level, className, role, ...props }: RiskBadgeProps) {
  const tone = riskToneForLevel(level)
  const classes = className
    ? `risk-badge risk-badge--${tone} ${className}`
    : `risk-badge risk-badge--${tone}`

  return (
    <span className={classes} role={role ?? 'status'} {...props}>
      {riskLabel(level)}
    </span>
  )
}
