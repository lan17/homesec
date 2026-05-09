import type { HTMLAttributes } from 'react'

import { riskLabelForLevel, riskToneForLevel } from './riskTone'

interface RiskBadgeProps extends HTMLAttributes<HTMLSpanElement> {
  level: string | null | undefined
}

export function RiskBadge({ level, className, role, ...props }: RiskBadgeProps) {
  const tone = riskToneForLevel(level)
  const classes = className
    ? `risk-badge risk-badge--${tone} ${className}`
    : `risk-badge risk-badge--${tone}`

  return (
    <span className={classes} role={role ?? 'status'} {...props}>
      {riskLabelForLevel(level)}
    </span>
  )
}
