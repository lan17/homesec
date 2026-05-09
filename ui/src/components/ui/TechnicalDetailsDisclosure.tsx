import type { PropsWithChildren, ReactNode } from 'react'

interface TechnicalDetailsDisclosureProps extends PropsWithChildren {
  summary?: ReactNode
  className?: string
}

export function TechnicalDetailsDisclosure({
  summary = 'Technical details',
  className,
  children,
}: TechnicalDetailsDisclosureProps) {
  const classes = className
    ? `technical-details ${className}`
    : 'technical-details'

  return (
    <details className={classes}>
      <summary className="technical-details__summary">{summary}</summary>
      <div className="technical-details__body">{children}</div>
    </details>
  )
}
