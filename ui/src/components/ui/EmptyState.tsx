import type { ReactNode } from 'react'

type EmptyStateTone = 'neutral' | 'loading' | 'error'

interface EmptyStateProps {
  title: string
  description?: ReactNode
  action?: ReactNode
  tone?: EmptyStateTone
}

export function EmptyState({
  title,
  description,
  action,
  tone = 'neutral',
}: EmptyStateProps) {
  return (
    <div
      className={`empty-state empty-state--${tone}`}
      role={tone === 'error' ? 'alert' : undefined}
      aria-live={tone === 'loading' ? 'polite' : undefined}
    >
      <h2 className="empty-state__title">{title}</h2>
      {description ? <p className="empty-state__description">{description}</p> : null}
      {action ? <div className="empty-state__action">{action}</div> : null}
    </div>
  )
}
