import type { PropsWithChildren, ReactNode } from 'react'

type MediaPanelAspect = 'video' | 'square' | 'auto'

interface MediaPanelProps extends PropsWithChildren {
  title?: string
  subtitle?: ReactNode
  status?: ReactNode
  actions?: ReactNode
  placeholder?: ReactNode
  aspect?: MediaPanelAspect
  className?: string
}

export function MediaPanel({
  title,
  subtitle,
  status,
  actions,
  placeholder,
  aspect = 'video',
  className,
  children,
}: MediaPanelProps) {
  const classes = className ? `media-panel ${className}` : 'media-panel'
  const viewportClasses = `media-panel__viewport media-panel__viewport--${aspect}`
  const hasHeader = title || subtitle || status || actions

  return (
    <section className={classes}>
      {hasHeader ? (
        <header className="media-panel__header">
          <div>
            {title ? <h2 className="media-panel__title">{title}</h2> : null}
            {subtitle ? <p className="media-panel__subtitle">{subtitle}</p> : null}
          </div>
          {status || actions ? (
            <div className="media-panel__header-actions">
              {status}
              {actions}
            </div>
          ) : null}
        </header>
      ) : null}

      <div className={viewportClasses}>
        {children ?? <div className="media-panel__placeholder">{placeholder ?? 'Media unavailable'}</div>}
      </div>
    </section>
  )
}
