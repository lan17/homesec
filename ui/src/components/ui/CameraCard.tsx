import type { ReactNode } from 'react'

interface CameraCardMetaItem {
  label: string
  value: ReactNode
}

interface CameraCardProps {
  title: string
  subtitle?: ReactNode
  status?: ReactNode
  media?: ReactNode
  meta?: readonly CameraCardMetaItem[]
  actions?: ReactNode
  technicalDetails?: ReactNode
  className?: string
}

export function CameraCard({
  title,
  subtitle,
  status,
  media,
  meta,
  actions,
  technicalDetails,
  className,
}: CameraCardProps) {
  const classes = className ? `camera-card ${className}` : 'camera-card'

  return (
    <article className={classes}>
      <header className="camera-card__header">
        <div>
          <h2 className="camera-card__title">{title}</h2>
          {subtitle ? <p className="camera-card__subtitle">{subtitle}</p> : null}
        </div>
        {status ? <div className="camera-card__status">{status}</div> : null}
      </header>

      {media ? <div className="camera-card__media">{media}</div> : null}

      {meta && meta.length > 0 ? (
        <dl className="camera-card__meta">
          {meta.map((item) => (
            <div key={item.label} className="camera-card__meta-row">
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      ) : null}

      {technicalDetails ? <div className="camera-card__technical">{technicalDetails}</div> : null}
      {actions ? <div className="camera-card__actions">{actions}</div> : null}
    </article>
  )
}
