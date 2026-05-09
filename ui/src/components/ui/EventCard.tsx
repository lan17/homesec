import type { ReactNode } from 'react'

interface EventCardMetaItem {
  label: string
  value: ReactNode
}

interface EventCardProps {
  camera: ReactNode
  time: ReactNode
  title?: ReactNode
  summary?: ReactNode
  media?: ReactNode
  risk?: ReactNode
  status?: ReactNode
  meta?: readonly EventCardMetaItem[]
  actions?: ReactNode
  technicalDetails?: ReactNode
  className?: string
}

export function EventCard({
  camera,
  time,
  title,
  summary,
  media,
  risk,
  status,
  meta,
  actions,
  technicalDetails,
  className,
}: EventCardProps) {
  const classes = className ? `event-card ${className}` : 'event-card'

  return (
    <article className={classes}>
      {media ? <div className="event-card__media">{media}</div> : null}
      <div className="event-card__content">
        <header className="event-card__header">
          <div>
            <p className="event-card__camera">{camera}</p>
            <p className="event-card__time">{time}</p>
          </div>
          {risk || status ? (
            <div className="event-card__badges">
              {risk}
              {status}
            </div>
          ) : null}
        </header>

        {title ? <h2 className="event-card__title">{title}</h2> : null}
        {summary ? <p className="event-card__summary">{summary}</p> : null}

        {meta && meta.length > 0 ? (
          <dl className="event-card__meta">
            {meta.map((item) => (
              <div key={item.label} className="event-card__meta-row">
                <dt>{item.label}</dt>
                <dd>{item.value}</dd>
              </div>
            ))}
          </dl>
        ) : null}

        {technicalDetails ? <div className="event-card__technical">{technicalDetails}</div> : null}
        {actions ? <div className="event-card__actions">{actions}</div> : null}
      </div>
    </article>
  )
}
