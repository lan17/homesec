import type { PropsWithChildren, ReactNode } from 'react'

interface ResponsivePageShellProps extends PropsWithChildren {
  title: string
  lead?: ReactNode
  eyebrow?: ReactNode
  actions?: ReactNode
  className?: string
}

export function ResponsivePageShell({
  title,
  lead,
  eyebrow,
  actions,
  className,
  children,
}: ResponsivePageShellProps) {
  const classes = className
    ? `page responsive-page-shell fade-in-up ${className}`
    : 'page responsive-page-shell fade-in-up'

  return (
    <section className={classes}>
      <header className="page__header responsive-page-shell__header">
        <div>
          {eyebrow ? <p className="responsive-page-shell__eyebrow">{eyebrow}</p> : null}
          <h1 className="page__title">{title}</h1>
          {lead ? <p className="page__lead">{lead}</p> : null}
        </div>
        {actions ? <div className="responsive-page-shell__actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  )
}
