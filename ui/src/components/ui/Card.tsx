import type { PropsWithChildren } from 'react'

interface CardProps extends PropsWithChildren {
  title: string
  subtitle?: string
}

export function Card({ title, subtitle, children }: CardProps) {
  return (
    <section className="card">
      <header className="card__header">
        <h2 className="card__title">{title}</h2>
        {subtitle ? <p className="card__subtitle">{subtitle}</p> : null}
      </header>
      <div className="card__body">{children}</div>
    </section>
  )
}
