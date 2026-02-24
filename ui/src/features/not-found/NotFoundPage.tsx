import { Link } from 'react-router-dom'

export function NotFoundPage() {
  return (
    <section className="page fade-in-up">
      <h1 className="page__title">Route not found</h1>
      <p className="page__lead">
        The requested path does not exist. Go back to <Link to="/">dashboard</Link>.
      </p>
    </section>
  )
}
