import { Link } from 'react-router-dom'

import { Card } from '../../components/ui/Card'

export function ClipsPage() {
  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Clips</h1>
          <p className="page__lead">Cursor pagination, filters, and playback surface.</p>
        </div>
      </header>

      <div className="grid grid--cards">
        <Card title="Filter contract">
          <p className="muted">Status, activity type, risk level, and date range filters will live here.</p>
        </Card>
        <Card title="Pagination contract">
          <p className="muted">
            Keyset cursor paging wired to API response cursors. URL params will remain source-of-truth.
          </p>
        </Card>
        <Card title="Clip detail route">
          <p className="muted">
            Placeholder detail page is available at <Link to="/clips/example-clip">/clips/example-clip</Link>.
          </p>
        </Card>
      </div>
    </section>
  )
}
