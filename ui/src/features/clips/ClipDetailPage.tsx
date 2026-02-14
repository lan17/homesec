import { Link, useParams } from 'react-router-dom'

import { Card } from '../../components/ui/Card'

export function ClipDetailPage() {
  const { clipId } = useParams<{ clipId: string }>()

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Clip Detail</h1>
          <p className="page__lead">Clip ID: {clipId ?? 'unknown'}</p>
        </div>
      </header>
      <Card title="Playback + metadata">
        <p className="muted">
          This page will load clip metadata and storage-backed playback URL using typed API hooks.
        </p>
        <p className="muted">
          Return to <Link to="/clips">clips list</Link>.
        </p>
      </Card>
    </section>
  )
}
