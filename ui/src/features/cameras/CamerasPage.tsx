import { Card } from '../../components/ui/Card'

export function CamerasPage() {
  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <h1 className="page__title">Cameras</h1>
      </header>
      <Card title="Coming next" subtitle="Camera CRUD and source health">
        <p className="muted">
          This route is scaffolded. Next slices will integrate typed camera list/create/update calls.
        </p>
      </Card>
    </section>
  )
}
