import { Link } from 'react-router-dom'

import { Card } from '../../components/ui/Card'

export function SettingsPage() {
  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Settings</h1>
          <p className="page__lead">Configure cameras, storage, detection, and notifications.</p>
        </div>
      </header>

      <div className="settings-grid">
        <Card title="Setup wizard" subtitle="Homeowner configuration">
          <p className="muted">
            Update camera setup, storage, detection, and notification choices with the existing
            guided flow.
          </p>
          <div className="inline-form__actions">
            <Link className="button button--primary" to="/setup">
              Open setup wizard
            </Link>
          </div>
        </Card>

        <Card title="Cameras" subtitle="Camera definitions">
          <p className="muted">Manage existing cameras and preview controls.</p>
          <div className="inline-form__actions">
            <Link className="button button--ghost" to="/cameras">
              Open Cameras
            </Link>
          </div>
        </Card>

        <Card title="Advanced" subtitle="System status and diagnostics">
          <p className="muted">
            Runtime health, database backups, and diagnostic details are available in System.
          </p>
          <div className="inline-form__actions">
            <Link className="button button--ghost" to="/system">
              Open System
            </Link>
          </div>
        </Card>
      </div>
    </section>
  )
}
