import { Link } from 'react-router-dom'

import { Card } from '../../components/ui/Card'

const SETTINGS_SECTIONS = [
  {
    title: 'Cameras',
    subtitle: 'Camera setup',
    description: 'Add cameras or change camera configuration. Daily camera use stays on Live.',
    to: '/settings/cameras',
    action: 'Camera setup',
    primary: true,
  },
  {
    title: 'Notifications',
    subtitle: 'Alert destinations',
    description: 'Edit configured alert destinations and alert policy.',
    to: '/settings/notifications',
    action: 'Update notifications',
    primary: false,
  },
  {
    title: 'Detection',
    subtitle: 'What HomeSec watches for',
    description: 'Edit object detection and AI analysis settings.',
    to: '/settings/detection',
    action: 'Update detection',
    primary: false,
  },
  {
    title: 'Storage',
    subtitle: 'Where event video is saved',
    description: 'Edit the configured storage backend and local working paths.',
    to: '/settings/storage',
    action: 'Update storage',
    primary: false,
  },
  {
    title: 'Advanced',
    subtitle: 'System status and diagnostics',
    description: 'Runtime health, backups, reload controls, and diagnostics live in System.',
    to: '/system',
    action: 'Open System',
    primary: false,
  },
] as const

export function SettingsPage() {
  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Settings</h1>
          <p className="page__lead">
            Configure cameras, alerts, detection, and storage. Daily camera controls live on Live.
          </p>
        </div>
      </header>

      <div className="settings-grid">
        {SETTINGS_SECTIONS.map((section) => (
          <Card key={section.title} title={section.title} subtitle={section.subtitle}>
            <p className="muted">{section.description}</p>
            <div className="inline-form__actions">
              <Link
                className={section.primary ? 'button button--primary' : 'button button--ghost'}
                to={section.to}
              >
                {section.action}
              </Link>
            </div>
          </Card>
        ))}
      </div>
    </section>
  )
}
