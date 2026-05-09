import { Link } from 'react-router-dom'

import { Card } from '../../components/ui/Card'

const SETTINGS_SECTIONS = [
  {
    title: 'Cameras',
    subtitle: 'Camera setup and controls',
    description: 'Add cameras, enable or disable existing cameras, and open camera controls.',
    to: '/cameras',
    action: 'Manage cameras',
    primary: true,
  },
  {
    title: 'Notifications',
    subtitle: 'Alert destinations',
    description: 'Update where HomeSec sends alerts using the existing guided setup flow.',
    to: '/setup',
    action: 'Update notifications',
    primary: false,
  },
  {
    title: 'Detection',
    subtitle: 'What HomeSec watches for',
    description: 'Adjust object detection, AI summaries, and alert sensitivity in guided setup.',
    to: '/setup',
    action: 'Update detection',
    primary: false,
  },
  {
    title: 'Storage',
    subtitle: 'Where event video is saved',
    description: 'Choose or revise clip storage settings without entering system diagnostics.',
    to: '/setup',
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
            Manage cameras, alerts, detection, and storage. System details stay under Advanced.
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
