import { Link } from 'react-router-dom'

import { Card } from '../../components/ui/Card'
import { isIOSNativeApp } from '../../runtime/nativeRuntime'

const NATIVE_CONNECTION_SECTION = {
  title: 'Connection',
  subtitle: 'HomeSec server',
  description: 'Change the server this app connects to or update its API token.',
  to: '/native-setup',
  action: 'Change server',
  primary: false,
} as const

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
  const sections = isIOSNativeApp()
    ? [NATIVE_CONNECTION_SECTION, ...SETTINGS_SECTIONS]
    : SETTINGS_SECTIONS

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
        {sections.map((section) => (
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
