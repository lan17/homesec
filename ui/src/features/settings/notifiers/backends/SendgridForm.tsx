import type { NotifierBackendFormProps } from '../types'
import { readString } from './configReaders'

function joinEmails(value: unknown): string {
  if (!Array.isArray(value)) {
    return ''
  }
  return value
    .filter((item): item is string => typeof item === 'string')
    .join(', ')
}

function parseEmails(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
}

export function SendgridForm({ config, onChange }: NotifierBackendFormProps) {
  const fromEmail = readString(config, 'from_email', 'homesec@localhost')
  const toEmails = joinEmails(config.to_emails)
  const apiKeyEnv = readString(config, 'api_key_env', 'SENDGRID_API_KEY')

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="setup-notifier-sendgrid-from-email">
        From email
        <input
          id="setup-notifier-sendgrid-from-email"
          className="input"
          type="email"
          value={fromEmail}
          onChange={(event) => {
            onChange({
              ...config,
              from_email: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="setup-notifier-sendgrid-to-emails">
        Recipient emails (comma separated)
        <input
          id="setup-notifier-sendgrid-to-emails"
          className="input"
          type="text"
          value={toEmails}
          onChange={(event) => {
            onChange({
              ...config,
              to_emails: parseEmails(event.target.value),
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="setup-notifier-sendgrid-api-key-env">
        SendGrid API key env var
        <input
          id="setup-notifier-sendgrid-api-key-env"
          className="input"
          type="text"
          value={apiKeyEnv}
          onChange={(event) => {
            onChange({
              ...config,
              api_key_env: event.target.value,
            })
          }}
        />
      </label>
      <p className="subtle">
        Set this environment variable on the host before launch.
      </p>
    </div>
  )
}
