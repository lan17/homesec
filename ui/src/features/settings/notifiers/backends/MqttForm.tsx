import type { NotifierBackendFormProps } from '../types'
import { readNumber, readString } from './configReaders'

export function MqttForm({ config, onChange }: NotifierBackendFormProps) {
  const host = readString(config, 'host', 'localhost')
  const port = readNumber(config, 'port', 1883)
  const topicTemplate = readString(
    config,
    'topic_template',
    'homecam/alerts/{camera_name}',
  )
  const auth =
    config.auth && typeof config.auth === 'object' && !Array.isArray(config.auth)
      ? config.auth
      : {}
  const usernameEnv = readString(auth as Record<string, unknown>, 'username_env', '')
  const passwordEnv = readString(auth as Record<string, unknown>, 'password_env', '')

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="setup-notifier-mqtt-host">
        MQTT host
        <input
          id="setup-notifier-mqtt-host"
          className="input"
          type="text"
          value={host}
          onChange={(event) => {
            onChange({
              ...config,
              host: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="setup-notifier-mqtt-port">
        MQTT port
        <input
          id="setup-notifier-mqtt-port"
          className="input"
          type="number"
          min={1}
          max={65535}
          value={port}
          onChange={(event) => {
            const raw = event.target.value
            if (raw === '') {
              onChange({
                ...config,
                port: 1883,
              })
              return
            }
            const parsed = Number.parseInt(raw, 10)
            onChange({
              ...config,
              port: Number.isFinite(parsed) ? parsed : port,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="setup-notifier-mqtt-topic-template">
        Topic template
        <input
          id="setup-notifier-mqtt-topic-template"
          className="input"
          type="text"
          value={topicTemplate}
          onChange={(event) => {
            onChange({
              ...config,
              topic_template: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="setup-notifier-mqtt-username-env">
        Username env var (optional)
        <input
          id="setup-notifier-mqtt-username-env"
          className="input"
          type="text"
          value={usernameEnv}
          onChange={(event) => {
            onChange({
              ...config,
              auth: {
                ...auth,
                username_env: event.target.value,
              },
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor="setup-notifier-mqtt-password-env">
        Password env var (optional)
        <input
          id="setup-notifier-mqtt-password-env"
          className="input"
          type="text"
          value={passwordEnv}
          onChange={(event) => {
            onChange({
              ...config,
              auth: {
                ...auth,
                password_env: event.target.value,
              },
            })
          }}
        />
      </label>
    </div>
  )
}
