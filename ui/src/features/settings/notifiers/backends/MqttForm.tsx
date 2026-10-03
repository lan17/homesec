import type { NotifierBackendFormProps } from '../types'
import { readNumber, readString } from './configReaders'
import { CredentialField } from '../../CredentialField'

export function MqttForm({ config, onChange, idPrefix = 'setup-notifier-mqtt', credentials }: NotifierBackendFormProps) {
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
      <label className="field-label" htmlFor={`${idPrefix}-host`}>
        MQTT host
        <input
          id={`${idPrefix}-host`}
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

      <label className="field-label" htmlFor={`${idPrefix}-port`}>
        MQTT port
        <input
          id={`${idPrefix}-port`}
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
            const parsed = Number(raw)
            onChange({
              ...config,
              port: Number.isFinite(parsed) ? parsed : port,
            })
          }}
        />
      </label>

      <label className="field-label" htmlFor={`${idPrefix}-topic-template`}>
        Topic template
        <input
          id={`${idPrefix}-topic-template`}
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

      {credentials ? (
        <>
          <CredentialField id={`${idPrefix}-username`} label="MQTT username" envKey="auth.username_env"
            envValue={usernameEnv} credentials={credentials}
            onEnvChange={(value) => { onChange({ ...config, auth: { ...auth, username_env: value } }) }} />
          <CredentialField id={`${idPrefix}-password`} label="MQTT password" envKey="auth.password_env"
            envValue={passwordEnv} credentials={credentials}
            onEnvChange={(value) => { onChange({ ...config, auth: { ...auth, password_env: value } }) }} />
        </>
      ) : <><label className="field-label" htmlFor={`${idPrefix}-username-env`}>
        Username env var (optional)
        <input
          id={`${idPrefix}-username-env`}
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

      <label className="field-label" htmlFor={`${idPrefix}-password-env`}>
        Password env var (optional)
        <input
          id={`${idPrefix}-password-env`}
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
      </>}
    </div>
  )
}
