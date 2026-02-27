import { NOTIFIER_BACKENDS, NOTIFIER_BACKEND_ORDER } from './backends'
import type { NotifierBackend, NotifierFormState } from './types'

interface NotifierConfigFormProps {
  value: NotifierFormState
  onChange: (value: NotifierFormState) => void
}

export function NotifierConfigForm({ value, onChange }: NotifierConfigFormProps) {
  function updateBackendEnabled(backend: NotifierBackend, enabled: boolean): void {
    const defaults = NOTIFIER_BACKENDS[backend].defaultConfig
    onChange({
      ...value,
      [backend]: {
        enabled,
        config: enabled ? (value[backend].config ?? defaults) : value[backend].config,
      },
    })
  }

  function updateBackendConfig(backend: NotifierBackend, config: Record<string, unknown>): void {
    onChange({
      ...value,
      [backend]: {
        ...value[backend],
        config,
      },
    })
  }

  return (
    <section className="inline-form">
      <h3 className="backend-picker__title">Notifier backends</h3>
      <p className="subtle">Enable one or more channels for alert delivery.</p>

      <div className="notifier-config__cards">
        {NOTIFIER_BACKEND_ORDER.map((backendId) => {
          const backend = NOTIFIER_BACKENDS[backendId]
          const state = value[backendId]
          const BackendComponent = backend.component
          return (
            <section key={backendId} className="notifier-config__card">
              <label className="field-label form-checkbox-field">
                <input
                  type="checkbox"
                  checked={state.enabled}
                  onChange={(event) => {
                    updateBackendEnabled(backendId, event.target.checked)
                  }}
                />
                {backend.label}
              </label>
              <p className="subtle">{backend.description}</p>
              {state.enabled ? (
                <BackendComponent
                  config={state.config}
                  onChange={(nextConfig) => {
                    updateBackendConfig(backendId, nextConfig)
                  }}
                />
              ) : null}
            </section>
          )
        })}
      </div>
    </section>
  )
}
