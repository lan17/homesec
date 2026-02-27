import { STORAGE_BACKENDS, STORAGE_BACKEND_ORDER } from './backends'
import type { StorageBackend, StorageFormState } from './types'

interface StorageConfigFormProps {
  value: StorageFormState
  onChange: (value: StorageFormState) => void
}

export function StorageConfigForm({ value, onChange }: StorageConfigFormProps) {
  const backendDef = STORAGE_BACKENDS[value.backend]
  const BackendComponent = backendDef.component

  function handleSelectBackend(backend: StorageBackend): void {
    const selected = STORAGE_BACKENDS[backend]
    onChange({
      backend,
      config: selected.defaultConfig,
    })
  }

  return (
    <section className="inline-form">
      <h3 className="backend-picker__title">Configure storage backend</h3>
      <p className="subtle">Choose where clips are uploaded after capture.</p>

      <div className="backend-picker__grid">
        {STORAGE_BACKEND_ORDER.map((backendId) => {
          const backend = STORAGE_BACKENDS[backendId]
          const selected = backendId === value.backend
          return (
            <button
              key={backend.id}
              type="button"
              className="backend-picker__card"
              aria-label={backend.label}
              aria-pressed={selected}
              onClick={() => {
                handleSelectBackend(backend.id)
              }}
            >
              <p className="backend-picker__title-text">
                {backend.label}
                {selected ? ' (selected)' : ''}
              </p>
              <p className="backend-picker__description">{backend.description}</p>
            </button>
          )
        })}
      </div>

      <BackendComponent
        config={value.config}
        onChange={(nextConfig) => {
          onChange({
            ...value,
            config: nextConfig,
          })
        }}
      />
    </section>
  )
}
