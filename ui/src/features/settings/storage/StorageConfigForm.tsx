import type { StorageBackendsResponse } from '../../../api/generated/types'
import { STORAGE_BACKENDS, STORAGE_BACKEND_ORDER } from './backends'
import {
  buildSupportedStorageBackendOptions,
  defaultConfigForBackend,
} from './editorModel'
import type { StorageBackend, StorageFormState } from './types'

interface StorageConfigFormProps {
  value: StorageFormState
  backends?: StorageBackendsResponse | null
  onChange: (value: StorageFormState) => void
}

export function StorageConfigForm({ value, backends = null, onChange }: StorageConfigFormProps) {
  const backendDef = STORAGE_BACKENDS[value.backend]
  const BackendComponent = backendDef.component
  const backendOptions = buildSupportedStorageBackendOptions(backends)

  function handleSelectBackend(backend: StorageBackend): void {
    const selected = backendOptions.find((option) => option.backend === backend)
    onChange({
      backend,
      config: defaultConfigForBackend(backend, selected?.metadata ?? null),
    })
  }

  return (
    <section className="inline-form">
      <h3 className="backend-picker__title">Configure storage backend</h3>
      <p className="subtle">Choose where clips are uploaded after capture.</p>

      <div className="backend-picker__grid">
        {STORAGE_BACKEND_ORDER.map((backendId) => {
          const backend =
            backendOptions.find((option) => option.backend === backendId)
            ?? {
              backend: backendId,
              label: STORAGE_BACKENDS[backendId].label,
              description: STORAGE_BACKENDS[backendId].description,
              metadata: null,
            }
          const selected = backendId === value.backend
          return (
            <button
              key={backend.backend}
              type="button"
              className="backend-picker__card"
              aria-label={backend.label}
              aria-pressed={selected}
              onClick={() => {
                handleSelectBackend(backend.backend)
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
