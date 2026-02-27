import type { StorageBackendFormProps } from '../types'
import { readString } from './configReaders'

export function LocalStorageForm({ config, onChange }: StorageBackendFormProps) {
  const root = readString(config, 'root', './storage')

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="setup-storage-local-root">
        Storage root directory
        <input
          id="setup-storage-local-root"
          className="input"
          type="text"
          value={root}
          onChange={(event) => {
            onChange({
              ...config,
              root: event.target.value,
            })
          }}
        />
      </label>
      <p className="subtle">
        HomeSec writes uploaded clips under this local filesystem path.
      </p>
    </div>
  )
}
