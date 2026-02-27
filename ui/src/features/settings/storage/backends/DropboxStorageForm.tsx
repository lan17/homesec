import type { StorageBackendFormProps } from '../types'
import { readString } from './configReaders'

export function DropboxStorageForm({ config, onChange }: StorageBackendFormProps) {
  const root = readString(config, 'root', '/homesec')
  const tokenEnv = readString(config, 'token_env', 'DROPBOX_TOKEN')

  return (
    <div className="inline-form">
      <label className="field-label" htmlFor="setup-storage-dropbox-root">
        Dropbox root path
        <input
          id="setup-storage-dropbox-root"
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

      <label className="field-label" htmlFor="setup-storage-dropbox-token-env">
        Dropbox token env var
        <input
          id="setup-storage-dropbox-token-env"
          className="input"
          type="text"
          value={tokenEnv}
          onChange={(event) => {
            onChange({
              ...config,
              token_env: event.target.value,
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
