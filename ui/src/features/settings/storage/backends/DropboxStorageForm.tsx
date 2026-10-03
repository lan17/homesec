import type { StorageBackendFormProps } from '../types'
import { readString } from './configReaders'
import { CredentialField } from '../../CredentialField'

export function DropboxStorageForm({ config, onChange, credentials }: StorageBackendFormProps) {
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

      {credentials ? (
        <>
          <CredentialField id="storage-dropbox-token" label="Dropbox token" envKey="token_env"
            envValue={tokenEnv} credentials={credentials}
            onEnvChange={(value) => { onChange({ ...config, token_env: value }) }} />
          <details>
            <summary>Dropbox refresh-token credentials</summary>
            <p className="subtle">A configured access token takes precedence. To use refresh-token authentication, clear the access token and configure all three values below.</p>
            {[
              ['app_key_env', 'Dropbox app key', 'DROPBOX_APP_KEY'],
              ['app_secret_env', 'Dropbox app secret', 'DROPBOX_APP_SECRET'],
              ['refresh_token_env', 'Dropbox refresh token', 'DROPBOX_REFRESH_TOKEN'],
            ].map(([key, label, fallback]) => (
              <CredentialField key={key} id={`storage-dropbox-${key}`} label={label} envKey={key}
                envValue={readString(config, key, fallback)} credentials={credentials}
                onEnvChange={(value) => { onChange({ ...config, [key]: value }) }} />
            ))}
          </details>
        </>
      ) : <><label className="field-label" htmlFor="setup-storage-dropbox-token-env">
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
      </>}
    </div>
  )
}
