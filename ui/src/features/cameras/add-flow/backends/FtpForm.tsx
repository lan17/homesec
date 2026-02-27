import type { BackendFormStepProps } from './types'
import { readBoolean, readNumber, readString } from './configReaders'

export function FtpForm({ config, onChange }: BackendFormStepProps) {
  const host = readString(config, 'host', '0.0.0.0')
  const port = readNumber(config, 'port', 2121)
  const rootDir = readString(config, 'root_dir', './ftp_incoming')
  const anonymous = readBoolean(config, 'anonymous', true)

  return (
    <div className="inline-form">
      <div className="camera-form-grid">
        <label className="field-label" htmlFor="camera-ftp-host">
          Bind host
          <input
            id="camera-ftp-host"
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

        <label className="field-label" htmlFor="camera-ftp-port">
          Bind port
          <input
            id="camera-ftp-port"
            className="input"
            type="number"
            min={1}
            max={65535}
            value={port}
            onChange={(event) => {
              const nextPort = Number.parseInt(event.target.value, 10)
              if (Number.isNaN(nextPort)) {
                return
              }
              onChange({
                ...config,
                port: nextPort,
              })
            }}
          />
        </label>
      </div>

      <label className="field-label" htmlFor="camera-ftp-root-dir">
        Root directory
        <input
          id="camera-ftp-root-dir"
          className="input"
          type="text"
          value={rootDir}
          onChange={(event) => {
            onChange({
              ...config,
              root_dir: event.target.value,
            })
          }}
        />
      </label>

      <label className="field-label camera-checkbox-field" htmlFor="camera-ftp-anonymous">
        <input
          id="camera-ftp-anonymous"
          type="checkbox"
          checked={anonymous}
          onChange={(event) => {
            onChange({
              ...config,
              anonymous: event.target.checked,
            })
          }}
        />
        Allow anonymous uploads
      </label>
    </div>
  )
}
