import { Button } from '../../../components/ui/Button'
import type { DiscoveredCameraResponse } from '../../../api/generated/types'

export interface OnvifProbeCredentials {
  username: string
  password: string
  port: number
}

interface OnvifProbeStepProps {
  camera: DiscoveredCameraResponse
  credentials: OnvifProbeCredentials
  isProbing: boolean
  error: string | null
  onCredentialsChange: (next: OnvifProbeCredentials) => void
  onProbe: () => void
  onBack: () => void
  onCancel: () => void
}

export function OnvifProbeStep({
  camera,
  credentials,
  isProbing,
  error,
  onCredentialsChange,
  onProbe,
  onBack,
  onCancel,
}: OnvifProbeStepProps) {
  return (
    <form
      className="onvif-wizard__step inline-form"
      onSubmit={(event) => {
        event.preventDefault()
        onProbe()
      }}
    >
      <div className="onvif-step-header">
        <h3 className="onvif-step-title">Step 2: Authenticate and probe</h3>
        <div className="inline-form__actions">
          <Button variant="ghost" onClick={onBack} disabled={isProbing}>
            Back
          </Button>
          <Button variant="ghost" onClick={onCancel} disabled={isProbing}>
            Cancel
          </Button>
        </div>
      </div>

      <p className="muted">
        Selected camera: <span className="camera-mono">{camera.ip}</span>
      </p>

      <div className="camera-form-grid">
        <label className="field-label" htmlFor="onvif-username">
          ONVIF username
          <input
            id="onvif-username"
            className="input"
            type="text"
            value={credentials.username}
            onChange={(event) => {
              onCredentialsChange({ ...credentials, username: event.target.value })
            }}
            disabled={isProbing}
            autoComplete="username"
          />
        </label>
        <label className="field-label" htmlFor="onvif-password">
          ONVIF password
          <input
            id="onvif-password"
            className="input"
            type="password"
            value={credentials.password}
            onChange={(event) => {
              onCredentialsChange({ ...credentials, password: event.target.value })
            }}
            disabled={isProbing}
            autoComplete="current-password"
          />
        </label>
        <label className="field-label" htmlFor="onvif-port">
          ONVIF port
          <input
            id="onvif-port"
            className="input"
            type="number"
            value={credentials.port}
            min={1}
            max={65535}
            onChange={(event) => {
              const parsed = Number.parseInt(event.target.value, 10)
              if (Number.isNaN(parsed)) {
                return
              }
              onCredentialsChange({
                ...credentials,
                port: parsed,
              })
            }}
            disabled={isProbing}
          />
        </label>
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      <div className="inline-form__actions">
        <Button type="submit" disabled={isProbing}>
          {isProbing ? 'Probing...' : 'Probe camera'}
        </Button>
      </div>
    </form>
  )
}
