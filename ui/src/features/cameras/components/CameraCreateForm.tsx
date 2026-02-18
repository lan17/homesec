import type { FormEvent } from 'react'

import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { CAMERA_BACKEND_OPTIONS, type CameraBackend } from '../forms'

interface CameraCreateFormProps {
  cameraName: string
  cameraEnabled: boolean
  cameraBackend: CameraBackend
  sourceConfigRaw: string
  createFormError: string | null
  isMutating: boolean
  createPending: boolean
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onCameraNameChange: (value: string) => void
  onCameraEnabledChange: (value: boolean) => void
  onCameraBackendChange: (value: CameraBackend) => void
  onSourceConfigChange: (value: string) => void
  onResetTemplate: () => void
}

export function CameraCreateForm({
  cameraName,
  cameraEnabled,
  cameraBackend,
  sourceConfigRaw,
  createFormError,
  isMutating,
  createPending,
  onSubmit,
  onCameraNameChange,
  onCameraEnabledChange,
  onCameraBackendChange,
  onSourceConfigChange,
  onResetTemplate,
}: CameraCreateFormProps) {
  return (
    <Card title="Create Camera" subtitle="MVP form with backend template + JSON config">
      <form className="inline-form" onSubmit={onSubmit}>
        <div className="camera-form-grid">
          <label className="field-label" htmlFor="camera-name-input">
            Camera name
            <input
              id="camera-name-input"
              className="input"
              type="text"
              value={cameraName}
              placeholder="front_door"
              onChange={(event) => onCameraNameChange(event.target.value)}
              disabled={isMutating}
            />
          </label>
          <label className="field-label" htmlFor="camera-backend-select">
            Source backend
            <select
              id="camera-backend-select"
              className="input"
              value={cameraBackend}
              onChange={(event) => onCameraBackendChange(event.target.value as CameraBackend)}
              disabled={isMutating}
            >
              {CAMERA_BACKEND_OPTIONS.map((backend) => (
                <option key={backend} value={backend}>
                  {backend}
                </option>
              ))}
            </select>
          </label>
          <label className="field-label camera-checkbox-field" htmlFor="camera-enabled-checkbox">
            <input
              id="camera-enabled-checkbox"
              type="checkbox"
              checked={cameraEnabled}
              onChange={(event) => onCameraEnabledChange(event.target.checked)}
              disabled={isMutating}
            />
            Enabled
          </label>
        </div>

        <label className="field-label" htmlFor="camera-source-config-input">
          Source config (JSON)
          <textarea
            id="camera-source-config-input"
            className="input camera-source-config"
            value={sourceConfigRaw}
            onChange={(event) => onSourceConfigChange(event.target.value)}
            disabled={isMutating}
          />
        </label>

        {createFormError ? <p className="error-text">{createFormError}</p> : null}

        <div className="inline-form__actions">
          <Button type="submit" disabled={isMutating}>
            {createPending ? 'Creating...' : 'Create camera'}
          </Button>
          <Button variant="ghost" onClick={onResetTemplate} disabled={isMutating}>
            Reset template
          </Button>
        </div>
      </form>
    </Card>
  )
}
