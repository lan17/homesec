import type { FormEvent } from 'react'

import { Button } from '../../../components/ui/Button'
import type { CameraBackend } from '../forms'

interface ManualCameraConfigureStepProps {
  backend: CameraBackend
  cameraName: string
  cameraEnabled: boolean
  sourceConfigRaw: string
  errorMessage: string | null
  applyChangesImmediately: boolean
  isMutating: boolean
  createPending: boolean
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
  onBack: () => void
  onCancel: () => void
  onCameraNameChange: (value: string) => void
  onCameraEnabledChange: (value: boolean) => void
  onSourceConfigChange: (value: string) => void
  onApplyChangesImmediatelyChange: (value: boolean) => void
  onResetTemplate: () => void
}

export function ManualCameraConfigureStep({
  backend,
  cameraName,
  cameraEnabled,
  sourceConfigRaw,
  errorMessage,
  applyChangesImmediately,
  isMutating,
  createPending,
  onSubmit,
  onBack,
  onCancel,
  onCameraNameChange,
  onCameraEnabledChange,
  onSourceConfigChange,
  onApplyChangesImmediatelyChange,
  onResetTemplate,
}: ManualCameraConfigureStepProps) {
  return (
    <form className="inline-form camera-add-flow" onSubmit={onSubmit}>
      <div className="camera-add-flow__header">
        <div>
          <h3 className="camera-add-flow__title">Configure {backend}</h3>
          <p className="subtle">Create camera config for {backend} source backend.</p>
        </div>
        <div className="inline-form__actions">
          <Button variant="ghost" onClick={onBack} disabled={isMutating}>
            Back
          </Button>
          <Button variant="ghost" onClick={onCancel} disabled={isMutating}>
            Cancel
          </Button>
        </div>
      </div>

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

      {errorMessage ? <p className="error-text">{errorMessage}</p> : null}

      <label className="field-label camera-checkbox-field" htmlFor="camera-apply-changes-checkbox">
        <input
          id="camera-apply-changes-checkbox"
          type="checkbox"
          checked={applyChangesImmediately}
          onChange={(event) => onApplyChangesImmediatelyChange(event.target.checked)}
          disabled={isMutating}
        />
        Apply changes immediately (runtime reload)
      </label>
      <p className="subtle">
        Applies to camera create, source-config patch, enable/disable, and delete actions.
      </p>

      <div className="inline-form__actions">
        <Button type="submit" disabled={isMutating}>
          {createPending ? 'Creating...' : 'Create camera'}
        </Button>
        <Button variant="ghost" onClick={onResetTemplate} disabled={isMutating}>
          Reset template
        </Button>
      </div>
    </form>
  )
}
