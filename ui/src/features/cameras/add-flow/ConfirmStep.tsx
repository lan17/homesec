import { Button } from '../../../components/ui/Button'
import type { TestConnectionResponse } from '../../../api/generated/types'

interface ConfirmStepProps {
  backendLabel: string
  cameraName: string
  config: Record<string, unknown>
  applyChangesImmediately: boolean
  testResult: TestConnectionResponse | null
  submitPending: boolean
  createError: string | null
  onCameraNameChange: (cameraName: string) => void
  onApplyChangesImmediatelyChange: (value: boolean) => void
  onSubmit: () => void
}

export function ConfirmStep({
  backendLabel,
  cameraName,
  config,
  applyChangesImmediately,
  testResult,
  submitPending,
  createError,
  onCameraNameChange,
  onApplyChangesImmediatelyChange,
  onSubmit,
}: ConfirmStepProps) {
  return (
    <section className="inline-form">
      <h3 className="camera-add-flow__title">Confirm camera details</h3>
      <p className="subtle">
        Review generated config and choose whether to apply runtime changes immediately.
      </p>

      <label className="field-label" htmlFor="camera-confirm-name">
        Camera name
        <input
          id="camera-confirm-name"
          className="input"
          type="text"
          value={cameraName}
          disabled={submitPending}
          onChange={(event) => {
            onCameraNameChange(event.target.value)
          }}
        />
      </label>

      <dl className="camera-runtime-kv">
        <div className="camera-runtime-row">
          <dt>Backend</dt>
          <dd>{backendLabel}</dd>
        </div>
        <div className="camera-runtime-row">
          <dt>Connection test</dt>
          <dd>
            {testResult
              ? testResult.success
                ? `Passed: ${testResult.message}`
                : `Failed: ${testResult.message}`
              : 'Not run'}
          </dd>
        </div>
      </dl>

      <label className="field-label form-checkbox-field" htmlFor="camera-confirm-apply-changes">
        <input
          id="camera-confirm-apply-changes"
          type="checkbox"
          checked={applyChangesImmediately}
          disabled={submitPending}
          onChange={(event) => {
            onApplyChangesImmediatelyChange(event.target.checked)
          }}
        />
        Apply changes immediately (runtime reload)
      </label>
      <p className="subtle">
        If disabled, config changes are persisted and can be applied from Runtime Control later.
      </p>

      <label className="field-label">
        Source config preview
        <pre className="camera-item__config">{JSON.stringify(config, null, 2)}</pre>
      </label>

      {createError ? <p className="error-text">{createError}</p> : null}

      <div className="inline-form__actions">
        <Button
          onClick={onSubmit}
          disabled={submitPending}
        >
          {submitPending ? 'Creating...' : 'Create camera'}
        </Button>
      </div>
    </section>
  )
}
