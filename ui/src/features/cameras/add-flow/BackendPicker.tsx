import { Button } from '../../../components/ui/Button'
import type { CameraAddBackend } from './types'
import { CAMERA_ADD_BACKEND_OPTIONS } from './types'

interface BackendPickerProps {
  applyChangesImmediately: boolean
  isMutating: boolean
  onSelect: (backend: CameraAddBackend) => void
  onApplyChangesImmediatelyChange: (value: boolean) => void
  onCancel: () => void
}

export function BackendPicker({
  applyChangesImmediately,
  isMutating,
  onSelect,
  onApplyChangesImmediatelyChange,
  onCancel,
}: BackendPickerProps) {
  return (
    <div className="camera-add-flow">
      <div className="camera-add-flow__header">
        <h3 className="camera-add-flow__title">Select source backend</h3>
        <Button variant="ghost" onClick={onCancel} disabled={isMutating}>
          Cancel
        </Button>
      </div>

      <p className="subtle">Choose how this camera will deliver clips into HomeSec.</p>

      <div className="camera-add-flow__backend-grid">
        {CAMERA_ADD_BACKEND_OPTIONS.map((backend) => (
          <button
            key={backend.id}
            type="button"
            className="camera-add-flow__backend-card"
            aria-label={backend.label}
            onClick={() => {
              onSelect(backend.id)
            }}
            disabled={isMutating}
          >
            <p className="camera-add-flow__backend-title">{backend.label}</p>
            <p className="camera-add-flow__backend-description">{backend.description}</p>
          </button>
        ))}
      </div>

      <label className="field-label camera-checkbox-field" htmlFor="camera-add-apply-changes-checkbox">
        <input
          id="camera-add-apply-changes-checkbox"
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
    </div>
  )
}
