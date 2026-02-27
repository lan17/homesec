import { Button } from '../../../components/ui/Button'
import { CAMERA_ADD_BACKENDS, CAMERA_BACKEND_ORDER } from './backends'
import type { CameraAddBackend } from './types'

interface BackendPickerProps {
  onSelect: (backend: CameraAddBackend) => void
  onCancel: () => void
}

export function BackendPicker({
  onSelect,
  onCancel,
}: BackendPickerProps) {
  return (
    <div className="camera-add-flow">
      <div className="camera-add-flow__header">
        <h3 className="camera-add-flow__title">Select source backend</h3>
        <Button variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
      </div>

      <p className="subtle">Choose how this camera will deliver clips into HomeSec.</p>

      <div className="camera-add-flow__backend-grid">
        {CAMERA_BACKEND_ORDER.map((backendId) => {
          const backend = CAMERA_ADD_BACKENDS[backendId]
          return (
            <button
              key={backend.id}
              type="button"
              className="camera-add-flow__backend-card"
              aria-label={backend.label}
              onClick={() => {
                onSelect(backend.id)
              }}
            >
              <p className="camera-add-flow__backend-title">{backend.label}</p>
              <p className="camera-add-flow__backend-description">{backend.description}</p>
            </button>
          )
        })}
      </div>
    </div>
  )
}
