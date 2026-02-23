import type { DiscoveredCameraResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { summarizeOnvifScopes } from '../presentationOnvif'

interface OnvifDiscoverStepProps {
  cameras: DiscoveredCameraResponse[]
  isScanning: boolean
  error: string | null
  onScan: () => void
  onSelect: (camera: DiscoveredCameraResponse) => void
  onCancel: () => void
}

export function OnvifDiscoverStep({
  cameras,
  isScanning,
  error,
  onScan,
  onSelect,
  onCancel,
}: OnvifDiscoverStepProps) {
  return (
    <div className="onvif-wizard__step">
      <div className="onvif-step-header">
        <h3 className="onvif-step-title">Step 1: Discover cameras</h3>
        <Button variant="ghost" onClick={onCancel} disabled={isScanning}>
          Cancel
        </Button>
      </div>

      <div className="inline-form__actions">
        <Button onClick={onScan} disabled={isScanning}>
          {isScanning ? 'Scanning...' : 'Run discovery scan'}
        </Button>
      </div>

      {error ? <p className="error-text">{error}</p> : null}

      {!isScanning && cameras.length === 0 ? (
        <p className="muted">No cameras found. Make sure cameras are on the same subnet.</p>
      ) : null}

      {cameras.length > 0 ? (
        <div className="onvif-camera-grid">
          {cameras.map((camera) => (
            <button
              key={`${camera.ip}:${camera.xaddr}`}
              type="button"
              className="onvif-camera-card"
              onClick={() => {
                onSelect(camera)
              }}
            >
              <p className="onvif-camera-card__title">{camera.ip}</p>
              <p className="subtle onvif-camera-card__line">{camera.xaddr}</p>
              <p className="muted onvif-camera-card__line">{summarizeOnvifScopes(camera.scopes)}</p>
              <p className="muted onvif-camera-card__line">
                Types: {camera.types.length > 0 ? camera.types.join(', ') : 'n/a'}
              </p>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  )
}
