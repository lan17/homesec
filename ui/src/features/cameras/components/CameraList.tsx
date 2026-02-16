import type { CameraResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { StatusBadge } from '../../../components/ui/StatusBadge'

interface CameraListProps {
  cameras: CameraResponse[]
  isPending: boolean
  isMutating: boolean
  onToggleEnabled: (camera: CameraResponse) => void
  onDelete: (camera: CameraResponse) => void
}

function cameraHealthTone(camera: CameraResponse): 'healthy' | 'unknown' | 'unhealthy' {
  if (!camera.enabled) {
    return 'unknown'
  }
  return camera.healthy ? 'healthy' : 'unhealthy'
}

function formatLastHeartbeat(value: number | null): string {
  if (!value) {
    return 'n/a'
  }
  return new Date(value * 1000).toLocaleString()
}

export function CameraList({
  cameras,
  isPending,
  isMutating,
  onToggleEnabled,
  onDelete,
}: CameraListProps) {
  return (
    <Card title="Camera Inventory" subtitle="Current runtime camera definitions">
      {isPending && cameras.length === 0 ? <p className="muted">Loading cameras...</p> : null}

      {!isPending && cameras.length === 0 ? (
        <p className="muted">No cameras configured yet. Create your first camera above.</p>
      ) : null}

      {cameras.length > 0 ? (
        <div className="cameras-grid">
          {cameras.map((camera) => (
            <article key={camera.name} className="camera-item">
              <header className="camera-item__header">
                <p className="camera-item__name">{camera.name}</p>
                <div className="camera-item__badges">
                  <StatusBadge tone={cameraHealthTone(camera)}>
                    {camera.healthy && camera.enabled ? 'HEALTHY' : camera.enabled ? 'UNHEALTHY' : 'DISABLED'}
                  </StatusBadge>
                  <span className="clips-chip">{camera.source_backend}</span>
                </div>
              </header>

              <dl className="camera-item__meta">
                <div className="camera-item__meta-row">
                  <dt>Enabled</dt>
                  <dd>{camera.enabled ? 'true' : 'false'}</dd>
                </div>
                <div className="camera-item__meta-row">
                  <dt>Last heartbeat</dt>
                  <dd>{formatLastHeartbeat(camera.last_heartbeat)}</dd>
                </div>
              </dl>

              <pre className="camera-item__config">{JSON.stringify(camera.source_config, null, 2)}</pre>

              <div className="inline-form__actions">
                <Button
                  onClick={() => {
                    onToggleEnabled(camera)
                  }}
                  disabled={isMutating}
                >
                  {camera.enabled ? 'Disable' : 'Enable'}
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    onDelete(camera)
                  }}
                  disabled={isMutating}
                >
                  Delete
                </Button>
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </Card>
  )
}
