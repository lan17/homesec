import { createSearchParams, Link } from 'react-router-dom'

import type { CameraCreate, CameraResponse } from '../../../api/generated/types'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { StatusBadge } from '../../../components/ui/StatusBadge'
import { TechnicalDetailsDisclosure } from '../../../components/ui/TechnicalDetailsDisclosure'
import { cameraHealthLabel, cameraHealthTone, formatLastSeen } from '../cameraHealth'
import { CameraPreviewPanel } from './CameraPreviewPanel'
import { CameraSourceConfigEditor } from './CameraSourceConfigEditor'

interface CameraListProps {
  cameras: CameraResponse[]
  isPending: boolean
  isMutating: boolean
  updatePending: boolean
  applyChangesImmediately: boolean
  onToggleEnabled: (camera: CameraResponse) => void
  onPatchSourceConfig: (
    cameraName: string,
    sourceConfigPatch: CameraCreate['source_config'],
  ) => Promise<boolean>
  onDelete: (camera: CameraResponse) => void
}

function cameraEventsSearch(cameraName: string): string {
  return createSearchParams({ camera: cameraName }).toString()
}

export function CameraList({
  cameras,
  isPending,
  isMutating,
  updatePending,
  applyChangesImmediately,
  onToggleEnabled,
  onPatchSourceConfig,
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
                    {cameraHealthLabel(camera)}
                  </StatusBadge>
                  <span className="camera-chip">{camera.source_backend}</span>
                </div>
              </header>

              <dl className="camera-item__meta">
                <div className="camera-item__meta-row">
                  <dt>Status</dt>
                  <dd>{cameraHealthLabel(camera)}</dd>
                </div>
                <div className="camera-item__meta-row">
                  <dt>Last seen</dt>
                  <dd>{formatLastSeen(camera.last_heartbeat)}</dd>
                </div>
              </dl>

              {camera.source_backend === 'rtsp' ? (
                <CameraPreviewPanel cameraName={camera.name} />
              ) : null}

              <TechnicalDetailsDisclosure summary="Technical source details">
                <pre className="camera-item__config">{JSON.stringify(camera.source_config, null, 2)}</pre>
              </TechnicalDetailsDisclosure>

              <div className="inline-form__actions">
                <Link className="button button--primary" to={`/events?${cameraEventsSearch(camera.name)}`}>
                  View Events
                </Link>
                <CameraSourceConfigEditor
                  camera={camera}
                  isMutating={isMutating}
                  updatePending={updatePending}
                  applyChangesImmediately={applyChangesImmediately}
                  onSubmitPatch={onPatchSourceConfig}
                />
                <Button
                  variant="ghost"
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
                <Link className="button button--ghost" to="/system">
                  Open System
                </Link>
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </Card>
  )
}
