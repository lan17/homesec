import { createSearchParams, Link } from 'react-router-dom'

import type { CameraResponse } from '../../api/generated/types'
import { CameraCard } from '../../components/ui/CameraCard'
import { MediaPanel } from '../../components/ui/MediaPanel'
import { StatusBadge, type StatusBadgeTone } from '../../components/ui/StatusBadge'
import { TechnicalDetailsDisclosure } from '../../components/ui/TechnicalDetailsDisclosure'
import { CameraPreviewPanel } from '../cameras/components/CameraPreviewPanel'

interface LiveCameraCardProps {
  camera: CameraResponse
}

function cameraStatusTone(camera: CameraResponse): StatusBadgeTone {
  if (!camera.enabled) {
    return 'unknown'
  }
  return camera.healthy ? 'healthy' : 'unhealthy'
}

function cameraStatusLabel(camera: CameraResponse): string {
  if (!camera.enabled) {
    return 'Disabled'
  }
  return camera.healthy ? 'Online' : 'Offline'
}

function formatLastSeen(value: number | null): string {
  if (!value) {
    return 'Status unavailable'
  }
  return new Date(value * 1000).toLocaleString()
}

function eventsSearch(cameraName: string): string {
  return createSearchParams({ camera: cameraName }).toString()
}

function canShowPreview(camera: CameraResponse): boolean {
  return camera.enabled && camera.source_backend === 'rtsp'
}

function renderPreview(camera: CameraResponse) {
  if (canShowPreview(camera)) {
    return (
      <CameraPreviewPanel
        cameraName={camera.name}
        title="Preview"
        subtitle="Start preview when you want to watch this camera."
        showTalkControl={false}
      />
    )
  }

  return (
    <MediaPanel
      title="Preview"
      placeholder={
        camera.enabled
          ? 'Live preview is not available for this camera source.'
          : 'Enable this camera to use live preview.'
      }
    />
  )
}

export function LiveCameraCard({ camera }: LiveCameraCardProps) {
  return (
    <CameraCard
      title={camera.name}
      status={
        <StatusBadge tone={cameraStatusTone(camera)}>
          {cameraStatusLabel(camera)}
        </StatusBadge>
      }
      media={renderPreview(camera)}
      meta={[
        { label: 'Last seen', value: formatLastSeen(camera.last_heartbeat) },
      ]}
      technicalDetails={
        <TechnicalDetailsDisclosure>
          <dl className="camera-card__meta">
            <div className="camera-card__meta-row">
              <dt>Source</dt>
              <dd>{camera.source_backend}</dd>
            </div>
            <div className="camera-card__meta-row">
              <dt>Enabled</dt>
              <dd>{camera.enabled ? 'true' : 'false'}</dd>
            </div>
          </dl>
        </TechnicalDetailsDisclosure>
      }
      actions={
        <>
          <Link className="button button--primary" to={`/events?${eventsSearch(camera.name)}`}>
            View Events
          </Link>
          <Link className="button button--ghost" to="/cameras">
            Camera controls
          </Link>
        </>
      }
    />
  )
}
