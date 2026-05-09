import { createSearchParams, Link } from 'react-router-dom'

import type { CameraResponse } from '../../api/generated/types'
import { Button } from '../../components/ui/Button'
import { CameraCard } from '../../components/ui/CameraCard'
import { MediaPanel } from '../../components/ui/MediaPanel'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { TechnicalDetailsDisclosure } from '../../components/ui/TechnicalDetailsDisclosure'
import { CameraPreviewPanel } from '../cameras/components/CameraPreviewPanel'
import { cameraHealthLabel, cameraHealthTone, formatLastSeen } from '../cameras/cameraHealth'
import { formatCameraSourceLabel } from '../cameras/presentation'

interface LiveCameraCardProps {
  camera: CameraResponse
  isCompactViewport?: boolean
  isFocused?: boolean
  onFocusCamera?: (cameraName: string) => void
}

function eventsSearch(cameraName: string): string {
  return createSearchParams({ camera: cameraName }).toString()
}

function canShowPreview(camera: CameraResponse): boolean {
  return camera.enabled && camera.source_backend === 'rtsp'
}

function cardClassName(isFocused: boolean | undefined): string {
  return isFocused ? 'live-camera-card live-camera-card--focused' : 'live-camera-card'
}

function renderPreview({
  camera,
  isCompactViewport = false,
  isFocused = false,
  onFocusCamera,
}: LiveCameraCardProps) {
  if (canShowPreview(camera)) {
    if (isCompactViewport && !isFocused) {
      return (
        <MediaPanel
          title="Preview"
          subtitle="Open one camera at a time on mobile."
          actions={
            <Button
              variant="ghost"
              onClick={() => {
                onFocusCamera?.(camera.name)
              }}
            >
              Open live view
            </Button>
          }
          placeholder="Open live view to start preview controls."
        />
      )
    }

    return (
      <CameraPreviewPanel
        cameraName={camera.name}
        title="Live view"
        subtitle="Watch this camera and use push-to-talk when available."
        showTalkControl
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

export function LiveCameraCard(props: LiveCameraCardProps) {
  const { camera, isFocused } = props
  return (
    <CameraCard
      className={cardClassName(isFocused)}
      title={camera.name}
      status={
        <StatusBadge tone={cameraHealthTone(camera)}>
          {cameraHealthLabel(camera)}
        </StatusBadge>
      }
      media={renderPreview(props)}
      meta={[
        { label: 'Last seen', value: formatLastSeen(camera.last_heartbeat) },
      ]}
      technicalDetails={
        <TechnicalDetailsDisclosure>
          <dl className="camera-card__meta">
            <div className="camera-card__meta-row">
              <dt>Source</dt>
              <dd>{formatCameraSourceLabel(camera.source_backend)}</dd>
            </div>
            <div className="camera-card__meta-row">
              <dt>Enabled</dt>
              <dd>{camera.enabled ? 'Yes' : 'No'}</dd>
            </div>
          </dl>
        </TechnicalDetailsDisclosure>
      }
      actions={
        <>
          <Link className="button button--primary" to={`/events?${eventsSearch(camera.name)}`}>
            View Events
          </Link>
          <Link className="button button--ghost" to="/settings/cameras">
            Camera settings
          </Link>
        </>
      }
    />
  )
}
