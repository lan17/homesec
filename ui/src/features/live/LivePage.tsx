import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'

import type { CameraResponse } from '../../api/generated/types'
import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import { ResponsivePageShell } from '../../components/ui/ResponsivePageShell'
import { describeCameraError } from '../cameras/presentation'
import { useSetupRedirect } from '../setup/useSetupRedirect'
import { LiveCameraCard } from './LiveCameraCard'
import { useCompactViewport } from './useCompactViewport'

const EMPTY_CAMERAS: CameraResponse[] = []

export function LivePage() {
  const { shouldRedirect, isChecking } = useSetupRedirect()
  const camerasQuery = useCamerasQuery()
  const cameras = camerasQuery.data ?? EMPTY_CAMERAS
  const unauthorized = isUnauthorizedAPIError(camerasQuery.error)
  const isCompactViewport = useCompactViewport()
  const cameraNames = useMemo(() => cameras.map((camera) => camera.name), [cameras])
  const [selectedCameraName, setSelectedCameraName] = useState<string | null>(null)
  const focusedCameraName =
    selectedCameraName && cameraNames.includes(selectedCameraName)
      ? selectedCameraName
      : cameraNames[0] ?? null

  if (isChecking || shouldRedirect) {
    return null
  }

  async function refreshCameras(): Promise<void> {
    await camerasQuery.refetch()
  }

  async function submitApiKey(apiKey: string): Promise<void> {
    saveApiKey(apiKey)
    await camerasQuery.refetch()
  }

  async function clearStoredApiKey(): Promise<void> {
    clearApiKey()
    await camerasQuery.refetch()
  }

  return (
    <ResponsivePageShell
      title="Live"
      lead="See your cameras first, then jump to camera controls or events."
      actions={
        <Button variant="ghost" onClick={refreshCameras} disabled={camerasQuery.isFetching}>
          {camerasQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      }
    >
      {unauthorized ? (
        <Card title="Authentication required">
          <ApiKeyGate
            busy={camerasQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        </Card>
      ) : null}

      {camerasQuery.error && !unauthorized ? (
        <Card title="Live view unavailable">
          <p className="error-text">{describeCameraError(camerasQuery.error)}</p>
        </Card>
      ) : null}

      {camerasQuery.isPending && cameras.length === 0 ? (
        <EmptyState
          title="Loading cameras"
          description="Checking configured cameras..."
          tone="loading"
        />
      ) : null}

      {!camerasQuery.isPending && !camerasQuery.error && cameras.length === 0 ? (
        <EmptyState
          title="No cameras yet"
          description="Add a camera in Settings to start using live view."
          action={
            <Link className="button button--primary" to="/settings">
              Open Settings
            </Link>
          }
        />
      ) : null}

      {cameras.length > 0 ? (
        <div className="live-camera-list">
          {cameras.map((camera) => (
            <LiveCameraCard
              key={camera.name}
              camera={camera}
              isCompactViewport={isCompactViewport}
              isFocused={focusedCameraName === camera.name}
              onFocusCamera={setSelectedCameraName}
            />
          ))}
        </div>
      ) : null}
    </ResponsivePageShell>
  )
}
