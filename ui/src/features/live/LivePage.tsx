import { createSearchParams, Link } from 'react-router-dom'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { CameraCard } from '../../components/ui/CameraCard'
import { Card } from '../../components/ui/Card'
import { EmptyState } from '../../components/ui/EmptyState'
import { ResponsivePageShell } from '../../components/ui/ResponsivePageShell'
import { StatusBadge } from '../../components/ui/StatusBadge'
import { describeCameraError } from '../cameras/presentation'
import { useSetupRedirect } from '../setup/useSetupRedirect'

function cameraStatusTone(enabled: boolean, healthy: boolean): 'healthy' | 'unknown' | 'unhealthy' {
  if (!enabled) {
    return 'unknown'
  }
  return healthy ? 'healthy' : 'unhealthy'
}

function cameraStatusLabel(enabled: boolean, healthy: boolean): string {
  if (!enabled) {
    return 'Disabled'
  }
  return healthy ? 'Online' : 'Offline'
}

function eventsSearch(cameraName: string): string {
  return createSearchParams({ camera: cameraName }).toString()
}

export function LivePage() {
  const { shouldRedirect, isChecking } = useSetupRedirect()
  const camerasQuery = useCamerasQuery()
  const cameras = camerasQuery.data ?? []
  const unauthorized = isUnauthorizedAPIError(camerasQuery.error)

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
            <CameraCard
              key={camera.name}
              title={camera.name}
              subtitle={camera.source_backend}
              status={
                <StatusBadge tone={cameraStatusTone(camera.enabled, camera.healthy)}>
                  {cameraStatusLabel(camera.enabled, camera.healthy)}
                </StatusBadge>
              }
              actions={
                <>
                <Link className="button button--primary" to="/cameras">
                  Camera controls
                </Link>
                  <Link
                    className="button button--ghost"
                    to={`/events?${eventsSearch(camera.name)}`}
                  >
                    View Events
                  </Link>
                </>
              }
            />
          ))}
        </div>
      ) : null}
    </ResponsivePageShell>
  )
}
