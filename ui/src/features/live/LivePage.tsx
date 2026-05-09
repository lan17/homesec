import { createSearchParams, Link } from 'react-router-dom'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
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
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Live</h1>
          <p className="page__lead">See your cameras first, then jump to camera controls or events.</p>
        </div>
        <Button variant="ghost" onClick={refreshCameras} disabled={camerasQuery.isFetching}>
          {camerasQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

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
        <Card title="Loading cameras">
          <p className="muted">Checking configured cameras...</p>
        </Card>
      ) : null}

      {!camerasQuery.isPending && !camerasQuery.error && cameras.length === 0 ? (
        <Card title="No cameras yet">
          <p className="muted">Add a camera in Settings to start using live view.</p>
          <div className="inline-form__actions">
            <Link className="button button--primary" to="/settings">
              Open Settings
            </Link>
          </div>
        </Card>
      ) : null}

      {cameras.length > 0 ? (
        <div className="live-camera-list">
          {cameras.map((camera) => (
            <article key={camera.name} className="live-camera-row">
              <div>
                <h2 className="live-camera-row__title">{camera.name}</h2>
                <p className="muted">{camera.source_backend}</p>
              </div>
              <StatusBadge tone={cameraStatusTone(camera.enabled, camera.healthy)}>
                {cameraStatusLabel(camera.enabled, camera.healthy)}
              </StatusBadge>
              <div className="inline-form__actions live-camera-row__actions">
                <Link className="button button--primary" to="/cameras">
                  Camera controls
                </Link>
                <Link className="button button--ghost" to={`/events?${eventsSearch(camera.name)}`}>
                  View Events
                </Link>
              </div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}
