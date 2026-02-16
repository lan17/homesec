import { useMemo, useState, type FormEvent } from 'react'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import {
  useCreateCameraMutation,
  useDeleteCameraMutation,
  useUpdateCameraMutation,
} from '../../api/hooks/useCameraMutations'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { useRuntimeReloadMutation } from '../../api/hooks/useRuntimeReloadMutation'
import { useRuntimeStatusQuery } from '../../api/hooks/useRuntimeStatusQuery'
import type { CameraResponse } from '../../api/generated/types'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import { StatusBadge } from '../../components/ui/StatusBadge'
import {
  CAMERA_BACKEND_OPTIONS,
  type CameraBackend,
  defaultSourceConfigForBackend,
  parseSourceConfigJson,
} from './forms'
import { describeCameraError, formatRuntimeTimestamp, runtimeStatusTone } from './presentation'

function cameraHealthTone(camera: CameraResponse): 'healthy' | 'unknown' | 'unhealthy' {
  if (!camera.enabled) {
    return 'unknown'
  }
  return camera.healthy ? 'healthy' : 'unhealthy'
}

export function CamerasPage() {
  const camerasQuery = useCamerasQuery()
  const runtimeStatusQuery = useRuntimeStatusQuery()
  const createCameraMutation = useCreateCameraMutation()
  const updateCameraMutation = useUpdateCameraMutation()
  const deleteCameraMutation = useDeleteCameraMutation()
  const runtimeReloadMutation = useRuntimeReloadMutation()

  const [cameraName, setCameraName] = useState('')
  const [cameraEnabled, setCameraEnabled] = useState(true)
  const [cameraBackend, setCameraBackend] = useState<CameraBackend>('rtsp')
  const [sourceConfigRaw, setSourceConfigRaw] = useState(defaultSourceConfigForBackend('rtsp'))
  const [createFormError, setCreateFormError] = useState<string | null>(null)
  const [hasPendingReload, setHasPendingReload] = useState(false)
  const [pendingReloadMessage, setPendingReloadMessage] = useState<string | null>(null)
  const [actionFeedback, setActionFeedback] = useState<string | null>(null)

  const cameras = useMemo(() => camerasQuery.data ?? [], [camerasQuery.data])
  const runtimeStatus = runtimeStatusQuery.data

  const unauthorized =
    isUnauthorizedAPIError(camerasQuery.error)
    || isUnauthorizedAPIError(runtimeStatusQuery.error)
    || isUnauthorizedAPIError(createCameraMutation.error)
    || isUnauthorizedAPIError(updateCameraMutation.error)
    || isUnauthorizedAPIError(deleteCameraMutation.error)
    || isUnauthorizedAPIError(runtimeReloadMutation.error)

  const isMutating =
    createCameraMutation.isPending
    || updateCameraMutation.isPending
    || deleteCameraMutation.isPending
    || runtimeReloadMutation.isPending

  async function refreshAll(): Promise<void> {
    await Promise.all([camerasQuery.refetch(), runtimeStatusQuery.refetch()])
  }

  async function submitApiKey(apiKey: string): Promise<void> {
    saveApiKey(apiKey)
    await refreshAll()
  }

  async function clearStoredApiKey(): Promise<void> {
    clearApiKey()
    await refreshAll()
  }

  function updateBackend(nextBackend: CameraBackend): void {
    setCameraBackend(nextBackend)
    setSourceConfigRaw(defaultSourceConfigForBackend(nextBackend))
    setCreateFormError(null)
  }

  function markPendingReload(message: string): void {
    setHasPendingReload(true)
    setPendingReloadMessage(message)
  }

  async function handleCreateCamera(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault()

    const normalizedName = cameraName.trim()
    if (!normalizedName) {
      setCreateFormError('Camera name is required.')
      return
    }

    const parsedSourceConfig = parseSourceConfigJson(sourceConfigRaw)
    if (!parsedSourceConfig.ok) {
      setCreateFormError(parsedSourceConfig.message)
      return
    }

    setCreateFormError(null)
    setActionFeedback(null)
    try {
      const response = await createCameraMutation.mutateAsync({
        name: normalizedName,
        enabled: cameraEnabled,
        source_backend: cameraBackend,
        source_config: parsedSourceConfig.value,
      })

      setCameraName('')
      if (response.restart_required) {
        markPendingReload(`Camera "${normalizedName}" created. Runtime reload required.`)
      } else {
        setActionFeedback(`Camera "${normalizedName}" created.`)
      }
    } catch {
      return
    }
  }

  async function handleToggleEnabled(camera: CameraResponse): Promise<void> {
    setActionFeedback(null)
    try {
      const response = await updateCameraMutation.mutateAsync({
        name: camera.name,
        payload: { enabled: !camera.enabled },
      })

      if (response.restart_required) {
        const verb = camera.enabled ? 'disabled' : 'enabled'
        markPendingReload(`Camera "${camera.name}" ${verb}. Runtime reload required.`)
        return
      }

      const verb = camera.enabled ? 'disabled' : 'enabled'
      setActionFeedback(`Camera "${camera.name}" ${verb}.`)
    } catch {
      return
    }
  }

  async function handleDelete(camera: CameraResponse): Promise<void> {
    const shouldDelete = window.confirm(`Delete camera "${camera.name}"?`)
    if (!shouldDelete) {
      return
    }

    setActionFeedback(null)
    try {
      const response = await deleteCameraMutation.mutateAsync({ name: camera.name })
      if (response.restart_required) {
        markPendingReload(`Camera "${camera.name}" deleted. Runtime reload required.`)
        return
      }
      setActionFeedback(`Camera "${camera.name}" deleted.`)
    } catch {
      return
    }
  }

  async function handleApplyRuntimeReload(): Promise<void> {
    setActionFeedback(null)
    try {
      const response = await runtimeReloadMutation.mutateAsync()
      setHasPendingReload(false)
      setPendingReloadMessage(null)
      setActionFeedback(response.message)
      await runtimeStatusQuery.refetch()
    } catch {
      return
    }
  }

  const pageError = useMemo(() => {
    if (unauthorized) {
      return null
    }

    const errors = [
      camerasQuery.error,
      runtimeStatusQuery.error,
      createCameraMutation.error,
      updateCameraMutation.error,
      deleteCameraMutation.error,
      runtimeReloadMutation.error,
    ]
    const firstError = errors.find((error) => error !== null)
    return firstError ? describeCameraError(firstError) : null
  }, [
    camerasQuery.error,
    createCameraMutation.error,
    deleteCameraMutation.error,
    runtimeReloadMutation.error,
    runtimeStatusQuery.error,
    unauthorized,
    updateCameraMutation.error,
  ])

  return (
    <section className="page fade-in-up">
      <header className="page__header">
        <div>
          <h1 className="page__title">Cameras</h1>
          <p className="page__lead">Manage camera definitions and apply runtime reloads.</p>
        </div>
        <Button
          variant="ghost"
          onClick={() => {
            void refreshAll()
          }}
          disabled={isMutating || camerasQuery.isFetching}
        >
          {camerasQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      {unauthorized ? (
        <Card title="Authentication required">
          <ApiKeyGate busy={isMutating || camerasQuery.isFetching} onSubmit={submitApiKey} onClear={clearStoredApiKey} />
        </Card>
      ) : null}

      {pageError ? (
        <Card title="Camera operations failed">
          <p className="error-text">{pageError}</p>
        </Card>
      ) : null}

      <Card title="Create Camera" subtitle="MVP form with backend template + JSON config">
        <form
          className="inline-form"
          onSubmit={(event) => {
            void handleCreateCamera(event)
          }}
        >
          <div className="clips-filter-grid">
            <label className="field-label" htmlFor="camera-name-input">
              Camera name
              <input
                id="camera-name-input"
                className="input"
                type="text"
                value={cameraName}
                placeholder="front_door"
                onChange={(event) => setCameraName(event.target.value)}
                disabled={isMutating}
              />
            </label>
            <label className="field-label" htmlFor="camera-backend-select">
              Source backend
              <select
                id="camera-backend-select"
                className="input"
                value={cameraBackend}
                onChange={(event) => updateBackend(event.target.value as CameraBackend)}
                disabled={isMutating}
              >
                {CAMERA_BACKEND_OPTIONS.map((backend) => (
                  <option key={backend} value={backend}>
                    {backend}
                  </option>
                ))}
              </select>
            </label>
            <label className="field-label camera-checkbox-field" htmlFor="camera-enabled-checkbox">
              <input
                id="camera-enabled-checkbox"
                type="checkbox"
                checked={cameraEnabled}
                onChange={(event) => setCameraEnabled(event.target.checked)}
                disabled={isMutating}
              />
              Enabled
            </label>
          </div>

          <label className="field-label" htmlFor="camera-source-config-input">
            Source config (JSON)
            <textarea
              id="camera-source-config-input"
              className="input camera-source-config"
              value={sourceConfigRaw}
              onChange={(event) => setSourceConfigRaw(event.target.value)}
              disabled={isMutating}
            />
          </label>

          {createFormError ? <p className="error-text">{createFormError}</p> : null}

          <div className="inline-form__actions">
            <Button type="submit" disabled={isMutating}>
              {createCameraMutation.isPending ? 'Creating...' : 'Create camera'}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setSourceConfigRaw(defaultSourceConfigForBackend(cameraBackend))
              }}
              disabled={isMutating}
            >
              Reset template
            </Button>
          </div>
        </form>
      </Card>

      <Card title="Runtime Control" subtitle="Apply pending config changes via runtime reload">
        {hasPendingReload ? (
          <div className="camera-restart-banner">
            <p className="muted">
              {pendingReloadMessage ?? 'One or more camera changes are pending runtime reload.'}
            </p>
            <Button
              onClick={() => {
                void handleApplyRuntimeReload()
              }}
              disabled={runtimeReloadMutation.isPending}
            >
              {runtimeReloadMutation.isPending ? 'Reloading...' : 'Apply runtime reload'}
            </Button>
          </div>
        ) : (
          <p className="muted">No pending restart-required camera changes.</p>
        )}

        {actionFeedback ? <p className="subtle">{actionFeedback}</p> : null}

        {runtimeStatusQuery.isPending && !runtimeStatus ? (
          <p className="muted">Fetching runtime status...</p>
        ) : null}

        {runtimeStatus ? (
          <dl className="camera-runtime-kv">
            <div className="camera-runtime-row">
              <dt>Status</dt>
              <dd>
                <StatusBadge tone={runtimeStatusTone(runtimeStatus)}>
                  {runtimeStatus.state.toUpperCase()}
                </StatusBadge>
              </dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Generation</dt>
              <dd>{runtimeStatus.generation}</dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Active config version</dt>
              <dd className="camera-mono">{runtimeStatus.active_config_version ?? 'n/a'}</dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Last reload at</dt>
              <dd>{formatRuntimeTimestamp(runtimeStatus.last_reload_at)}</dd>
            </div>
            <div className="camera-runtime-row">
              <dt>Last reload error</dt>
              <dd>{runtimeStatus.last_reload_error ?? 'none'}</dd>
            </div>
          </dl>
        ) : null}
      </Card>

      <Card title="Camera Inventory" subtitle="Current runtime camera definitions">
        {camerasQuery.isPending && cameras.length === 0 ? (
          <p className="muted">Loading cameras...</p>
        ) : null}

        {!camerasQuery.isPending && cameras.length === 0 ? (
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
                    <dd>
                      {camera.last_heartbeat ? new Date(camera.last_heartbeat * 1000).toLocaleString() : 'n/a'}
                    </dd>
                  </div>
                </dl>

                <pre className="camera-item__config">{JSON.stringify(camera.source_config, null, 2)}</pre>

                <div className="inline-form__actions">
                  <Button
                    onClick={() => {
                      void handleToggleEnabled(camera)
                    }}
                    disabled={isMutating}
                  >
                    {camera.enabled ? 'Disable' : 'Enable'}
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={() => {
                      void handleDelete(camera)
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
    </section>
  )
}
