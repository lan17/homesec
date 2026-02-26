import { useMemo, useState } from 'react'

import { clearApiKey, isUnauthorizedAPIError, saveApiKey } from '../../api/client'
import { useCamerasQuery } from '../../api/hooks/useCamerasQuery'
import { useRuntimeStatusQuery } from '../../api/hooks/useRuntimeStatusQuery'
import { ApiKeyGate } from '../../components/ui/ApiKeyGate'
import { Button } from '../../components/ui/Button'
import { Card } from '../../components/ui/Card'
import type { CameraCreate } from '../../api/generated/types'
import { useCameraActions } from './hooks/useCameraActions'
import { describeCameraError } from './presentation'
import { RuntimeReloadBanner } from './components/RuntimeReloadBanner'
import { CameraList } from './components/CameraList'
import { CameraAddFlow } from './add-flow/CameraAddFlow'

export function CamerasPage() {
  const camerasQuery = useCamerasQuery()
  const runtimeStatusQuery = useRuntimeStatusQuery()

  const cameraActions = useCameraActions({
    onRuntimeStatusRefresh: async () => {
      await runtimeStatusQuery.refetch()
    },
  })

  const [applyChangesImmediately, setApplyChangesImmediately] = useState(false)
  const [showCameraAddFlow, setShowCameraAddFlow] = useState(false)

  const cameras = useMemo(() => camerasQuery.data ?? [], [camerasQuery.data])
  const runtimeStatus = runtimeStatusQuery.data

  const unauthorized =
    isUnauthorizedAPIError(camerasQuery.error)
    || isUnauthorizedAPIError(runtimeStatusQuery.error)
    || isUnauthorizedAPIError(cameraActions.errors.create)
    || isUnauthorizedAPIError(cameraActions.errors.update)
    || isUnauthorizedAPIError(cameraActions.errors.delete)
    || isUnauthorizedAPIError(cameraActions.errors.reload)

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

  async function handleToggleEnabled(camera: (typeof cameras)[number]): Promise<void> {
    await cameraActions.toggleCameraEnabled(camera, applyChangesImmediately)
  }

  async function handlePatchSourceConfig(
    cameraName: string,
    sourceConfigPatch: CameraCreate['source_config'],
  ): Promise<boolean> {
    return cameraActions.patchCameraSourceConfig(
      cameraName,
      sourceConfigPatch,
      applyChangesImmediately,
    )
  }

  async function handleDelete(camera: (typeof cameras)[number]): Promise<void> {
    const shouldDelete = window.confirm(`Delete camera "${camera.name}"?`)
    if (!shouldDelete) {
      return
    }
    await cameraActions.deleteCamera(camera.name, applyChangesImmediately)
  }

  async function handleApplyRuntimeReload(): Promise<void> {
    await cameraActions.applyRuntimeReload()
  }

  const pageError = useMemo(() => {
    if (unauthorized) {
      return null
    }

    const errors = [
      camerasQuery.error,
      runtimeStatusQuery.error,
      cameraActions.errors.create,
      cameraActions.errors.update,
      cameraActions.errors.delete,
      cameraActions.errors.reload,
    ]
    const firstError = errors.find((error) => error !== null)
    return firstError ? describeCameraError(firstError) : null
  }, [
    cameraActions.errors.create,
    cameraActions.errors.delete,
    cameraActions.errors.reload,
    cameraActions.errors.update,
    camerasQuery.error,
    runtimeStatusQuery.error,
    unauthorized,
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
          disabled={cameraActions.isMutating || camerasQuery.isFetching}
        >
          {camerasQuery.isFetching ? 'Refreshing...' : 'Refresh'}
        </Button>
      </header>

      {unauthorized ? (
        <Card title="Authentication required">
          <ApiKeyGate
            busy={cameraActions.isMutating || camerasQuery.isFetching}
            onSubmit={submitApiKey}
            onClear={clearStoredApiKey}
          />
        </Card>
      ) : null}

      {pageError ? (
        <Card title="Camera operations failed">
          <p className="error-text">{pageError}</p>
        </Card>
      ) : null}

      {showCameraAddFlow ? (
        <CameraAddFlow
          defaultApplyChangesImmediately={applyChangesImmediately}
          onApplyChangesImmediatelyChange={setApplyChangesImmediately}
          onComplete={(payload, options) =>
            cameraActions.createCamera(payload, options.applyChangesImmediately)
          }
          onCancel={() => {
            setShowCameraAddFlow(false)
          }}
        />
      ) : (
        <Card title="Add Camera" subtitle="Launch reusable camera onboarding flow">
          <div className="inline-form__actions">
            <Button
              onClick={() => {
                setShowCameraAddFlow(true)
              }}
              disabled={cameraActions.isMutating}
            >
              Add camera
            </Button>
          </div>
        </Card>
      )}

      <RuntimeReloadBanner
        hasPendingReload={cameraActions.hasPendingReload}
        pendingReloadMessage={cameraActions.pendingReloadMessage}
        actionFeedback={cameraActions.actionFeedback}
        runtimeStatus={runtimeStatus}
        runtimeStatusPending={runtimeStatusQuery.isPending}
        reloadPending={cameraActions.pending.reload}
        onApplyRuntimeReload={() => {
          void handleApplyRuntimeReload()
        }}
      />

      <CameraList
        cameras={cameras}
        isPending={camerasQuery.isPending}
        isMutating={cameraActions.isMutating}
        updatePending={cameraActions.pending.update}
        applyChangesImmediately={applyChangesImmediately}
        onToggleEnabled={(camera) => {
          void handleToggleEnabled(camera)
        }}
        onPatchSourceConfig={handlePatchSourceConfig}
        onDelete={(camera) => {
          void handleDelete(camera)
        }}
      />
    </section>
  )
}
