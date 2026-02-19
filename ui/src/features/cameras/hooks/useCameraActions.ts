import { useState } from 'react'

import {
  useCreateCameraMutation,
  useDeleteCameraMutation,
  useUpdateCameraMutation,
} from '../../../api/hooks/useCameraMutations'
import { useRuntimeReloadMutation } from '../../../api/hooks/useRuntimeReloadMutation'
import type {
  CameraCreate,
  CameraResponse,
  ConfigChangeResponse,
} from '../../../api/generated/types'

interface UseCameraActionsOptions {
  onRuntimeStatusRefresh: () => Promise<void>
}

interface CameraActionErrors {
  create: unknown
  update: unknown
  delete: unknown
  reload: unknown
}

interface CameraActionPending {
  create: boolean
  update: boolean
  delete: boolean
  reload: boolean
}

interface CameraActionState {
  hasPendingReload: boolean
  pendingReloadMessage: string | null
  actionFeedback: string | null
  isMutating: boolean
  pending: CameraActionPending
  errors: CameraActionErrors
}

interface UseCameraActionsResult extends CameraActionState {
  createCamera: (payload: CameraCreate, applyChanges: boolean) => Promise<boolean>
  toggleCameraEnabled: (camera: CameraResponse, applyChanges: boolean) => Promise<boolean>
  patchCameraSourceConfig: (
    cameraName: string,
    sourceConfigPatch: CameraCreate['source_config'],
    applyChanges: boolean,
  ) => Promise<boolean>
  deleteCamera: (cameraName: string, applyChanges: boolean) => Promise<boolean>
  applyRuntimeReload: () => Promise<boolean>
}

export function useCameraActions({
  onRuntimeStatusRefresh,
}: UseCameraActionsOptions): UseCameraActionsResult {
  const createCameraMutation = useCreateCameraMutation()
  const updateCameraMutation = useUpdateCameraMutation()
  const deleteCameraMutation = useDeleteCameraMutation()
  const runtimeReloadMutation = useRuntimeReloadMutation()

  const [hasPendingReload, setHasPendingReload] = useState(false)
  const [pendingReloadMessage, setPendingReloadMessage] = useState<string | null>(null)
  const [actionFeedback, setActionFeedback] = useState<string | null>(null)

  function markPendingReload(message: string): void {
    setHasPendingReload(true)
    setPendingReloadMessage(message)
  }

  function clearPendingReload(): void {
    setHasPendingReload(false)
    setPendingReloadMessage(null)
  }

  function applyMutationOutcome({
    response,
    restartMessage,
    successMessage,
  }: {
    response: ConfigChangeResponse
    restartMessage: string
    successMessage: string
  }): boolean {
    if (response.restart_required) {
      markPendingReload(restartMessage)
      return false
    }
    clearPendingReload()
    if (response.runtime_reload) {
      setActionFeedback(`${successMessage} ${response.runtime_reload.message}.`)
      return true
    }
    setActionFeedback(successMessage)
    return false
  }

  async function createCamera(payload: CameraCreate, applyChanges: boolean): Promise<boolean> {
    setActionFeedback(null)
    try {
      const response = await createCameraMutation.mutateAsync({ payload, applyChanges })
      const reloadRequested = applyMutationOutcome({
        response,
        restartMessage: `Camera "${payload.name}" created. Runtime reload required.`,
        successMessage: `Camera "${payload.name}" created.`,
      })
      if (reloadRequested) {
        await onRuntimeStatusRefresh()
      }
      return true
    } catch {
      return false
    }
  }

  async function toggleCameraEnabled(camera: CameraResponse, applyChanges: boolean): Promise<boolean> {
    setActionFeedback(null)
    const isEnabling = !camera.enabled
    const verb = isEnabling ? 'enabled' : 'disabled'

    try {
      const response = await updateCameraMutation.mutateAsync({
        name: camera.name,
        payload: { enabled: isEnabling },
        applyChanges,
      })

      const reloadRequested = applyMutationOutcome({
        response,
        restartMessage: `Camera "${camera.name}" ${verb}. Runtime reload required.`,
        successMessage: `Camera "${camera.name}" ${verb}.`,
      })
      if (reloadRequested) {
        await onRuntimeStatusRefresh()
      }
      return true
    } catch {
      return false
    }
  }

  async function patchCameraSourceConfig(
    cameraName: string,
    sourceConfigPatch: CameraCreate['source_config'],
    applyChanges: boolean,
  ): Promise<boolean> {
    setActionFeedback(null)
    try {
      const response = await updateCameraMutation.mutateAsync({
        name: cameraName,
        payload: { source_config: sourceConfigPatch },
        applyChanges,
      })
      const reloadRequested = applyMutationOutcome({
        response,
        restartMessage: `Camera "${cameraName}" source config updated. Runtime reload required.`,
        successMessage: `Camera "${cameraName}" source config updated.`,
      })
      if (reloadRequested) {
        await onRuntimeStatusRefresh()
      }
      return true
    } catch {
      return false
    }
  }

  async function deleteCamera(cameraName: string, applyChanges: boolean): Promise<boolean> {
    setActionFeedback(null)
    try {
      const response = await deleteCameraMutation.mutateAsync({ name: cameraName, applyChanges })
      const reloadRequested = applyMutationOutcome({
        response,
        restartMessage: `Camera "${cameraName}" deleted. Runtime reload required.`,
        successMessage: `Camera "${cameraName}" deleted.`,
      })
      if (reloadRequested) {
        await onRuntimeStatusRefresh()
      }
      return true
    } catch {
      return false
    }
  }

  async function applyRuntimeReload(): Promise<boolean> {
    setActionFeedback(null)
    try {
      const response = await runtimeReloadMutation.mutateAsync()
      clearPendingReload()
      setActionFeedback(response.message)
      await onRuntimeStatusRefresh()
      return true
    } catch {
      return false
    }
  }

  const pending: CameraActionPending = {
    create: createCameraMutation.isPending,
    update: updateCameraMutation.isPending,
    delete: deleteCameraMutation.isPending,
    reload: runtimeReloadMutation.isPending,
  }
  const errors: CameraActionErrors = {
    create: createCameraMutation.error,
    update: updateCameraMutation.error,
    delete: deleteCameraMutation.error,
    reload: runtimeReloadMutation.error,
  }
  const isMutating = pending.create || pending.update || pending.delete || pending.reload

  return {
    createCamera,
    toggleCameraEnabled,
    patchCameraSourceConfig,
    deleteCamera,
    applyRuntimeReload,
    hasPendingReload,
    pendingReloadMessage,
    actionFeedback,
    isMutating,
    pending,
    errors,
  }
}
