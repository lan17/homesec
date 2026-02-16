import { useState } from 'react'

import {
  useCreateCameraMutation,
  useDeleteCameraMutation,
  useUpdateCameraMutation,
} from '../../../api/hooks/useCameraMutations'
import { useRuntimeReloadMutation } from '../../../api/hooks/useRuntimeReloadMutation'
import type { CameraCreate, CameraResponse } from '../../../api/generated/types'

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
  createCamera: (payload: CameraCreate) => Promise<boolean>
  toggleCameraEnabled: (camera: CameraResponse) => Promise<boolean>
  deleteCamera: (cameraName: string) => Promise<boolean>
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
    restartRequired,
    restartMessage,
    successMessage,
  }: {
    restartRequired: boolean
    restartMessage: string
    successMessage: string
  }): void {
    if (restartRequired) {
      markPendingReload(restartMessage)
      return
    }
    clearPendingReload()
    setActionFeedback(successMessage)
  }

  async function createCamera(payload: CameraCreate): Promise<boolean> {
    setActionFeedback(null)
    try {
      const response = await createCameraMutation.mutateAsync(payload)
      applyMutationOutcome({
        restartRequired: response.restart_required,
        restartMessage: `Camera "${payload.name}" created. Runtime reload required.`,
        successMessage: `Camera "${payload.name}" created.`,
      })
      return true
    } catch {
      return false
    }
  }

  async function toggleCameraEnabled(camera: CameraResponse): Promise<boolean> {
    setActionFeedback(null)
    const isEnabling = !camera.enabled
    const verb = isEnabling ? 'enabled' : 'disabled'

    try {
      const response = await updateCameraMutation.mutateAsync({
        name: camera.name,
        payload: { enabled: isEnabling },
      })

      applyMutationOutcome({
        restartRequired: response.restart_required,
        restartMessage: `Camera "${camera.name}" ${verb}. Runtime reload required.`,
        successMessage: `Camera "${camera.name}" ${verb}.`,
      })
      return true
    } catch {
      return false
    }
  }

  async function deleteCamera(cameraName: string): Promise<boolean> {
    setActionFeedback(null)
    try {
      const response = await deleteCameraMutation.mutateAsync({ name: cameraName })
      applyMutationOutcome({
        restartRequired: response.restart_required,
        restartMessage: `Camera "${cameraName}" deleted. Runtime reload required.`,
        successMessage: `Camera "${cameraName}" deleted.`,
      })
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
