import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { CameraCreate, CameraUpdate, ConfigChangeResponse } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

interface CreateCameraInput {
  payload: CameraCreate
  applyChanges?: boolean
}

interface UpdateCameraInput {
  name: string
  payload: CameraUpdate
  applyChanges?: boolean
}

interface DeleteCameraInput {
  name: string
  applyChanges?: boolean
}

function invalidateCameras(queryClient: ReturnType<typeof useQueryClient>): Promise<void> {
  return queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameras })
}

export function useCreateCameraMutation() {
  const queryClient = useQueryClient()
  return useMutation<ConfigChangeResponse, Error, CreateCameraInput>({
    mutationFn: ({ payload, applyChanges }) =>
      apiClient.createCamera(payload, { applyChanges }),
    onSuccess: async () => {
      await invalidateCameras(queryClient)
    },
  })
}

export function useUpdateCameraMutation() {
  const queryClient = useQueryClient()
  return useMutation<ConfigChangeResponse, Error, UpdateCameraInput>({
    mutationFn: ({ name, payload, applyChanges }) =>
      apiClient.updateCamera(name, payload, { applyChanges }),
    onSuccess: async () => {
      await invalidateCameras(queryClient)
    },
  })
}

export function useDeleteCameraMutation() {
  const queryClient = useQueryClient()
  return useMutation<ConfigChangeResponse, Error, DeleteCameraInput>({
    mutationFn: ({ name, applyChanges }) => apiClient.deleteCamera(name, { applyChanges }),
    onSuccess: async () => {
      await invalidateCameras(queryClient)
    },
  })
}
