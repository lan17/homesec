import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { CameraCreate, CameraUpdate, ConfigChangeResponse } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

interface UpdateCameraInput {
  name: string
  payload: CameraUpdate
}

interface DeleteCameraInput {
  name: string
}

function invalidateCameras(queryClient: ReturnType<typeof useQueryClient>): Promise<void> {
  return queryClient.invalidateQueries({ queryKey: QUERY_KEYS.cameras })
}

export function useCreateCameraMutation() {
  const queryClient = useQueryClient()
  return useMutation<ConfigChangeResponse, Error, CameraCreate>({
    mutationFn: (payload) => apiClient.createCamera(payload),
    onSuccess: async () => {
      await invalidateCameras(queryClient)
    },
  })
}

export function useUpdateCameraMutation() {
  const queryClient = useQueryClient()
  return useMutation<ConfigChangeResponse, Error, UpdateCameraInput>({
    mutationFn: ({ name, payload }) => apiClient.updateCamera(name, payload),
    onSuccess: async () => {
      await invalidateCameras(queryClient)
    },
  })
}

export function useDeleteCameraMutation() {
  const queryClient = useQueryClient()
  return useMutation<ConfigChangeResponse, Error, DeleteCameraInput>({
    mutationFn: ({ name }) => apiClient.deleteCamera(name),
    onSuccess: async () => {
      await invalidateCameras(queryClient)
    },
  })
}
