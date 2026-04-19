import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { StorageChangeResponse, StorageUpdate } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

interface UpdateStorageInput {
  payload: StorageUpdate
  applyChanges?: boolean
}

export function useUpdateStorageMutation() {
  const queryClient = useQueryClient()
  return useMutation<StorageChangeResponse, Error, UpdateStorageInput>({
    mutationFn: ({ payload, applyChanges }) =>
      apiClient.updateStorage(payload, { applyChanges }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.storage }),
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.storageBackends }),
      ])
    },
  })
}
