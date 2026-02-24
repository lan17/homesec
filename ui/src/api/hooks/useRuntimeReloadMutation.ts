import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { RuntimeReloadResponse } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

export function useRuntimeReloadMutation() {
  const queryClient = useQueryClient()
  return useMutation<RuntimeReloadResponse, Error>({
    mutationFn: () => apiClient.reloadRuntime(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: QUERY_KEYS.runtimeStatus })
    },
  })
}
