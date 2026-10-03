import { useMutation, useQueryClient } from '@tanstack/react-query'

import { apiClient, type ConfigPatch } from '../client'
import { QUERY_KEYS } from './queryKeys'

export function useConfigMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (patch: ConfigPatch) => apiClient.patchConfig(patch),
    onSuccess: (config) => {
      queryClient.setQueryData(QUERY_KEYS.config, config)
    },
  })
}
