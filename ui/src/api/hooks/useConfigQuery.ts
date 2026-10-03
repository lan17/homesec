import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import { QUERY_KEYS } from './queryKeys'

export function useConfigQuery() {
  return useQuery({
    queryKey: QUERY_KEYS.config,
    queryFn: ({ signal }) => apiClient.getConfig({ signal }),
    staleTime: 0,
  })
}
