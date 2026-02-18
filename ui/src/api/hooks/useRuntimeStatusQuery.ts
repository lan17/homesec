import { useQuery } from '@tanstack/react-query'

import { apiClient, type RuntimeStatusSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'

const RUNTIME_STATUS_REFRESH_MS = 5_000

export function useRuntimeStatusQuery() {
  return useQuery<RuntimeStatusSnapshot>({
    queryKey: QUERY_KEYS.runtimeStatus,
    queryFn: ({ signal }) => apiClient.getRuntimeStatus({ signal }),
    staleTime: RUNTIME_STATUS_REFRESH_MS,
    refetchInterval: RUNTIME_STATUS_REFRESH_MS,
  })
}
