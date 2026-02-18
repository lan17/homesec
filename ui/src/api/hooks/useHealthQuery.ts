import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { HealthSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'
const HEALTH_REFRESH_MS = 10_000

export function useHealthQuery() {
  return useQuery<HealthSnapshot>({
    queryKey: QUERY_KEYS.health,
    queryFn: ({ signal }) => apiClient.getHealth({ signal }),
    staleTime: HEALTH_REFRESH_MS,
    refetchInterval: HEALTH_REFRESH_MS,
  })
}
