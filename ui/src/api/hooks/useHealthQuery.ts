import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { HealthSnapshot } from '../client'

const HEALTH_QUERY_KEY = ['health'] as const
const HEALTH_REFRESH_MS = 10_000

export function useHealthQuery() {
  return useQuery<HealthSnapshot>({
    queryKey: HEALTH_QUERY_KEY,
    queryFn: ({ signal }) => apiClient.getHealth({ signal }),
    staleTime: HEALTH_REFRESH_MS,
    refetchInterval: HEALTH_REFRESH_MS,
  })
}
