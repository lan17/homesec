import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { StatsSnapshot } from '../client'

const STATS_QUERY_KEY = ['stats'] as const
const STATS_REFRESH_MS = 10_000

export function useStatsQuery() {
  return useQuery<StatsSnapshot>({
    queryKey: STATS_QUERY_KEY,
    queryFn: ({ signal }) => apiClient.getStats({ signal }),
    staleTime: STATS_REFRESH_MS,
    refetchInterval: STATS_REFRESH_MS,
  })
}
