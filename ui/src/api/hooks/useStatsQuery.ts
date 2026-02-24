import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { StatsSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'
const STATS_REFRESH_MS = 10_000

export function useStatsQuery() {
  return useQuery<StatsSnapshot>({
    queryKey: QUERY_KEYS.stats,
    queryFn: ({ signal }) => apiClient.getStats({ signal }),
    staleTime: STATS_REFRESH_MS,
    refetchInterval: STATS_REFRESH_MS,
  })
}
