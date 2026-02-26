import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { SetupStatusSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'

const SETUP_STATUS_STALE_MS = 60_000

export function useSetupStatusQuery() {
  return useQuery<SetupStatusSnapshot>({
    queryKey: QUERY_KEYS.setupStatus,
    queryFn: ({ signal }) => apiClient.getSetupStatus({ signal }),
    staleTime: SETUP_STATUS_STALE_MS,
    refetchOnWindowFocus: true,
  })
}
