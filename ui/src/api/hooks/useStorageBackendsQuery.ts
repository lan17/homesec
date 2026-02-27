import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { StorageBackendsResponse } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

const STORAGE_BACKENDS_REFRESH_MS = 60_000

export function useStorageBackendsQuery() {
  return useQuery<StorageBackendsResponse>({
    queryKey: QUERY_KEYS.storageBackends,
    queryFn: ({ signal }) => apiClient.listStorageBackends({ signal }),
    staleTime: STORAGE_BACKENDS_REFRESH_MS,
    refetchInterval: STORAGE_BACKENDS_REFRESH_MS,
  })
}
