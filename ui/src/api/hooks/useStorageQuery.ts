import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { StorageResponse } from '../generated/types'
import { QUERY_KEYS } from './queryKeys'

const STORAGE_REFRESH_MS = 30_000

export function useStorageQuery() {
  return useQuery<StorageResponse>({
    queryKey: QUERY_KEYS.storage,
    queryFn: ({ signal }) => apiClient.getStorage({ signal }),
    staleTime: STORAGE_REFRESH_MS,
    refetchInterval: STORAGE_REFRESH_MS,
  })
}
