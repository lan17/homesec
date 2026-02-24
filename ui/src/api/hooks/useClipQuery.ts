import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { ClipSnapshot } from '../client'
import { QUERY_KEYS } from './queryKeys'

export function useClipQuery(clipId: string | undefined) {
  return useQuery<ClipSnapshot>({
    queryKey: QUERY_KEYS.clip(clipId),
    queryFn: ({ signal }) => apiClient.getClip(clipId ?? '', { signal }),
    enabled: Boolean(clipId),
    staleTime: 10_000,
  })
}
