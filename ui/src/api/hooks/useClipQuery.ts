import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../client'
import type { ClipSnapshot } from '../client'

export function useClipQuery(clipId: string | undefined) {
  return useQuery<ClipSnapshot>({
    queryKey: ['clip', clipId],
    queryFn: ({ signal }) => apiClient.getClip(clipId ?? '', { signal }),
    enabled: Boolean(clipId),
    staleTime: 10_000,
  })
}
